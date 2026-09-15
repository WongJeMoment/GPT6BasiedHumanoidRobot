"""G1 抛物与接箱放架环境；接收外部关节动作，负责物理、观测、奖励和复位。"""
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from .robot import FINGER, make_robot_cfg


@configclass
class ThrowEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 20.0
    action_space = 23
    observation_space = 88
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 120, render_interval=2)
    scene = InteractiveSceneCfg(num_envs=64, env_spacing=8.0, replicate_physics=True)


def make_env_cfg(settings, device="cuda:0", seed=42):
    settings.validate()
    cfg = ThrowEnvCfg()
    cfg.settings = settings
    cfg.seed = seed
    cfg.sim.device = device
    cfg.sim.dt = settings.physics_dt
    cfg.decimation = settings.decimation
    cfg.sim.render_interval = settings.decimation
    cfg.episode_length_s = settings.episode_seconds
    # 放稳任务有实际时限，不能在超时后虚构未来价值进行 bootstrap。
    cfg.is_finite_horizon = settings.shelf_task
    cfg.scene.num_envs = settings.num_envs
    cfg.scene.env_spacing = settings.env_spacing
    cfg.observation_space = 85 + len(settings.objects)
    if settings.shelf_task:
        from .shelf_task import SHELF_OBSERVATIONS
        cfg.observation_space += SHELF_OBSERVATIONS
    cfg.viewer.eye = (4.5, 4.5, 3.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.8)
    if settings.ground_ruler:
        cfg.viewer.eye = (7.0, -7.0, 6.0)
        cfg.viewer.lookat = (2.0, 0.0, 0.5)
    return cfg


class G1ThrowEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.settings = cfg.settings
        super().__init__(cfg, render_mode, **kwargs)
        if self.robot.num_joints != 23 or any(FINGER.fullmatch(n) for n in self.robot.joint_names):
            raise RuntimeError(f"手部零自由度校验失败，实际关节: {self.robot.joint_names}")
        self.actions = torch.zeros((self.num_envs, 23), device=self.device)
        self.active_object = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.last_object = self.active_object.clone()
        self.launch_speed = torch.zeros(self.num_envs, device=self.device)
        self.next_throw = torch.zeros(self.num_envs, device=self.device)
        self.throw_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.indices = torch.arange(self.num_envs, device=self.device)
        self.shelf_task = None
        if self.settings.shelf_task:
            from .shelf_task import ShelfTask
            self.shelf_task = ShelfTask(self.num_envs, self.device, self.settings.shelf, self.settings.objects[0].size)
            self.shelf_hand_ids, _ = self.robot.find_bodies(["left_palm_link", "right_palm_link"], preserve_order=True)
            self.shelf_contact_groups = []
            for side in ("left", "right"):
                self.shelf_contact_groups.append([i + 1 for i, name in enumerate(self.shelf_robot_bodies)
                    if name.startswith(f"{side}_") and any(part in name for part in
                        ("shoulder", "elbow", "palm", "zero", "one", "two", "three", "four", "five", "six"))])
            self.shelf_contact_groups.append([self.shelf_robot_bodies.index("torso_link") + 1])
        self.joint_target = self.robot.data.default_joint_pos.clone()
        self.previous_actions = self.actions.clone()
        self.fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.dropped = torch.zeros_like(self.fallen)
        self.launch_plan = None  # 评估时预采样每个环境的来物，避免策略改变随机数顺序
        self.terminal_status_enabled = False  # 运行入口按需开启，外部动作接口不增加快照开销

    def _setup_scene(self):
        self.robot = Articulation(make_robot_cfg(self.settings))
        self.scene.articulations["robot"] = self.robot
        self.objects = []
        for i, spec in enumerate(self.settings.objects):
            kwargs = dict(
                activate_contact_sensors=self.settings.shelf_task,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8, solver_velocity_iteration_count=2,
                    max_depenetration_velocity=2.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.15
                ),
            )
            if spec.shape == "sphere":
                spawn = sim_utils.SphereCfg(radius=spec.size[0], **kwargs)
            elif spec.shape == "cuboid":
                spawn = sim_utils.CuboidCfg(size=spec.size, **kwargs)
            elif spec.shape == "cylinder":
                spawn = sim_utils.CylinderCfg(radius=spec.size[0], height=spec.size[1], **kwargs)
            else:
                from .stl_assets import prepare_stl, spawn_stl
                material = kwargs.pop("physics_material")
                spawn = sim_utils.UsdFileCfg(usd_path=prepare_stl(spec), func=spawn_stl, **kwargs)
                spawn.throw_physics_material = material
            obj = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Object_{spec.name}", spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -20.0 - i * 2)),
            ))
            self.objects.append(obj)
            self.scene.rigid_objects[spec.name] = obj
        if self.settings.shelf_task:
            from .shelf_task import spawn_shelf
            from isaaclab.sim.utils import get_current_stage
            from pxr import Usd, UsdPhysics
            spawn_shelf(self.scene, self.settings.shelf)
            root = get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")
            bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            self.shelf_robot_bodies = [p.GetName() for p in bodies]
            # 箱体作为单刚体传感器，逐个过滤所有机器人刚体，防止腿/躯干托住也被误判为松手。
            filters = [str(p.GetPath()).replace("/env_0/", "/env_.*/") for p in bodies]
            self.shelf_sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Object_{self.settings.objects[0].name}",
                filter_prim_paths_expr=["/World/envs/env_.*/ShelfTop", *filters], update_period=0.0,
            ))
            self.scene.sensors["shelf_box_contact"] = self.shelf_sensor
        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
        if self.settings.ground_ruler:
            from .ground_ruler import spawn_ground_ruler
            spawn_ground_ruler(self.settings)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2500.0)
        light.func("/World/Light", light)

    def _uniform(self, bounds, count):
        return torch.rand(count, device=self.device) * (bounds[1] - bounds[0]) + bounds[0]

    def _park_objects(self, env_ids):
        for i, obj in enumerate(self.objects):
            state = obj.data.default_root_state[env_ids].clone()
            state[:, :3] = self.scene.env_origins[env_ids]
            state[:, 2] -= 20.0 + 2 * i
            state[:, 7:] = 0
            obj.write_root_state_to_sim(state, env_ids=env_ids)

    def _throw(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        s = self.settings
        self._park_objects(env_ids)
        count = len(self.objects)
        chosen = torch.randint(count, (n,), device=self.device)
        if count > 1:
            previous = self.last_object[env_ids]
            alternate = (previous + torch.randint(1, count, (n,), device=self.device)) % count
            chosen = torch.where(previous >= 0, alternate, chosen)
        speed = self._uniform(s.speed_range, n)
        distance = self._uniform(s.distance_range, n)
        angle = self._uniform(s.azimuth_range, n)
        height = self._uniform(s.launch_height_range, n)
        target_lateral = self._uniform(s.target_lateral_range, n)
        launch_lateral = self._uniform(s.launch_lateral_range, n) if s.launch_lateral_range is not None else None
        if self.launch_plan is not None:
            # 评估计划按环境索引选取，早跌倒的环境不会改变其他环境的发射样本。
            chosen = self.launch_plan["object"][env_ids]
            speed = self.launch_plan["speed"][env_ids]
            distance = self.launch_plan["distance"][env_ids]
            angle = self.launch_plan["angle"][env_ids]
            height = self.launch_plan["height"][env_ids]
            target_lateral = self.launch_plan["target_lateral"][env_ids]
            if s.launch_lateral_range is not None:
                launch_lateral = self.launch_plan["launch_lateral"][env_ids]
        # +X 为机器人前方，瞄准点在当前根位置左右随机偏移。
        offset_x = distance if launch_lateral is not None else distance * torch.cos(angle)
        offset_y = launch_lateral if launch_lateral is not None else distance * torch.sin(angle)
        position = self.robot.data.root_pos_w[env_ids].clone()
        position[:, 0] += offset_x
        position[:, 1] += offset_y
        position[:, 2] = self.scene.env_origins[env_ids, 2] + height
        delta_x = -offset_x
        delta_y = target_lateral - offset_y
        target_distance = torch.sqrt(delta_x.square() + delta_y.square()).clamp_min(1e-6)
        dz = s.target_height - height
        g = abs(self.cfg.sim.gravity[2])
        v2 = speed.square()
        discriminant = v2.square() - g * (g * target_distance.square() + 2 * dz * v2)
        # 低弹道精确满足采样的初速度大小，补偿重力。
        tangent = (v2 - discriminant.clamp_min(0).sqrt()) / (g * target_distance)
        horizontal_speed = speed / torch.sqrt(1 + tangent.square())
        velocity = torch.stack((
            horizontal_speed * delta_x / target_distance,
            horizontal_speed * delta_y / target_distance, horizontal_speed * tangent,
        ), dim=-1)
        for i, obj in enumerate(self.objects):
            mask = chosen == i
            ids = env_ids[mask]
            if len(ids):
                state = obj.data.default_root_state[ids].clone()
                state[:, :3] = position[mask]
                state[:, 7:10] = velocity[mask]
                state[:, 10:] = 0
                obj.write_root_state_to_sim(state, env_ids=ids)
        self.active_object[env_ids] = chosen
        self.last_object[env_ids] = chosen
        self.launch_speed[env_ids] = speed
        self.throw_count[env_ids] += 1
        if self.shelf_task is not None:
            self.shelf_task.reset(env_ids)
        self.next_throw[env_ids] = (
            self.episode_length_buf[env_ids] * self.step_dt + self._uniform(s.interval_range, n)
            if s.continuous else float("inf")
        )

    def _pre_physics_step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clamp(-1.0, 1.0)
        ids = (self.episode_length_buf * self.step_dt >= self.next_throw).nonzero().flatten()
        self._throw(ids)
        if self.shelf_task is not None:
            # 本环境只接受关节目标，不内置接住后放架的运动策略。
            limits = self.robot.data.soft_joint_pos_limits
            desired = (self.robot.data.default_joint_pos + self.settings.action_scale * self.actions).clamp(
                limits[..., 0], limits[..., 1])
            limit = self.settings.shelf.joint_speed * self.step_dt
            self.joint_target += (desired - self.joint_target).clamp(-limit, limit)

    def _apply_action(self):
        # 只停放备用刚体；抛出、接触或掉落的当前物体始终由 PhysX 自由积分。
        from .object_pool import park_inactive_objects
        park_inactive_objects(self.objects, self.scene.env_origins, self.active_object)
        target = (self.joint_target if self.shelf_task is not None else
                  self.robot.data.default_joint_pos + self.settings.action_scale * self.actions)
        limits = self.robot.data.soft_joint_pos_limits
        self.robot.set_joint_position_target(target.clamp(limits[..., 0], limits[..., 1]))

    def _get_observations(self):
        states = torch.stack([o.data.root_state_w for o in self.objects], dim=1)
        state = states[self.indices, self.active_object.clamp_min(0)]
        active = (self.active_object >= 0).unsqueeze(-1)
        relative = torch.where(active, state[:, :3] - self.robot.data.root_pos_w, 0.0)
        velocity = torch.where(active, state[:, 7:10], 0.0)
        kind = torch.nn.functional.one_hot(self.active_object.clamp_min(0), len(self.objects)) * active
        data = self.robot.data
        obs = torch.cat((
            data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
            data.joint_pos - data.default_joint_pos, data.joint_vel * 0.1,
            self.actions, relative, velocity, kind, self.launch_speed.unsqueeze(-1),
        ), dim=-1)
        if self.shelf_task is not None:
            from .shelf_task import box_geometry
            task = self.shelf_task
            local = state[:, :3] - self.scene.env_origins
            extent, _ = box_geometry(state[:, 3:7], task.half_size)
            goal_error = torch.where(active, task.goal - local, 0.0)
            obs = torch.cat((obs,
                torch.where(active, state[:, 3:7], 0.0), torch.where(active, state[:, 10:13], 0.0),
                torch.where(active, state[:, 7:10] - data.root_lin_vel_w, 0.0),
                task.surface + self.scene.env_origins - data.root_pos_w,
                task.size.expand(self.num_envs, -1), goal_error, task.contact.float(),
                task.robot_touch.float()[:, None], task.supported.float()[:, None], task.caught.float()[:, None],
                torch.nn.functional.one_hot(task.phase, 5).float(),
                (task.catch_time / task.cfg.catch_seconds)[:, None],
                (task.settle_time / task.cfg.settle_seconds)[:, None],
                (1 - self.episode_length_buf * self.step_dt / self.settings.episode_seconds).clamp(0, 1)[:, None],
                self.joint_target - data.default_joint_pos,
                (data.root_pos_w[:, 2] - self.scene.env_origins[:, 2])[:, None],
                torch.where(active, extent, 0.0), data.root_quat_w,
                task.best_distance[:, None], task.shelf_first.float()[:, None],
            ), dim=-1)
        return {"policy": obs}

    def _get_rewards(self):
        if self.terminal_status_enabled:
            from .terminal_status import capture_terminal_state
            # DirectRLEnv 在计算奖励后自动复位；必须在此保留结束步的实际状态。
            self.extras["terminal"] = capture_terminal_state(self)
        s, data = self.settings, self.robot.data
        if self.shelf_task is not None:
            task = self.shelf_task
            position = self.objects[0].data.root_pos_w
            palms = data.body_pos_w[:, self.shelf_hand_ids]
            distance = (palms - position[:, None]).norm(dim=-1).mean(-1)
            upright = data.projected_gravity_b[:, 2] < -0.85
            reward = task.reward(self.active_object >= 0, torch.exp(-distance.square() / .16), upright,
                                 self.fallen, self.reset_time_outs, self.actions,
                                 self.actions - self.previous_actions,
                                 self.episode_length_buf * self.step_dt - s.first_throw_delay, self.step_dt)
            # 必须复制终止步快照：DirectRLEnv 会在返回 step 前局部复位。
            self.extras["shelf"] = {name: getattr(task, name).clone() for name in
                ("phase", "caught", "success", "contact", "on_shelf", "supported", "robot_touch", "shelf_first", "catch_time", "settle_time")}
            self.extras["shelf"].update(fallen=self.fallen.clone(), dropped=self.dropped.clone())
            self.extras["log"] = {}
            return reward
        reward = (
            s.upright_reward * (-data.projected_gravity_b[:, 2]).clamp(0, 1)
            - s.posture_penalty * (data.joint_pos - data.default_joint_pos).square().sum(-1)
            - s.action_penalty * self.actions.square().sum(-1)
            - s.velocity_penalty * data.root_lin_vel_b.square().sum(-1)
            - s.fall_penalty * self.reset_terminated.float()
        ) * self.step_dt
        return reward

    def _get_dones(self):
        fallen = (self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
                  < self.settings.minimum_base_height)
        fallen |= self.robot.data.projected_gravity_b[:, 2] > -0.5
        self.fallen.copy_(fallen)
        self.dropped.zero_()
        if self.shelf_task is not None:
            task, data = self.shelf_task, self.robot.data
            state = self.objects[0].data.root_state_w
            forces = self.shelf_sensor.data.force_matrix_w
            if forces is None or forces.shape[2] != 1 + len(self.shelf_robot_bodies):
                raise RuntimeError("放架任务需要箱体对架面和各机器人刚体的独立接触力")
            force = forces[:, 0].norm(dim=-1)
            contact = torch.stack([force[:, group].sum(-1) > task.cfg.contact_force
                                   for group in self.shelf_contact_groups], -1)
            robot_touch = (force[:, 1:] > task.cfg.contact_force).any(-1)
            shelf_touch = force[:, 0] > task.cfg.contact_force
            weight = self.settings.objects[0].mass * abs(self.cfg.sim.gravity[2])
            supported = forces[:, 0, 0, 2] > weight * task.cfg.support_weight_fraction
            task.update(self.active_object >= 0, state[:, :3] - self.scene.env_origins, state[:, 3:7],
                        state[:, 7:10], state[:, 10:13], data.root_pos_w - self.scene.env_origins,
                        data.root_lin_vel_w, contact, robot_touch, shelf_touch, supported,
                        data.projected_gravity_b[:, 2] < -0.85, fallen, self.step_dt)
            self.dropped.copy_(task.dropped)
            terminated = fallen | self.dropped | task.success
            # 成功或物理失败优先；成功步不同时标记 timeout。
            timeout = (self.episode_length_buf >= self.max_episode_length - 1) & ~terminated
            return terminated, timeout
        return fallen | self.dropped, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if self.shelf_task is not None and len(env_ids):
            self.extras["log"] = {
                "Episode/place_success": self.shelf_task.success[env_ids].float().mean(),
                "Episode/box_caught": self.shelf_task.caught[env_ids].float().mean(),
                "Episode/fall": self.fallen[env_ids].float().mean(),
                "Episode/drop": self.dropped[env_ids].float().mean(),
            }
            self.shelf_task.reset(env_ids)
        super()._reset_idx(env_ids)
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids], self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.fallen[env_ids] = False
        self.dropped[env_ids] = False
        self.joint_target[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.active_object[env_ids] = -1
        # 保留 last_object，让跨回合的相邻抛掷也不重复物体种类。
        self.launch_speed[env_ids] = 0
        self.next_throw[env_ids] = self.settings.first_throw_delay
        self.throw_count[env_ids] = 0
        self._park_objects(env_ids)
