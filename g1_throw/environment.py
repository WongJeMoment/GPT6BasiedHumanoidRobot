"""DirectRLEnv：支持原站立任务，以及独立 controllers 目录的分层抱接。"""
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
    cfg.scene.num_envs = settings.num_envs
    cfg.scene.env_spacing = settings.env_spacing
    cfg.observation_space = 85 + len(settings.objects)
    cfg.viewer.eye = (4.5, 4.5, 3.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.8)
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
        self.catch_controller = None
        if self.settings.controller == "hierarchical":
            from controllers.hierarchical import HierarchicalCatchController
            self.catch_controller = HierarchicalCatchController(self)
        self.joint_target = self.robot.data.default_joint_pos.clone()

    def _setup_scene(self):
        self.robot = Articulation(make_robot_cfg(self.settings))
        self.scene.articulations["robot"] = self.robot
        self.objects = []
        for i, spec in enumerate(self.settings.objects):
            kwargs = dict(
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
        self.catch_sensors = []
        if self.settings.controller == "hierarchical":
            # 每个传感器仅绑定一个刚体，按物体过滤，避免将落地接触误报为接球。
            for side in ("left", "right"):
                hand_sensors = []
                # 固定手指仍有独立碰撞几何，必须覆盖它们，不能只检测掌部。
                for part in ("palm", "zero", "one", "two", "three", "four", "five", "six"):
                    sensor = ContactSensor(ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{side}_{part}_link",
                        filter_prim_paths_expr=[f"/World/envs/env_.*/Object_{s.name}" for s in self.settings.objects],
                        update_period=0.0,
                    ))
                    hand_sensors.append(sensor)
                    self.scene.sensors[f"{side}_{part}_catch"] = sensor
                self.catch_sensors.append(hand_sensors)
        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
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
        # +X 为机器人前方。向当前根位置的水平坐标、配置的目标高度发射。
        position = self.robot.data.root_pos_w[env_ids].clone()
        position[:, 0] += distance * torch.cos(angle)
        position[:, 1] += distance * torch.sin(angle)
        position[:, 2] = self.scene.env_origins[env_ids, 2] + height
        dz = s.target_height - height
        g = abs(self.cfg.sim.gravity[2])
        v2 = speed.square()
        discriminant = v2.square() - g * (g * distance.square() + 2 * dz * v2)
        # 低弹道精确满足采样的初速度大小，补偿重力。
        tangent = (v2 - discriminant.clamp_min(0).sqrt()) / (g * distance)
        horizontal_speed = speed / torch.sqrt(1 + tangent.square())
        velocity = torch.stack((
            -horizontal_speed * torch.cos(angle),
            -horizontal_speed * torch.sin(angle), horizontal_speed * tangent,
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
        if self.catch_controller is not None:
            # 连续抛掷即新任务；清除上一物体的成功标志与阶段计时。
            self.catch_controller.planner.reset(env_ids)
        self.next_throw[env_ids] = (
            self.episode_length_buf[env_ids] * self.step_dt + self._uniform(s.interval_range, n)
            if s.continuous else float("inf")
        )

    def _pre_physics_step(self, actions):
        self.actions = actions.clamp(-1.0, 1.0)
        ids = (self.episode_length_buf * self.step_dt >= self.next_throw).nonzero().flatten()
        self._throw(ids)
        if self.catch_controller is not None:
            nominal = self.catch_controller.compute()
            desired = nominal + self.settings.catch_control.residual_scale * self.actions
            limit = self.settings.catch_control.joint_speed * self.step_dt
            self.joint_target += (desired - self.joint_target).clamp(-limit, limit)

    def _apply_action(self):
        target = (self.joint_target if self.catch_controller is not None else
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
        return {"policy": torch.cat((
            data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
            data.joint_pos - data.default_joint_pos, data.joint_vel * 0.1,
            self.actions, relative, velocity, kind, self.launch_speed.unsqueeze(-1),
        ), dim=-1)}

    def _get_rewards(self):
        s, data = self.settings, self.robot.data
        reward = (
            s.upright_reward * (-data.projected_gravity_b[:, 2]).clamp(0, 1)
            - s.posture_penalty * (data.joint_pos - data.default_joint_pos).square().sum(-1)
            - s.action_penalty * self.actions.square().sum(-1)
            - s.velocity_penalty * data.root_lin_vel_b.square().sum(-1)
            - s.fall_penalty * self.reset_terminated.float()
        ) * self.step_dt
        if self.catch_controller is not None:
            planner = self.catch_controller.planner
            # 抱持奖励只在持续双侧接触且低速时产生；success 为本次抛掷锁存指标。
            reward += 8.0 * (planner.stable_time > 0).float() * self.step_dt
            self.extras["catch"] = {"phase": planner.phase.clone(), "success": planner.success.clone()}
        return reward

    def _get_dones(self):
        fallen = (self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
                  < self.settings.minimum_base_height)
        fallen |= self.robot.data.projected_gravity_b[:, 2] > -0.5
        return fallen, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids], self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )
        self.actions[env_ids] = 0
        if self.catch_controller is not None:
            self.catch_controller.reset(env_ids)
        self.joint_target[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.active_object[env_ids] = -1
        # 保留 last_object，让跨回合的相邻抛掷也不重复物体种类。
        self.launch_speed[env_ids] = 0
        self.next_throw[env_ids] = self.settings.first_throw_delay
        self.throw_count[env_ids] = 0
        self._park_objects(env_ids)
