"""与 Isaac Lab 场景同规则的 CPU MuJoCo 环境，支持批量 step 与局部自动复位。"""
from copy import deepcopy
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from g1_throw.shelf_task import SHELF_OBSERVATIONS, ShelfTask, box_geometry
from g1_throw.terminal_status import capture_terminal_state
from .model import JOINT_NAMES, build_model, default_pose


class G1MujocoEnv:
    """每个环境使用独立 MjData，共享只读结构；输出 CPU torch 张量。

    step 返回 (obs, reward, terminated, truncated, info)，结束步信息在复位前复制。
    批量物理按环境依次执行，不提供 Isaac Lab 的 GPU 并行吞吐。
    """

    def __init__(self, settings=None, num_envs=1, seed=42):
        if settings is None:
            from env_configs.catch_and_place import CONFIG
            settings = CONFIG
        self.settings = deepcopy(settings)
        self.settings.num_envs = num_envs
        self.settings.validate()
        if self.settings.robot_usd is not None:
            raise ValueError("MuJoCo 使用本地 G1 MJCF，不能加载 robot_usd；请将该项设为 None")
        self.num_envs, self.device = num_envs, "cpu"
        self.seed = seed
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.model, self.xml = build_model(self.settings)
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.data = self.datas[0]  # 预览默认只显示环境 0。
        self.step_dt = self.settings.physics_dt * self.settings.decimation
        self.max_episode_length = math.ceil(self.settings.episode_seconds / self.step_dt)
        self.common_step_counter = 0
        self.action_dim = 23
        self.observation_dim = 85 + len(self.settings.objects) + (SHELF_OBSERVATIONS if self.settings.shelf_task else 0)
        self.cfg = SimpleNamespace(observation_space=self.observation_dim, action_space=23)
        self.indices = torch.arange(num_envs)
        self.active_object = torch.full((num_envs,), -1, dtype=torch.long)
        self.last_object = self.active_object.clone()
        self.throw_count = torch.zeros(num_envs, dtype=torch.long)
        self.total_throws = torch.zeros_like(self.throw_count)
        self.episode_length_buf = torch.zeros_like(self.throw_count)
        self.launch_speed = torch.zeros(num_envs)
        self.next_throw = torch.zeros(num_envs)
        self.actions = torch.zeros(num_envs, 23)
        self.previous_actions = torch.zeros_like(self.actions)
        self.fallen = torch.zeros(num_envs, dtype=torch.bool)
        self.dropped = torch.zeros_like(self.fallen)
        self.reset_terminated = torch.zeros_like(self.fallen)
        self.reset_time_outs = torch.zeros_like(self.fallen)
        self.launch_plan = None
        self.terminal_status_enabled = True
        self.extras = {}
        self.closed = False
        self.joint_ids = np.array([self.model.joint(name).id for name in JOINT_NAMES])
        self.joint_qpos = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofs = self.model.jnt_dofadr[self.joint_ids]
        limits = self.model.jnt_range[self.joint_ids]
        mid, half = limits.mean(axis=1), (limits[:, 1] - limits[:, 0]) * .45
        self.soft_limits = torch.tensor(np.stack((mid - half, mid + half), -1), dtype=torch.float32)
        pose = torch.tensor(default_pose(self.settings), dtype=torch.float32).repeat(num_envs, 1)
        self.joint_target = pose.clone()
        self.root_id = self.model.body("pelvis").id
        self.root_joint = self.model.joint("floating_base_joint").id
        self.object_body_ids = [self.model.body(f"Object_{spec.name}").id for spec in self.settings.objects]
        self.object_geom_ids = [self.model.geom(f"Object_{spec.name}").id for spec in self.settings.objects]
        self.object_joint_ids = [self.model.joint(f"Object_{spec.name}_joint").id for spec in self.settings.objects]
        self.robot_body_ids = [i for i in range(1, self.model.nbody)
                               if self.model.body_rootid[i] == self.root_id]
        self.shelf_robot_bodies = [self.model.body(i).name for i in self.robot_body_ids]
        self.robot_body_columns = {body: i + 1 for i, body in enumerate(self.robot_body_ids)}
        self.shelf_contact_groups = [
            [i + 1 for i, name in enumerate(self.shelf_robot_bodies)
             if name.startswith(side + "_") and any(part in name for part in
                 ("shoulder", "elbow", "palm", "zero", "one", "two", "three", "four", "five", "six"))]
            for side in ("left", "right")
        ]
        self.shelf_contact_groups.append([self.shelf_robot_bodies.index("torso_link") + 1])
        self.shelf_hand_ids = [self.shelf_robot_bodies.index(f"{side}_palm_link") for side in ("left", "right")]
        self.scene = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
        # 将 MuJoCo 数据映射成共用终端记录器所需的字段；这里不加载 Isaac Lab。
        root = torch.zeros(num_envs, 13)
        self.robot = SimpleNamespace(num_joints=23, joint_names=list(JOINT_NAMES), data=SimpleNamespace(
            default_joint_pos=pose, root_state_w=root, root_pos_w=root[:, :3], root_quat_w=root[:, 3:7],
            root_lin_vel_w=root[:, 7:10], root_ang_vel_w=root[:, 10:13],
            root_lin_vel_b=torch.zeros(num_envs, 3), root_ang_vel_b=torch.zeros(num_envs, 3),
            projected_gravity_b=torch.zeros(num_envs, 3), joint_pos=pose.clone(), joint_vel=torch.zeros_like(pose),
            body_pos_w=torch.zeros(num_envs, len(self.robot_body_ids), 3),
        ))
        self.objects = []
        for _ in self.settings.objects:
            state = torch.zeros(num_envs, 13)
            self.objects.append(SimpleNamespace(data=SimpleNamespace(root_state_w=state,
                                root_pos_w=state[:, :3])))
        self.shelf_task = (ShelfTask(num_envs, self.device, self.settings.shelf, self.settings.objects[0].size)
                           if self.settings.shelf_task else None)
        self.shelf_top_geom = self.model.geom("ShelfTop").id if self.shelf_task is not None else -1
        self.shelf_sensor = SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.zeros(num_envs, 1, 1 + len(self.robot_body_ids), 3)))
        self.reset()

    def _ids(self, env_ids):
        ids = self.indices if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long).flatten()
        if ((ids < 0) | (ids >= self.num_envs)).any():
            raise IndexError("env_ids 超出环境范围")
        return ids

    def _uniform(self, bounds, count):
        return torch.rand(count, generator=self.generator) * (bounds[1] - bounds[0]) + bounds[0]

    def _set_object_state(self, index, object_index, position, quaternion=(1., 0., 0., 0.),
                          velocity=(0., 0., 0.), angular_velocity=(0., 0., 0.)):
        data = self.datas[index]
        joint = self.object_joint_ids[object_index]
        q, v = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        data.qpos[q:q + 3], data.qpos[q + 3:q + 7] = position, quaternion
        data.qvel[v:v + 3] = velocity
        # MuJoCo freejoint 的平移速度在世界坐标，角速度在局部坐标。
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
        data.qvel[v + 3:v + 6] = rotation.reshape(3, 3).T @ np.asarray(angular_velocity)

    def _park_inactive(self, index):
        for j in range(len(self.objects)):
            if j != int(self.active_object[index]):
                self._set_object_state(index, j, (0., 0., -20. - 2 * j))

    def _select_collisions(self, index):
        # 共用模型的碰撞掩码按当前 MjData 选择；物理循环串行，禁止并发 step。
        # 地下备用物体必须禁用碰撞，否则无限平面会把它判为严重穿透。
        for j, geom in enumerate(self.object_geom_ids):
            active = j == int(self.active_object[index])
            self.model.geom_contype[geom] = 4 if active else 0
            self.model.geom_conaffinity[geom] = 3 if active else 0

    def _body_state(self, data, body):
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
        return np.concatenate((data.xpos[body], data.xquat[body], velocity[3:], velocity[:3]))

    def _read_state(self, index):
        data, target = self.datas[index], self.robot.data
        target.root_state_w[index] = torch.from_numpy(self._body_state(data, self.root_id)).float()
        rotation = data.xmat[self.root_id].reshape(3, 3)
        target.root_lin_vel_b[index] = torch.from_numpy(rotation.T @ target.root_lin_vel_w[index].numpy()).float()
        target.root_ang_vel_b[index] = torch.from_numpy(rotation.T @ target.root_ang_vel_w[index].numpy()).float()
        target.projected_gravity_b[index] = torch.from_numpy(rotation.T @ np.array([0., 0., -1.])).float()
        target.joint_pos[index] = torch.from_numpy(data.qpos[self.joint_qpos]).float()
        target.joint_vel[index] = torch.from_numpy(data.qvel[self.joint_dofs]).float()
        target.body_pos_w[index] = torch.from_numpy(data.xpos[self.robot_body_ids]).float()
        for obj, body in zip(self.objects, self.object_body_ids):
            obj.data.root_state_w[index] = torch.from_numpy(self._body_state(data, body)).float()
        matrix = self.shelf_sensor.data.force_matrix_w[index, 0]
        matrix.zero_()
        if self.shelf_task is None or self.active_object[index] < 0:
            return
        box_geom = self.object_geom_ids[0]
        force = np.empty(6)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if box_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom1 if contact.geom2 == box_geom else contact.geom2
            column = 0 if other == self.shelf_top_geom else self.robot_body_columns.get(self.model.geom_bodyid[other])
            if column is None:
                continue
            mujoco.mj_contactForce(self.model, data, contact_index, force)
            # 法向从 geom1 指向 geom2，返回的是施加在 geom2 上的力。
            force_on_box = contact.frame.reshape(3, 3).T @ force[:3]
            if contact.geom1 == box_geom:
                force_on_box *= -1
            matrix[column] += torch.from_numpy(force_on_box).float()

    def forward(self, env_ids=None):
        """外部测试夹具写 qpos/qvel 后同步运动学、真实接触和观测字段。"""
        for index in self._ids(env_ids).tolist():
            self._select_collisions(index)
            mujoco.mj_forward(self.model, self.datas[index])
            self._read_state(index)

    def reset(self, seed=None, env_ids=None):
        if seed is not None:
            self.seed = seed
            self.generator.manual_seed(seed)
            self.last_object[:] = -1
        ids = self._ids(env_ids)
        self._reset_idx(ids)
        self.reset_terminated[ids] = False
        self.reset_time_outs[ids] = False
        self.extras = {}
        return self._get_observations(), self.extras

    def _reset_idx(self, env_ids):
        if self.shelf_task is not None:
            self.shelf_task.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.active_object[env_ids] = -1
        self.throw_count[env_ids] = 0
        self.launch_speed[env_ids] = 0
        self.next_throw[env_ids] = self.settings.first_throw_delay
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.fallen[env_ids] = False
        self.dropped[env_ids] = False
        self.joint_target[env_ids] = self.robot.data.default_joint_pos[env_ids]
        for index in env_ids.tolist():
            data = self.datas[index]
            mujoco.mj_resetData(self.model, data)
            data.qpos[self.joint_qpos] = self.robot.data.default_joint_pos[index].numpy()
            data.ctrl[:] = self.joint_target[index].numpy()
            self._park_inactive(index)
        self.forward(env_ids)

    def _throw(self, env_ids):
        env_ids = self._ids(env_ids)
        count = len(env_ids)
        if not count:
            return
        settings = self.settings
        object_count = len(self.objects)
        chosen = torch.randint(object_count, (count,), generator=self.generator)
        if object_count > 1:
            previous = self.last_object[env_ids]
            alternate = (previous + torch.randint(1, object_count, (count,), generator=self.generator)) % object_count
            chosen = torch.where(previous >= 0, alternate, chosen)
        sample = {name: self._uniform(bounds, count) for name, bounds in (
            ("speed", settings.speed_range), ("distance", settings.distance_range),
            ("angle", settings.azimuth_range), ("height", settings.launch_height_range),
            ("target_lateral", settings.target_lateral_range),
        )}
        if settings.launch_lateral_range is not None:
            sample["launch_lateral"] = self._uniform(settings.launch_lateral_range, count)
        if self.launch_plan is not None:
            chosen = self.launch_plan["object"][env_ids].cpu()
            sample = {name: self.launch_plan[name][env_ids].cpu() for name in sample}
        speed, distance, angle = (sample[name] for name in ("speed", "distance", "angle"))
        x = distance if "launch_lateral" in sample else distance * angle.cos()
        y = sample["launch_lateral"] if "launch_lateral" in sample else distance * angle.sin()
        position = self.robot.data.root_pos_w[env_ids].clone()
        position[:, 0] += x
        position[:, 1] += y
        position[:, 2] = sample["height"]
        dx, dy = -x, sample["target_lateral"] - y
        horizontal_distance = torch.sqrt(dx.square() + dy.square()).clamp_min(1e-6)
        dz = settings.target_height - sample["height"]
        gravity = abs(float(self.model.opt.gravity[2]))
        v2 = speed.square()
        discriminant = v2.square() - gravity * (gravity * horizontal_distance.square() + 2 * dz * v2)
        tangent = (v2 - discriminant.clamp_min(0).sqrt()) / (gravity * horizontal_distance)
        horizontal_speed = speed / torch.sqrt(1 + tangent.square())
        velocity = torch.stack((horizontal_speed * dx / horizontal_distance,
                                horizontal_speed * dy / horizontal_distance, horizontal_speed * tangent), -1)
        self.active_object[env_ids] = chosen
        self.last_object[env_ids] = chosen
        self.launch_speed[env_ids] = speed
        self.throw_count[env_ids] += 1
        self.total_throws[env_ids] += 1
        if self.shelf_task is not None:
            self.shelf_task.reset(env_ids)
        for offset, index in enumerate(env_ids.tolist()):
            self._park_inactive(index)
            self._set_object_state(index, int(chosen[offset]), position[offset].numpy(), velocity=velocity[offset].numpy())
        self.next_throw[env_ids] = (self.episode_length_buf[env_ids] * self.step_dt
                                   + self._uniform(settings.interval_range, count) if settings.continuous else float("inf"))
        self.forward(env_ids)

    def _get_dones(self):
        data = self.robot.data
        self.fallen.copy_((data.root_pos_w[:, 2] < self.settings.minimum_base_height)
                          | (data.projected_gravity_b[:, 2] > -.5))
        self.dropped.zero_()
        if self.shelf_task is not None:
            task = self.shelf_task
            state = self.objects[0].data.root_state_w
            forces = self.shelf_sensor.data.force_matrix_w[:, 0]
            magnitude = forces.norm(dim=-1)
            contact = torch.stack([magnitude[:, group].sum(-1) > task.cfg.contact_force
                                   for group in self.shelf_contact_groups], -1)
            robot_touch = (magnitude[:, 1:] > task.cfg.contact_force).any(-1)
            shelf_touch = magnitude[:, 0] > task.cfg.contact_force
            weight = self.settings.objects[0].mass * abs(float(self.model.opt.gravity[2]))
            supported = forces[:, 0, 2] > weight * task.cfg.support_weight_fraction
            task.update(self.active_object >= 0, state[:, :3], state[:, 3:7], state[:, 7:10], state[:, 10:13],
                        data.root_pos_w, data.root_lin_vel_w, contact, robot_touch, shelf_touch, supported,
                        data.projected_gravity_b[:, 2] < -.85, self.fallen, self.step_dt)
            self.dropped.copy_(task.dropped)
            ended = self.fallen | self.dropped | task.success
            return ended, (self.episode_length_buf >= self.max_episode_length - 1) & ~ended
        return self.fallen.clone(), self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        settings, data, task = self.settings, self.robot.data, self.shelf_task
        if task is not None:
            palms = data.body_pos_w[:, self.shelf_hand_ids]
            distance = (palms - self.objects[0].data.root_pos_w[:, None]).norm(dim=-1).mean(-1)
            return task.reward(self.active_object >= 0, torch.exp(-distance.square() / .16),
                               data.projected_gravity_b[:, 2] < -.85, self.fallen, self.reset_time_outs,
                               self.actions, self.actions - self.previous_actions,
                               self.episode_length_buf * self.step_dt - settings.first_throw_delay, self.step_dt)
        return (settings.upright_reward * (-data.projected_gravity_b[:, 2]).clamp(0, 1)
                - settings.posture_penalty * (data.joint_pos - data.default_joint_pos).square().sum(-1)
                - settings.action_penalty * self.actions.square().sum(-1)
                - settings.velocity_penalty * data.root_lin_vel_b.square().sum(-1)
                - settings.fall_penalty * self.reset_terminated.float()) * self.step_dt

    def _get_observations(self):
        states = torch.stack([obj.data.root_state_w for obj in self.objects], 1)
        state = states[self.indices, self.active_object.clamp_min(0)]
        active = (self.active_object >= 0).unsqueeze(-1)
        data = self.robot.data
        obs = torch.cat((data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
                         data.joint_pos - data.default_joint_pos, data.joint_vel * .1, self.actions,
                         torch.where(active, state[:, :3] - data.root_pos_w, 0.),
                         torch.where(active, state[:, 7:10], 0.),
                         torch.nn.functional.one_hot(self.active_object.clamp_min(0), len(self.objects)) * active,
                         self.launch_speed[:, None]), -1)
        task = self.shelf_task
        if task is not None:
            extent, _ = box_geometry(state[:, 3:7], task.half_size)
            obs = torch.cat((obs, torch.where(active, state[:, 3:7], 0.), torch.where(active, state[:, 10:13], 0.),
                torch.where(active, state[:, 7:10] - data.root_lin_vel_w, 0.),
                task.surface - data.root_pos_w, task.size.expand(self.num_envs, -1),
                torch.where(active, task.goal - state[:, :3], 0.), task.contact.float(),
                task.robot_touch.float()[:, None], task.supported.float()[:, None], task.caught.float()[:, None],
                torch.nn.functional.one_hot(task.phase, 5).float(),
                (task.catch_time / task.cfg.catch_seconds)[:, None],
                (task.settle_time / task.cfg.settle_seconds)[:, None],
                (1 - self.episode_length_buf * self.step_dt / self.settings.episode_seconds).clamp(0, 1)[:, None],
                self.joint_target - data.default_joint_pos, data.root_pos_w[:, 2:3],
                torch.where(active, extent, 0.), data.root_quat_w, task.best_distance[:, None],
                task.shelf_first.float()[:, None]), -1)
        return {"policy": obs}

    def step(self, actions):
        if self.closed:
            raise RuntimeError("环境已经关闭")
        actions = torch.as_tensor(actions, dtype=torch.float32, device="cpu").detach()
        if actions.shape == (23,) and self.num_envs == 1:
            actions = actions.unsqueeze(0)
        if actions.shape != (self.num_envs, 23) or not torch.isfinite(actions).all():
            raise ValueError(f"actions 必须是有限的 ({self.num_envs}, 23) 张量")
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1, 1))
        self._throw((self.episode_length_buf * self.step_dt >= self.next_throw).nonzero().flatten())
        desired = (self.robot.data.default_joint_pos + self.settings.action_scale * self.actions).clamp(
            self.soft_limits[:, 0], self.soft_limits[:, 1])
        if self.shelf_task is not None:
            limit = self.settings.shelf.joint_speed * self.step_dt
            self.joint_target += (desired - self.joint_target).clamp(-limit, limit)
        else:
            self.joint_target.copy_(desired)
        for index, data in enumerate(self.datas):
            self._select_collisions(index)
            data.ctrl[:] = self.joint_target[index].numpy()
            for _ in range(self.settings.decimation):
                self._park_inactive(index)
                mujoco.mj_step(self.model, data)
            self._park_inactive(index)
            mujoco.mj_forward(self.model, data)
            self._read_state(index)
        self.common_step_counter += 1
        self.episode_length_buf += 1
        ended, timeout = self._get_dones()
        self.reset_terminated.copy_(ended)
        self.reset_time_outs.copy_(timeout)
        reward = self._get_rewards()
        self.extras = {"log": {}}
        if self.terminal_status_enabled:
            self.extras["terminal"] = capture_terminal_state(self)
        if self.shelf_task is not None:
            self.extras["shelf"] = {name: getattr(self.shelf_task, name).clone() for name in (
                "phase", "caught", "success", "contact", "on_shelf", "supported", "robot_touch",
                "shelf_first", "catch_time", "settle_time",
            )}
            self.extras["shelf"].update(fallen=self.fallen.clone(), dropped=self.dropped.clone())
        ids = (ended | timeout).nonzero().flatten()
        if len(ids):
            if self.shelf_task is not None:
                self.extras["log"] = {f"Episode/{label}": value[ids].float().mean() for label, value in (
                    ("place_success", self.shelf_task.success), ("box_caught", self.shelf_task.caught),
                    ("fall", self.fallen), ("drop", self.dropped))}
            self._reset_idx(ids)
        return self._get_observations(), reward, ended.clone(), timeout.clone(), self.extras

    def close(self):
        self.closed = True
