"""Isaac Lab 适配层：状态估计 → 高层规划 → 低层关节目标。"""
import torch
from .planner import CatchPlanner
from .whole_body import WholeBodyController, rotate_yaw


class HierarchicalCatchController:
    def __init__(self, env):
        self.env = env
        self.cfg = env.settings.catch_control
        self.planner = CatchPlanner(env.num_envs, env.device, self.cfg)
        self.motor = WholeBodyController(env.robot, self.cfg)
        # 使用外接半径保守估计接物开口；STL 只给最长边，形状不确定性更大。
        radii = []
        for spec in env.settings.objects:
            radius = spec.size[0] if spec.shape in ("sphere", "cylinder") else max(spec.size) / 2
            radii.append(radius)
        self.radii = torch.tensor(radii, device=env.device)

    def reset(self, ids):
        self.planner.reset(ids)
        self.motor.reset(ids)

    def compute(self):
        env, data = self.env, self.env.robot.data
        states = torch.stack([obj.data.root_state_w for obj in env.objects], 1)
        selected = env.active_object.clamp_min(0)
        state = states[env.indices, selected]
        w, x, y, z = data.root_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        position = rotate_yaw(state[:, :3] - data.root_pos_w, yaw, inverse=True)
        velocity = rotate_yaw(state[:, 7:10] - data.root_lin_vel_w, yaw, inverse=True)
        gravity = torch.tensor(env.cfg.sim.gravity, device=env.device).expand_as(position)
        contacts = []
        for hand_sensors in env.catch_sensors:
            hand_force = torch.zeros(env.num_envs, device=env.device)
            for sensor in hand_sensors:
                forces = sensor.data.force_matrix_w
                if forces is None:
                    raise RuntimeError("接物传感器缺少物体过滤接触力")
                hand_force += forces[env.indices, 0, selected].norm(dim=-1)
            contacts.append(hand_force > self.cfg.contact_force)
        contacts = torch.stack(contacts, -1)
        upright = ((data.projected_gravity_b[:, 2] < -0.85)
                   & (data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] > env.settings.minimum_base_height))
        prediction, phase = self.planner.update(position, velocity, gravity, env.active_object >= 0,
                                                 contacts, upright, env.step_dt)
        return self.motor.compute(prediction, position, velocity, phase, self.planner.elapsed,
                                  self.radii[selected], yaw, env.step_dt)
