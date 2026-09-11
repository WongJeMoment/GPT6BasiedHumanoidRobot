"""Isaac Lab 适配层：状态估计 → 高层规划 → 低层关节目标。"""
import torch
from .planner import CatchPlanner
from .whole_body import WholeBodyController, rotate_yaw


class HierarchicalCatchController:
    def __init__(self, env):
        self.env = env
        self.cfg = env.settings.catch_control
        self.planner = CatchPlanner(env.num_envs, env.device, self.cfg)
        self.motor = WholeBodyController(env.robot, self.cfg, env.settings.active_legs)
        # 区分前后厚度、左右宽度和高度，箱体不能只用一个半径确定胸前距离。
        half_sizes, shapes = [], []
        for spec in env.settings.objects:
            if spec.shape == "sphere":
                half_sizes.append([spec.size[0]] * 3)
            elif spec.shape == "cylinder":
                half_sizes.append([spec.size[0], spec.size[0], spec.size[1] / 2])
            elif spec.shape == "cuboid":
                half_sizes.append([v / 2 for v in spec.size])
            else:
                half_sizes.append([spec.size[0] / 2] * 3)  # STL 最长边构造保守包围盒
            shapes.append({"sphere": 0, "cylinder": 1}.get(spec.shape, 2))
        self.half_sizes = torch.tensor(half_sizes, device=env.device)
        self.shapes = torch.tensor(shapes, device=env.device)

    def reset(self, ids):
        self.planner.reset(ids)
        self.motor.reset(ids)

    def compute(self):
        from isaaclab.utils.math import matrix_from_quat
        env, data = self.env, self.env.robot.data
        states = torch.stack([obj.data.root_state_w for obj in env.objects], 1)
        selected = env.active_object.clamp_min(0)
        state = states[env.indices, selected]
        w, x, y, z = data.root_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        position = rotate_yaw(state[:, :3] - data.root_pos_w, yaw, inverse=True)
        velocity = rotate_yaw(state[:, 7:10] - data.root_lin_vel_w, yaw, inverse=True)
        gravity = torch.tensor(env.cfg.sim.gravity, device=env.device).expand_as(position)
        rotation = matrix_from_quat(state[:, 3:7])
        rotation = rotate_yaw(rotation.transpose(1, 2), yaw[:, None], inverse=True).transpose(1, 2)
        size = self.half_sizes[selected]
        extent = (rotation.abs() @ size[..., None]).squeeze(-1)
        cylinder = size[:, :1] * rotation[:, :, :2].square().sum(-1).sqrt() + size[:, 2:] * rotation[:, :, 2].abs()
        extent = torch.where((self.shapes[selected] == 1)[:, None], cylinder, extent)
        extent = torch.where((self.shapes[selected] == 0)[:, None], size, extent)
        contacts = []
        # 三组顺序固定为左臂、右臂、躯干；只累加当前活动物体的接触。
        for group_sensors in env.catch_sensors:
            group_force = torch.zeros(env.num_envs, device=env.device)
            for sensor in group_sensors:
                forces = sensor.data.force_matrix_w
                if forces is None:
                    raise RuntimeError("接物传感器缺少物体过滤接触力")
                group_force += forces[env.indices, 0, selected].norm(dim=-1)
            contacts.append(group_force > self.cfg.contact_force)
        contacts = torch.stack(contacts, -1)
        upright = ((data.projected_gravity_b[:, 2] < -0.85)
                   & (data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] > env.settings.minimum_base_height))
        prediction, phase = self.planner.update(position, velocity, gravity, env.active_object >= 0,
                                                 contacts, upright, env.step_dt)
        return self.motor.compute(prediction, position, phase, self.planner.elapsed,
                                  extent, yaw, env.step_dt, contacts)
