"""低层：双臂阻尼逆运动学 + 屈膝缓冲 + 踝髋平衡反馈。

这是站定抱接的反馈基线，不包含跨步规划或保证动力学可行的全身 QP。
"""
import torch

from .planner import Phase


def rotate_yaw(vector, yaw, inverse=False):
    """世界坐标与偏航坐标互换，保持 Z 轴竖直。"""
    angle = -yaw if inverse else yaw
    c, s = angle.cos(), angle.sin()
    x, y, z = vector.unbind(-1)
    return torch.stack((c * x - s * y, s * x + c * y, z), -1)


def damped_ik(jacobian, error, damping):
    """解 J dq = dx；阻尼避免手臂接近伸直时的奇异放大。"""
    jt = jacobian.transpose(-1, -2)
    eye = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype)
    return (jt @ torch.linalg.solve(jacobian @ jt + damping**2 * eye, error[..., None])).squeeze(-1)


class WholeBodyController:
    def __init__(self, robot, cfg):
        self.robot, self.cfg = robot, cfg
        names = robot.joint_names
        self.joints = {name: i for i, name in enumerate(names)}
        self.arms = []
        self.hands = []
        for side in ("left", "right"):
            # 使用 USD 中真实掌部刚体，不用肘部位置近似掌心。
            body = f"{side}_palm_link"
            if body not in robot.body_names:
                raise ValueError(f"找不到 {body}；需要官方 23 DOF G1，实际刚体：{robot.body_names}")
            self.hands.append(robot.body_names.index(body))
            self.arms.append([i for i, name in enumerate(names)
                              if name.startswith(side + "_") and ("shoulder" in name or "elbow" in name)])
        self.target = robot.data.default_joint_pos.clone()
        self.feet = [robot.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        # PhysX 的默认质量缓存在 CPU；只在初始化时搬到控制设备。
        mass = robot.data.default_mass.to(self.target.device)
        self.mass_weights = mass / mass.sum(-1, keepdim=True)

    def reset(self, ids):
        self.target[ids] = self.robot.data.default_joint_pos[ids]

    def compute(self, prediction, position, velocity, phase, elapsed, radius, yaw, dt):
        robot, c = self.robot, self.cfg
        data = robot.data
        desired = data.default_joint_pos.clone()
        engaged = (phase == Phase.INTERCEPT) | (phase == Phase.ABSORB) | (phase == Phase.HOLD)
        # 同一个拦截点同时决定手的位置、膝部压缩量和躯干朝向。
        center = torch.zeros_like(position)
        center[:, 0], center[:, 2] = c.intercept_x, 0.28
        center = torch.where(engaged[:, None], prediction, center)
        receiving = (phase == Phase.ABSORB) | (phase == Phase.HOLD)
        center = torch.where(receiving[:, None], position + velocity * 0.035, center)
        center[:, 0] = center[:, 0].clamp(0.18, 0.46)
        center[:, 1] = center[:, 1].clamp(-0.22, 0.22)
        center[:, 2] = center[:, 2].clamp(0.05, 0.48)
        # 顺来球方向后撤，而非在接触瞬间锁死手臂。
        retract = (elapsed / c.absorb_seconds).clamp(0, 1) * 0.08
        center[:, 0] -= torch.where(phase == Phase.ABSORB, retract, 0.0)
        compression = torch.where(engaged, (0.3 - center[:, 2]).clamp(0, 0.2) + 0.08, 0.0)
        compression += receiving.float() * 0.08
        gravity = data.projected_gravity_b
        # 质量加权质心及线性倒立摆捕获点：抬臂使质心前移时，下肢提前补偿。
        weights = self.mass_weights
        com = (data.body_com_pos_w * weights[..., None]).sum(1)
        com_velocity = (data.body_com_lin_vel_w * weights[..., None]).sum(1)
        support = data.body_pos_w[:, self.feet].mean(1)
        height = (com[:, 2] - support[:, 2]).clamp_min(0.2)
        capture = com + com_velocity * torch.sqrt(height / 9.81)[:, None]
        support_error = rotate_yaw(capture - support, yaw, inverse=True)
        pitch = (1.2 * gravity[:, 0] + 0.25 * data.root_lin_vel_b[:, 0]
                 + 0.15 * data.root_ang_vel_b[:, 1]
                 + 1.5 * support_error[:, 0]).clamp(-0.35, 0.35)
        roll = (-0.65 * gravity[:, 1] - 0.12 * data.root_lin_vel_b[:, 1]
                + 0.08 * data.root_ang_vel_b[:, 0]
                - 1.5 * support_error[:, 1]).clamp(-0.16, 0.16)
        for side in ("left", "right"):
            for suffix, delta in (("hip_pitch", -compression * 0.5 + pitch * 0.4),
                                  ("knee", compression), ("ankle_pitch", -compression * 0.5 + pitch),
                                  ("hip_roll", -roll * 0.4), ("ankle_roll", roll)):
                desired[:, self.joints[f"{side}_{suffix}_joint"]] += delta
        desired[:, self.joints["torso_joint"]] += (center[:, 1] * 0.6).clamp(-0.15, 0.15)
        jacobians = robot.root_physx_view.get_jacobians()
        for side_index, (body, arm) in enumerate(zip(self.hands, self.arms)):
            # 浮动基座 Jacobian 前六列为基座自由度；不能当作关节列。
            row = body - 1 if robot.is_fixed_base else body
            columns = arm if robot.is_fixed_base else [i + 6 for i in arm]
            jac = jacobians[:, row, :, columns]
            palm = data.body_pos_w[:, body]
            linear = jac[:, :3]
            goal = center.clone()
            opening = radius + torch.where(receiving, -0.015, 0.045)
            goal[:, 1] += (1 if side_index == 0 else -1) * opening
            goal[:, 2] -= radius * 0.3  # 从物体中线略下方抱接，提供托举分量
            goal = data.root_pos_w + rotate_yaw(goal, yaw)
            error = (goal - palm).clamp(-0.12, 0.12)
            delta = damped_ik(linear, error, c.ik_damping)
            desired[:, arm] = data.joint_pos[:, arm] + delta
        # 目标限速、关节软限位；由 Isaac Lab 原有 PD 执行器输出力矩。
        limit = c.joint_speed * dt
        self.target += (desired - self.target).clamp(-limit, limit)
        limits = data.soft_joint_pos_limits
        self.target.clamp_(limits[..., 0], limits[..., 1])
        return self.target
