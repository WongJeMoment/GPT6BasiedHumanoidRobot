"""低层：身体与双臂围合、多点阻尼逆运动学、屈膝缓冲和踝髋平衡反馈。

这是站定抱接的反馈基线，不包含跨步规划或保证动力学可行的全身 QP。
"""
import torch

from .planner import Phase
from .hug_targets import hug_targets


def rotate_yaw(vector, yaw, inverse=False):
    """世界坐标与偏航坐标互换，保持 Z 轴竖直。"""
    angle = -yaw if inverse else yaw
    c, s = angle.cos(), angle.sin()
    x, y, z = vector.unbind(-1)
    return torch.stack((c * x - s * y, s * x + c * y, z), -1)


def damped_ik(jacobian, error, damping):
    """解 J dq = dx；阻尼避免手臂接近伸直时的奇异放大。"""
    jt = jacobian.transpose(-1, -2)
    eye = torch.eye(jacobian.shape[-2], device=jacobian.device, dtype=jacobian.dtype)
    return (jt @ torch.linalg.solve(jacobian @ jt + damping**2 * eye, error[..., None])).squeeze(-1)


class WholeBodyController:
    def __init__(self, robot, cfg):
        self.robot, self.cfg = robot, cfg
        names = robot.joint_names
        self.joints = {name: i for i, name in enumerate(names)}
        self.arms = []
        self.hands = []
        self.elbows = []
        for side in ("left", "right"):
            # 使用 USD 中真实掌部刚体，不用肘部位置近似掌心。
            body = f"{side}_palm_link"
            if body not in robot.body_names:
                raise ValueError(f"找不到 {body}；需要官方 23 DOF G1，实际刚体：{robot.body_names}")
            self.hands.append(robot.body_names.index(body))
            self.elbows.append(robot.body_names.index(f"{side}_elbow_pitch_link"))
            self.arms.append([i for i, name in enumerate(names)
                              if name.startswith(side + "_") and ("shoulder" in name or "elbow" in name)])
        self.target = robot.data.default_joint_pos.clone()
        self.torso = robot.body_names.index("torso_link")
        self.draw_in = torch.zeros(self.target.shape[0], device=self.target.device)
        self.previous_phase = torch.zeros(robot.num_instances, device=self.target.device, dtype=torch.long)
        self.feet = [robot.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        # PhysX 的默认质量缓存在 CPU；只在初始化时搬到控制设备。
        mass = robot.data.default_mass.to(self.target.device)
        self.mass_weights = mass / mass.sum(-1, keepdim=True)

    def reset(self, ids):
        self.target[ids] = self.robot.data.default_joint_pos[ids]
        self.previous_phase[ids] = Phase.READY
        self.draw_in[ids] = 0

    def compute(self, prediction, position, phase, elapsed, extent, yaw, dt, contact):
        from isaaclab.utils.math import quat_apply
        robot, c = self.robot, self.cfg
        data = robot.data
        desired = data.default_joint_pos.clone()
        engaged = (phase == Phase.INTERCEPT) | (phase == Phase.ABSORB) | (phase == Phase.HOLD)
        receiving = (phase == Phase.ABSORB) | (phase == Phase.HOLD)
        # 胸前锚点跟随真实躯干位置和姿态，身体直接参与围合物体。
        offset = torch.zeros_like(position)
        offset[:, 0], offset[:, 2] = c.chest_surface_x, c.chest_height
        chest_world = data.body_pos_w[:, self.torso] + quat_apply(data.body_quat_w[:, self.torso], offset)
        chest = rotate_yaw(chest_world - data.root_pos_w, yaw, inverse=True)
        entering = (phase == Phase.ABSORB) & (self.previous_phase != Phase.ABSORB)
        self.draw_in[entering | ~receiving] = 0
        # 几何目标可能因手指厚度、碰撞凸包和 PD 跟踪误差而停在胸前。
        # 双臂承接却尚未抵胸时渐进收臂；一旦真实躯干接触就停止增加收紧量。
        needs_support = receiving & contact[:, :2].any(-1) & ~contact[:, 2]
        self.draw_in += needs_support.float() * c.draw_in_speed * dt
        self.draw_in.clamp_(0, c.max_draw_in)
        chest[:, 0] -= self.draw_in
        # 允许在胸宽内对偏侧物体做小幅对中，避免单侧承物时另一臂错过。
        chest[:, 1] += torch.where(receiving, (position[:, 1] - chest[:, 1]).clamp(-0.08, 0.08), 0.0)
        self.previous_phase.copy_(phase)
        center, palm_targets, elbow_targets = hug_targets(
            prediction, chest, position, phase, elapsed, extent, c)
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
        # 腰部朝向来物；G1 的 torso_joint 仅偏航，俯仰由双髋与踝部协同完成。
        facing = torch.where(receiving, position[:, 1], center[:, 1])
        desired[:, self.joints["torso_joint"]] += (facing * 0.6).clamp(-0.15, 0.15)
        jacobians = robot.root_physx_view.get_jacobians()
        for side_index, (body, arm) in enumerate(zip(self.hands, self.arms)):
            # 浮动基座 Jacobian 前六列为基座自由度；不能当作关节列。
            row = body - 1 if robot.is_fixed_base else body
            columns = arm if robot.is_fixed_base else [i + 6 for i in arm]
            jac = jacobians[:, row, :, columns]
            palm = data.body_pos_w[:, body]
            linear = jac[:, :3]
            goal = data.root_pos_w + rotate_yaw(palm_targets[:, side_index], yaw)
            error = (goal - palm).clamp(-0.12, 0.12)
            elbow_body = self.elbows[side_index]
            elbow_row = elbow_body - 1 if robot.is_fixed_base else elbow_body
            elbow_jac = jacobians[:, elbow_row, :3, columns]
            elbow_goal = data.root_pos_w + rotate_yaw(elbow_targets[:, side_index], yaw)
            elbow_error = (elbow_goal - data.body_pos_w[:, elbow_body]).clamp(-0.12, 0.12)
            # 双点 IK 约束手掌和肘部，防止只移动掌心而手臂没有形成包围。
            weight = c.forearm_weight * receiving[:, None, None]
            task_jac = torch.cat((linear, weight * elbow_jac), 1)
            task_error = torch.cat((error, weight.squeeze(-1) * elbow_error), 1)
            delta = damped_ik(task_jac, task_error, c.ik_damping)
            desired[:, arm] = data.joint_pos[:, arm] + delta
        # 目标限速、关节软限位；由 Isaac Lab 原有 PD 执行器输出力矩。
        limit = c.joint_speed * dt
        self.target += (desired - self.target).clamp(-limit, limit)
        limits = data.soft_joint_pos_limits
        self.target.clamp_(limits[..., 0], limits[..., 1])
        return self.target
