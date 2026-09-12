"""接箱放架的任务状态和奖励。纯张量实现，不生成动作或修改箱体物理状态。"""
from enum import IntEnum

import torch


class ShelfPhase(IntEnum):
    WAITING = 0
    CATCH = 1
    PLACE = 2
    SETTLE = 3
    SUCCESS = 4


# 物体姿态4/角速度3/相对速度3，架面位置3/尺寸3/目标误差3，接触3，
# 全身接触1/架面承重1/曾接住1，阶段5/计时2/剩余时间1，关节目标23，
# 基座高度1/箱体包围盒3/基座姿态4，历史最短距离1/先碰架标记1。
# 加原有单箱观测86，共152维。
SHELF_OBSERVATIONS = 66


def box_geometry(quaternion, half_size):
    """返回旋转后箱体的世界轴包围盒半尺寸和局部 Z 轴竖直程度；四元数为 wxyz。"""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    rotation = torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), -1).reshape(-1, 3, 3)
    return (rotation.abs() @ half_size[..., None]).squeeze(-1), rotation[:, 2, 2].abs()


class ShelfTask:
    def __init__(self, count, device, cfg, box_size):
        self.cfg = cfg
        self.half_size = torch.tensor(box_size, dtype=torch.float32, device=device) / 2
        self.surface = torch.tensor(cfg.position, dtype=torch.float32, device=device)
        self.size = torch.tensor(cfg.size, dtype=torch.float32, device=device)
        self.goal = self.surface.clone()
        self.goal[2] += self.half_size[2]
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.contact = torch.zeros((count, 3), dtype=torch.bool, device=device)
        for name in ("caught", "new_catch", "success", "dropped", "on_shelf", "supported", "robot_touch", "shelf_first"):
            setattr(self, name, torch.zeros(count, dtype=torch.bool, device=device))
        for name in ("catch_time", "settle_time", "best_distance", "progress"):
            setattr(self, name, torch.zeros(count, device=device))

    def reset(self, ids):
        for name in ("phase", "contact", "caught", "new_catch", "success", "dropped", "on_shelf",
                     "supported", "robot_touch", "shelf_first", "catch_time", "settle_time", "best_distance", "progress"):
            getattr(self, name)[ids] = 0

    def update(self, active, position, quaternion, velocity, angular_velocity, robot_position,
               robot_velocity, contact, robot_touch, shelf_touch, supported, upright, fallen, dt):
        """每个控制步的物理积分后调用一次；position 与机器人位置均减去环境原点。"""
        cfg = self.cfg
        extent, vertical = box_geometry(quaternion, self.half_size)
        bottom = position[:, 2] - extent[:, 2]
        self.dropped.copy_(active & ((bottom <= cfg.ground_tolerance)
                                    | (position[:, :2].norm(dim=-1) > cfg.escape_distance)))
        valid = active & upright & ~fallen & ~self.dropped
        self.contact.copy_(contact & active[:, None])
        self.robot_touch.copy_(robot_touch & active)
        self.supported.copy_(supported & active)
        # 箱体先撞架再被提起不能补算接住；标记保留到下一回合。
        self.shelf_first |= active & shelf_touch & ~self.caught
        relative = position - robot_position
        grasp = (valid & self.contact[:, :2].all(-1) & ~shelf_touch & ~self.shelf_first
                 & ((velocity - robot_velocity).norm(dim=-1) < cfg.catch_speed)
                 & (relative[:, :2].norm(dim=-1) < 0.9) & (position[:, 2] > 0.5))
        self.catch_time.copy_(torch.where(grasp, (self.catch_time + dt).clamp_max(cfg.catch_seconds), 0.0))
        self.new_catch.copy_(~self.caught & (self.catch_time >= cfg.catch_seconds))
        self.caught |= self.new_catch

        # 检查整个旋转后的箱体投影；只有中心在架面内不够，悬空探出边缘不能成功。
        inside = ((position[:, :2] - self.surface[:2]).abs() + extent[:, :2]
                  <= self.size[:2] / 2 - cfg.margin).all(-1)
        self.on_shelf.copy_(active & inside & ((bottom - self.surface[2]).abs() <= cfg.height_tolerance)
                            & (vertical >= cfg.upright_cos) & self.supported)
        settled = (valid & self.caught & self.on_shelf & ~self.robot_touch
                   & (velocity.norm(dim=-1) < cfg.linear_speed)
                   & (angular_velocity.norm(dim=-1) < cfg.angular_speed))
        self.settle_time.copy_(torch.where(settled, (self.settle_time + dt).clamp_max(cfg.settle_seconds), 0.0))
        self.success.copy_(settled & (self.settle_time >= cfg.settle_seconds))

        distance = (position - self.goal).norm(dim=-1)
        self.best_distance.copy_(torch.where(self.new_catch, distance, self.best_distance))
        self.progress.copy_(torch.where(valid & self.caught, (self.best_distance - distance).clamp_min(0), 0.0))
        self.best_distance.copy_(torch.where(valid & self.caught, torch.minimum(self.best_distance, distance),
                                             self.best_distance))
        self.phase.copy_(torch.where(active, int(ShelfPhase.CATCH), int(ShelfPhase.WAITING)))
        self.phase[self.caught & active] = ShelfPhase.PLACE
        self.phase[settled] = ShelfPhase.SETTLE
        self.phase[self.success] = ShelfPhase.SUCCESS

    def reward(self, active, proximity, upright, fallen, timeout, actions, action_change, elapsed, dt):
        failed = self.dropped | fallen
        valid = active & upright & ~failed
        # 接住前的辅助奖励随时间衰减；放架阶段只奖刷新最短距离，不能往返刷进度。
        approach = (~self.caught).float() * torch.exp(-elapsed.clamp_min(0) / 2.0) * (
            2 * proximity + 2 * self.contact[:, :2].float().mean(-1))
        settle = 4 * self.settle_time / self.cfg.settle_seconds
        regularization = .02 * actions.square().sum(-1) + .015 * action_change.square().sum(-1)
        dense = (valid.float() * (approach + settle) - active.float() - regularization) * dt
        positive = valid.float() * (15 * self.new_catch.float() + 30 * self.progress + 100 * self.success.float())
        return (dense + positive - 60 * self.dropped.float() - 80 * fallen.float()
                - 30 * (timeout & ~self.success & ~failed).float())


def spawn_shelf(scene, cfg):
    """两块架板、四根架腿，均为有摩擦的固定运动学刚体。"""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    x, y, top = cfg.position
    sx, sy, thickness = cfg.size
    parts = [("ShelfTop", cfg.size, (x, y, top - thickness / 2)),
             ("ShelfLower", cfg.size, (x, y, cfg.lower_height - thickness / 2))]
    leg_height = top - thickness
    for i, (dx, dy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        parts.append((f"ShelfLeg{i}", (cfg.leg_width, cfg.leg_width, leg_height),
                      (x + dx * (sx - cfg.leg_width) / 2, y + dy * (sy - cfg.leg_width) / 2, leg_height / 2)))
    for name, size, position in parts:
        spawn = sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.38, 0.20) if "Leg" not in name else (0.16, 0.18, 0.20)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6,
                                                          restitution=0.0),
        )
        scene.rigid_objects[name] = RigidObject(RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{name}", spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        ))
