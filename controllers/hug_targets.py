"""身体与双臂围合几何；张量运算可在不启动仿真的情况下测试。"""
import torch
from .planner import Phase


def hug_targets(prediction, chest, position, phase, elapsed, extent, cfg):
    """生成物体中心、双掌及双肘目标，均位于基座偏航坐标系。

    extent 为物体在此坐标系的 XYZ 半尺寸。躯干提供后方支撑，掌部位于
    物体前侧略下方，前臂包围两侧；闭环小幅收向胸前，避免提前撤走承接面。
    """
    engaged = (phase == Phase.INTERCEPT) | (phase == Phase.ABSORB) | (phase == Phase.HOLD)
    receiving = (phase == Phase.ABSORB) | (phase == Phase.HOLD)
    home = chest.clone()
    home[:, 0] += extent[:, 0] - cfg.hug_pressure
    ready = chest.clone()
    ready[:, 0] = cfg.intercept_x
    center = torch.where(engaged[:, None], prediction, ready)
    # 胸前目标仍是最终落点，但每步只给有限位置误差，让手臂持续跟随来物。
    # 三次平滑插值逐渐启用向胸前收紧，HOLD 不重新启动插值。
    progress = (elapsed / cfg.absorb_seconds).clamp(0, 1)
    progress = progress.square() * (3 - 2 * progress)
    progress = torch.where(phase == Phase.HOLD, 1.0, progress)
    correction = home - position
    correction[:, 0] = correction[:, 0].clamp(-0.10, 0.03)
    correction[:, 1] = correction[:, 1].clamp(-0.035, 0.035)
    correction[:, 2] = correction[:, 2].clamp(-0.015, 0.035)
    center = torch.where(receiving[:, None], position + progress[:, None] * correction, center)
    center[:, 0] = center[:, 0].clamp(0.12, 0.55)
    center[:, 1] = center[:, 1].clamp(-0.22, 0.22)
    center[:, 2] = center[:, 2].clamp(0.05, 0.48)
    palms, elbows = [], []
    for sign in (1, -1):
        palm = center.clone()
        # 拦截时开臂让物体进入胸前空间；缓冲中逐渐移到前侧内收托住。
        wrap = torch.where(receiving, progress, 0.0)
        # 固定手指会伸到掌部原点前方，掌部只偏前少量，避免把承接面推离胸口。
        palm[:, 0] += 0.25 * extent[:, 0] * wrap
        palm[:, 1] += sign * (extent[:, 1] + 0.045 - wrap * (0.20 * extent[:, 1] + 0.055))
        palm[:, 2] -= 0.25 * extent[:, 2]
        elbow = center.clone()
        # 肘部也伸到物体侧面，使肘→掌方向朝内，形成环抱而非两臂平行前伸。
        elbow[:, 0] += extent[:, 0] * 0.10
        elbow[:, 1] += sign * (extent[:, 1] + 0.075)
        elbow[:, 2] -= 0.30 * extent[:, 2]
        palms.append(palm)
        elbows.append(elbow)
    return center, torch.stack(palms, 1), torch.stack(elbows, 1)
