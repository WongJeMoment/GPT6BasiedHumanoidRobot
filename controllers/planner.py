"""高层：用观测驱动任务阶段，而不是按固定时间播放接球动画。"""
from enum import IntEnum
import torch


class Phase(IntEnum):
    READY = 0
    INTERCEPT = 1
    ABSORB = 2
    HOLD = 3
    RECOVER = 4


class CatchPlanner:
    def __init__(self, count, device, cfg):
        self.cfg = cfg
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(count, device=device)
        self.stable_time = torch.zeros_like(self.elapsed)
        self.success = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids):
        """局部重置：不影响其他并行环境的任务进度。"""
        self.phase[ids] = Phase.READY
        self.elapsed[ids] = 0
        self.stable_time[ids] = 0
        self.success[ids] = False

    def update(self, position, velocity, gravity, active, contact, upright, dt):
        """位置/速度/重力均位于仅随基座偏航旋转的坐标系。

        contact 必须来自双手对当前物体的过滤接触力，不能用距离冒充接触。
        返回重力补偿拦截点、阶段；不写入物体状态，也不创建吸附约束。
        """
        c = self.cfg
        previous = self.phase.clone()
        self.elapsed += dt
        # 来球 vx<0；对静止、远离和已飞过的物体显式判为不可拦截。
        t = (position[:, 0] - c.intercept_x) / (-velocity[:, 0]).clamp_min(0.01)
        prediction = position + velocity * t.clamp(0, c.horizon)[:, None]
        prediction += 0.5 * gravity * t.clamp(0, c.horizon)[:, None].square()
        reachable = (active & (velocity[:, 0] < -0.1) & (t > 0) & (t < c.horizon)
                     & (prediction[:, 1].abs() < 0.42)
                     & (prediction[:, 2] > -0.1) & (prediction[:, 2] < 0.65))
        ready = previous == Phase.READY
        self.phase[ready & reachable & upright] = Phase.INTERCEPT
        tracking = previous == Phase.INTERCEPT
        touching = contact.any(-1) & active
        self.phase[tracking & touching] = Phase.ABSORB
        missed = ~active | (position[:, 0] < 0.05) | (position[:, 2] < -0.35)
        self.phase[tracking & ~touching & (missed | (self.elapsed > c.horizon + 0.2))] = Phase.RECOVER
        absorbing = previous == Phase.ABSORB
        self.phase[absorbing & (self.elapsed >= c.absorb_seconds)] = Phase.HOLD
        holding = previous == Phase.HOLD
        self.phase[holding & (missed | (~contact.any(-1) & (self.elapsed > 0.25)))] = Phase.RECOVER
        stable = (active & upright & contact.all(-1) & (velocity.norm(dim=-1) < c.hold_speed)
                  & (position[:, 0] > 0.05) & (position[:, 0] < 0.65)
                  & (position[:, 1].abs() < 0.4) & (position[:, 2] > -0.1)
                  & ((self.phase == Phase.HOLD) | (self.phase == Phase.ABSORB)))
        self.stable_time = torch.where(stable, self.stable_time + dt, 0.0)
        self.success |= self.stable_time >= c.hold_seconds
        self.phase[(previous == Phase.RECOVER) & (self.elapsed >= c.recover_seconds)] = Phase.READY
        self.phase[~upright] = Phase.RECOVER
        self.elapsed[self.phase != previous] = 0
        return prediction, self.phase
