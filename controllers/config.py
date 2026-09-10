"""控制参数独立于场景参数：长度 m、时间 s、角度 rad。"""
from dataclasses import dataclass
import math


@dataclass
class CatchControlCfg:
    intercept_x: float = 0.36       # 胸前拦截平面，相对基座朝向
    horizon: float = 0.65          # 弹道预测最大时间
    absorb_seconds: float = 0.18   # 接触后双臂后撤缓冲时间
    hold_seconds: float = 0.4      # 双侧接触且低相对速度持续此时长才算成功
    recover_seconds: float = 0.5
    contact_force: float = 0.25    # 每只手对当前物体的最小接触力 N
    hold_speed: float = 0.65
    ik_damping: float = 0.08
    joint_speed: float = 5.0       # 目标关节速度上限 rad/s
    residual_scale: float = 0.08   # PPO 在分层目标上叠加的小幅残差

    def validate(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"接物控制参数 {name} 必须为有限正数")
