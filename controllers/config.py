"""控制参数独立于场景参数：长度 m、时间 s、角度 rad。"""
from dataclasses import dataclass
import math


@dataclass
class CatchControlCfg:
    intercept_x: float = 0.36       # 胸前拦截平面，相对基座朝向
    horizon: float = 0.65          # 弹道预测最大时间
    absorb_seconds: float = 0.18   # 接触后将物体收向胸前的缓冲时间
    hold_seconds: float = 0.4      # 躯干、左臂、右臂共同接触持续此时长才算成功
    recover_seconds: float = 0.5
    contact_force: float = 0.25    # 躯干/每侧手臂对当前物体的最小接触力 N
    chest_surface_x: float = 0.075 # 官方胸部碰撞网格最大 X 约 0.086 m，选其内侧支撑面
    chest_height: float = 0.28    # 抱持中心相对 torso_link 的局部 Z 偏置
    hug_pressure: float = 0.012   # 目标向胸面内偏少量距离，以 PD 跟踪形成抱紧力
    forearm_weight: float = 0.50  # 前臂包围目标在多点 IK 中的权重
    max_draw_in: float = 0.10     # 躯干未接触时允许的附加收臂距离上限
    draw_in_speed: float = 0.30   # 用真实躯干接触反馈修正几何偏差，m/s
    hold_speed: float = 0.65
    ik_damping: float = 0.08
    joint_speed: float = 5.0       # 目标关节速度上限 rad/s
    residual_scale: float = 0.08   # PPO 在分层目标上叠加的小幅残差
    support_shift_gain: float = 1.8  # 主动下肢：捕获点误差到髋踝平移修正
    upright_tilt_gain: float = 0.8   # 主动下肢：躯干俯仰恢复增益

    def validate(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"接物控制参数 {name} 必须为有限正数")
