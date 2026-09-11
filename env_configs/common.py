"""共享参数。长度 m、速度 m/s、质量 kg、时间 s。无需导入 Isaac Sim。"""
from dataclasses import dataclass, field
from controllers.config import CatchControlCfg


@dataclass
class ObjectSpec:
    name: str
    shape: str
    size: tuple[float, ...]
    mass: float
    color: tuple[float, float, float]
    mesh_path: str | None = None  # shape="stl" 时填写；相对项目根目录或绝对路径


@dataclass
class Settings:
    residual_rl: bool = False  # 扩展控制状态观测及接物密集奖励；旧检查点不可混用
    controller: str = "joint"  # joint：原关节动作；hierarchical：分层抱接 + PPO 残差
    catch_control: CatchControlCfg = field(default_factory=CatchControlCfg)
    num_envs: int = 64
    env_spacing: float = 8.0
    episode_seconds: float = 20.0
    physics_dt: float = 1 / 120
    decimation: int = 2
    speed_range: tuple[float, float] = (4.0, 5.0)
    distance_range: tuple[float, float] = (1.0, 1.4)
    launch_height_range: tuple[float, float] = (1.05, 1.20)
    azimuth_range: tuple[float, float] = (-0.12, 0.12)
    target_height: float = 1.0
    interval_range: tuple[float, float] = (2.5, 4.0)
    first_throw_delay: float = 0.5
    continuous: bool = False
    action_scale: float = 0.25
    minimum_base_height: float = 0.40
    upright_reward: float = 2.0
    posture_penalty: float = 0.2
    action_penalty: float = 0.01
    velocity_penalty: float = 0.05
    fall_penalty: float = 10.0
    hand_color: tuple[float, float, float] = (0.015, 0.015, 0.015)
    hand_roughness: float = 0.75
    robot_usd: str | None = None  # None 使用 Isaac Lab 官方 G1；可改为本地同结构 USD
    objects: list[ObjectSpec] = field(default_factory=lambda: [
        ObjectSpec("ball", "sphere", (0.20,), 0.50, (0.85, 0.18, 0.12)),
        ObjectSpec("box", "cuboid", (0.35, 0.35, 0.30), 0.70, (0.12, 0.40, 0.85)),
        ObjectSpec("can", "cylinder", (0.16, 0.40), 0.60, (0.95, 0.65, 0.12)),
    ])

    def validate(self):
        import math
        import re
        if self.controller not in ("joint", "hierarchical"):
            raise ValueError("controller 必须为 joint 或 hierarchical")
        if self.residual_rl and self.controller != "hierarchical":
            raise ValueError("residual_rl 需要 hierarchical 控制器")
        self.catch_control.validate()
        for name in ("speed_range", "distance_range", "launch_height_range", "interval_range"):
            low, high = getattr(self, name)
            if not (0 < low <= high and math.isfinite(high)):
                raise ValueError(f"{name} 必须满足 0 < 最小值 <= 最大值且为有限数")
        if not self.objects or len({x.name for x in self.objects}) != len(self.objects):
            raise ValueError("objects 不可为空且名称必须唯一")
        for obj in self.objects:
            expected = {"sphere": 1, "cuboid": 3, "cylinder": 2, "stl": 1}.get(obj.shape)
            if expected is None or len(obj.size) != expected or any(not math.isfinite(v) or v <= 0 for v in obj.size):
                raise ValueError(f"无效物体尺寸/形状: {obj}")
            if obj.shape == "stl":
                from pathlib import Path
                path = Path(obj.mesh_path or "").expanduser()
                if not path.is_absolute():
                    path = Path(__file__).resolve().parents[1] / path
                if path.suffix.lower() != ".stl" or not path.is_file():
                    raise ValueError(f"STL 文件不存在或格式错误: {path}")
            if obj.mass <= 0 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", obj.name):
                raise ValueError(f"无效物体名称/质量: {obj}")
        if self.num_envs < 1 or self.decimation < 1 or self.physics_dt <= 0:
            raise ValueError("环境数、decimation、physics_dt 必须为正")
        if not 0 <= self.first_throw_delay < self.episode_seconds:
            raise ValueError("首次抛掷必须在回合结束前")
        if self.azimuth_range[0] > self.azimuth_range[1]:
            raise ValueError("azimuth_range 顺序错误")
        if self.env_spacing < 2 * self.distance_range[1] + 1:
            raise ValueError("env_spacing 太小，至少为 2 * 最大抛掷距离 + 1")
        # 保证最慢速度也可覆盖所有发射点到标称目标高度的弹道。
        v2 = self.speed_range[0] ** 2
        d = self.distance_range[1]
        dz = self.target_height - self.launch_height_range[0]
        if v2 * v2 - 9.81 * (9.81 * d * d + 2 * dz * v2) < 0:
            raise ValueError("速度太低，无法到达目标；提高 speed_range 或减小距离/目标高度")
