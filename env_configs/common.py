"""共享参数。长度 m、速度 m/s、质量 kg、时间 s。无需导入 Isaac Sim。"""
from dataclasses import dataclass, field


@dataclass
class ObjectSpec:
    name: str
    shape: str
    size: tuple[float, ...]
    mass: float
    color: tuple[float, float, float]
    mesh_path: str | None = None  # shape="stl" 时填写；相对项目根目录或绝对路径


@dataclass
class ShelfCfg:
    # position 的 Z 是上层架面高度；尺寸包含板厚。架子固定在环境坐标系。
    position: tuple[float, float, float] = (0.45, 0.65, 0.75)
    size: tuple[float, float, float] = (0.60, 0.55, 0.04)
    lower_height: float = 0.25
    leg_width: float = 0.04
    margin: float = 0.025
    height_tolerance: float = 0.025
    catch_seconds: float = 0.20
    catch_speed: float = 0.65
    settle_seconds: float = 0.75
    linear_speed: float = 0.10
    angular_speed: float = 0.25
    upright_cos: float = 0.95
    contact_force: float = 0.25
    support_weight_fraction: float = 0.70
    ground_tolerance: float = 0.025
    escape_distance: float = 3.0
    joint_speed: float = 6.0

    def validate(self, settings):
        import math
        if len(self.position) != 3 or not all(math.isfinite(v) for v in self.position):
            raise ValueError("架面 position 必须是有限的 XYZ 坐标")
        if len(self.size) != 3 or not all(math.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("架板 size 必须是三个有限正数")
        for name, value in vars(self).items():
            if name not in ("position", "size") and not (math.isfinite(value) and value > 0):
                raise ValueError(f"架子参数 {name} 必须为有限正数")
        if not (self.size[2] < self.lower_height < self.position[2] - self.size[2]):
            raise ValueError("两层架板必须位于地面上且不能重叠")
        if self.leg_width >= min(self.size[:2]) / 2:
            raise ValueError("架腿宽度过大")
        if not (0 < self.upright_cos <= 1 and 0 < self.support_weight_fraction <= 1):
            raise ValueError("姿态余弦和承重比例必须位于 (0, 1]")
        if self.catch_seconds + self.settle_seconds >= settings.episode_seconds - settings.first_throw_delay:
            raise ValueError("回合必须留出接住和放稳的时间")
        if any(settings.objects[0].size[i] + 2 * self.margin >= self.size[i] for i in (0, 1)):
            raise ValueError("架面必须比箱体更大，并留出边缘余量")
        if any(abs(self.position[i]) + self.size[i] / 2 >= settings.env_spacing / 2 for i in (0, 1)):
            raise ValueError("架子超出当前并行环境的空间范围")
        if math.hypot(*self.position[:2]) + max(self.size[:2]) / 2 >= self.escape_distance:
            raise ValueError("架子必须位于物体有效活动范围内")
        if self.escape_distance <= settings.max_launch_distance:
            raise ValueError("物体活动范围必须大于最大抛掷距离，避免刚投放就被判为越界")


@dataclass
class Settings:
    shelf_task: bool = False  # 独立的接箱再放架任务；不自动生成任何放置动作
    shelf: ShelfCfg = field(default_factory=ShelfCfg)
    num_envs: int = 64
    env_spacing: float = 8.0
    ground_ruler: bool = False  # 地面视觉标尺：小格 1 dm；不参与碰撞
    episode_seconds: float = 20.0
    physics_dt: float = 1 / 120
    decimation: int = 2
    speed_range: tuple[float, float] = (6.0, 8.0)
    distance_range: tuple[float, float] = (1.0, 1.4)
    # None：按距离/方位角采样；设置后 distance_range 为 X 偏移，本参数独立采样 Y（m）。
    launch_lateral_range: tuple[float, float] | None = None
    launch_height_range: tuple[float, float] = (1.05, 1.20)
    azimuth_range: tuple[float, float] = (-0.12, 0.12)
    target_height: float = 1.0
    target_lateral_range: tuple[float, float] = (-0.65, 0.65)  # 相对机器人中心的横向瞄准偏移，m
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

    @property
    def max_launch_distance(self):
        import math
        lateral = 0.0 if self.launch_lateral_range is None else max(map(abs, self.launch_lateral_range))
        return math.hypot(self.distance_range[1], lateral)

    def validate(self):
        import math
        import re
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
        low, high = self.target_lateral_range
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError("target_lateral_range 必须为有限数且最小值 <= 最大值")
        if self.launch_lateral_range is not None:
            launch_low, launch_high = self.launch_lateral_range
            if not (math.isfinite(launch_low) and math.isfinite(launch_high) and launch_low <= launch_high):
                raise ValueError("launch_lateral_range 必须为有限数且最小值 <= 最大值")
        if self.env_spacing < 2 * self.max_launch_distance + 1:
            raise ValueError("env_spacing 太小，至少为 2 * 最大抛掷距离 + 1")
        if self.shelf_task:
            if self.continuous:
                raise ValueError("放架任务需要单次抛掷回合，不能启用 continuous")
            if len(self.objects) != 1 or self.objects[0].shape != "cuboid":
                raise ValueError("放架任务需要且仅需要一个 cuboid 箱体")
            if not math.isfinite(self.objects[0].mass) or not math.isfinite(self.action_scale) or self.action_scale <= 0:
                raise ValueError("箱体质量和关节动作幅度必须为有限正数")
            self.shelf.validate(self)
        # 保证最慢速度也可覆盖所有发射点到标称目标高度的弹道。
        v2 = self.speed_range[0] ** 2
        d = self.distance_range[1] + max(abs(low), abs(high))
        if self.launch_lateral_range is not None:
            lateral_distance = max(abs(high - launch_low), abs(low - launch_high))
            d = math.hypot(self.distance_range[1], lateral_distance)
        dz = self.target_height - self.launch_height_range[0]
        if v2 * v2 - 9.81 * (9.81 * d * d + 2 * dz * v2) < 0:
            raise ValueError("速度太低，无法到达目标；提高 speed_range 或减小距离/目标高度")
