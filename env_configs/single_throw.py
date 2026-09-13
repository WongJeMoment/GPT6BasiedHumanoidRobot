"""大物体单次抛掷：球直径 40 cm、箱体 35×35×30 cm、圆柱直径 32 cm / 高 40 cm。"""
from env_configs.common import ObjectSpec, Settings

CONFIG = Settings(
    continuous=False,
    episode_seconds=8.0,
    speed_range=(6.0, 8.0),       # 初速度 m/s
    distance_range=(1.0, 1.4),   # 前方近距离抛入，m
    launch_height_range=(1.05, 1.20),
    target_height=1.0,          # 目标为胸前高度，m
    azimuth_range=(-0.12, 0.12),
    objects=[
        ObjectSpec("ball", "sphere", (0.20,), 0.50, (0.85, 0.18, 0.12)),  # 半径
        ObjectSpec("box", "cuboid", (0.35, 0.35, 0.30), 0.70, (0.12, 0.40, 0.85)),
        ObjectSpec("can", "cylinder", (0.16, 0.40), 0.60, (0.95, 0.65, 0.12)),  # 半径、高度
    ],
)
