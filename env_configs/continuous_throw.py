"""连续抛掷：每次重新随机物体和速度，相邻两次保证物体种类不同。"""
from env_configs.common import ObjectSpec, Settings

CONFIG = Settings(
    continuous=True,
    episode_seconds=30.0,
    speed_range=(6.0, 8.0),
    interval_range=(4.0, 6.0),
    distance_range=(1.0, 1.4),
    launch_height_range=(1.05, 1.20),
    target_height=1.0,
    azimuth_range=(-0.12, 0.12),
    # STL 的 size=(最长边长度,)；单位 m，按比例缩放并自动居中。
    # mass 单位 kg；可删除条目或加入自己的 STL 文件。
    objects=[
        ObjectSpec("teapot", "stl", (0.45,), 0.60, (0.85, 0.22, 0.12), "assets/stl/teapot.stl"),
        ObjectSpec("ring", "stl", (0.45,), 0.40, (0.95, 0.65, 0.10), "assets/stl/torus.stl"),
        ObjectSpec("angle_block", "stl", (0.40,), 0.60, (0.15, 0.55, 0.85), "assets/stl/angle_block.stl"),
        ObjectSpec("plate", "stl", (0.50,), 0.40, (0.25, 0.75, 0.35), "assets/stl/plate_holes.stl"),
        ObjectSpec("feature_block", "stl", (0.45,), 0.60, (0.65, 0.25, 0.85), "assets/stl/featuretype.stl"),
        ObjectSpec("pocket", "stl", (0.45,), 0.40, (0.15, 0.75, 0.75), "assets/stl/octagonal_pocket.stl"),
        ObjectSpec("xyz_cube", "stl", (0.35,), 0.70, (0.85, 0.45, 0.15), "assets/stl/20mm-xyz-cube.stl"),
        ObjectSpec("rod", "stl", (0.55,), 0.50, (0.70, 0.70, 0.75), "assets/stl/cylinder.stl"),
    ],
)
