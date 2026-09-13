"""接住高速箱体后放上架子；本阶段只定义环境，不接入 GPT 规划器。"""
from env_configs.common import ObjectSpec, Settings, ShelfCfg

CONFIG = Settings(
    shelf_task=True,
    continuous=False,
    num_envs=64,
    env_spacing=12.0,
    ground_ruler=True,
    episode_seconds=12.0,
    action_scale=1.5,  # 默认姿态附近的关节目标偏移 rad，另有软限位和目标速度限制
    distance_range=(3.0, 5.0),  # 发射时相对机器人基座的 X 偏移，m
    launch_lateral_range=(-0.4, 0.4),  # Y 独立随机偏移 ±4 dm，不随 X 距离变化
    speed_range=(7.5, 8.0),    # 6 m/s 无法覆盖最远端；保证所有偏侧目标均有有效弹道
    target_lateral_range=(-0.65, 0.65),
    objects=[ObjectSpec("box", "cuboid", (0.35, 0.35, 0.30), 0.70, (0.12, 0.40, 0.85))],
    shelf=ShelfCfg(position=(0.45, 0.65, 0.75), escape_distance=7.0),
)
