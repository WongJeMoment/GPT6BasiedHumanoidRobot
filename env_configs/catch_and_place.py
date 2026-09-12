"""接住高速箱体后放上架子；本阶段只定义环境，不接入 GPT 规划器。"""
from env_configs.common import ObjectSpec, Settings, ShelfCfg

CONFIG = Settings(
    shelf_task=True,
    controller="joint",
    continuous=False,
    num_envs=64,
    episode_seconds=12.0,
    action_scale=1.5,  # 默认姿态附近的关节目标偏移 rad，另有软限位和目标速度限制
    speed_range=(6.0, 8.0),
    target_lateral_range=(-0.65, 0.65),
    objects=[ObjectSpec("box", "cuboid", (0.35, 0.35, 0.30), 0.70, (0.12, 0.40, 0.85))],
    shelf=ShelfCfg(position=(0.45, 0.65, 0.75)),
)
