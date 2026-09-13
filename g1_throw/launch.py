"""独立随机源生成固定来物样本，供环境复现和检查使用。"""
import torch


def launch_dataset(settings, episodes, seed):
    """在 CPU 独立随机源预采样，训练动作和自动重置均不能改变测试来物。"""
    generator = torch.Generator().manual_seed(seed)
    plan = {"object": torch.arange(episodes) % len(settings.objects)}
    for key, bounds in (("speed", settings.speed_range), ("distance", settings.distance_range),
                        ("angle", settings.azimuth_range), ("height", settings.launch_height_range),
                        ("target_lateral", settings.target_lateral_range)):
        plan[key] = torch.rand(episodes, generator=generator) * (bounds[1] - bounds[0]) + bounds[0]
    if settings.launch_lateral_range is not None:
        low, high = settings.launch_lateral_range
        plan["launch_lateral"] = torch.rand(episodes, generator=generator) * (high - low) + low
    return plan
