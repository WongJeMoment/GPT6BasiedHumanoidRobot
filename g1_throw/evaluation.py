"""固定来物计划的整回合评估；每个环境每轮只记第一次回合，避免自动重置重复计数。"""
import torch
from tensordict import TensorDict


def launch_dataset(settings, episodes, seed):
    """在 CPU 独立随机源预采样，训练动作和自动重置均不能改变测试来物。"""
    generator = torch.Generator().manual_seed(seed)
    plan = {"object": torch.arange(episodes) % len(settings.objects)}
    for key, bounds in (("speed", settings.speed_range), ("distance", settings.distance_range),
                        ("angle", settings.azimuth_range), ("height", settings.launch_height_range),
                        ("target_lateral", settings.target_lateral_range)):
        plan[key] = torch.rand(episodes, generator=generator) * (bounds[1] - bounds[0]) + bounds[0]
    return plan


@torch.inference_mode()
def evaluate(env, policy, episodes=96, seed=2026):
    from dataclasses import asdict
    if env.settings.continuous or env.catch_controller is None:
        raise ValueError("固定回合评估需要单次抛掷与分层控制")
    plan = launch_dataset(env.settings, episodes, seed)
    results = []
    previous_plan = env.launch_plan
    try:
        for offset in range(0, episodes, env.num_envs):
            count = min(env.num_envs, episodes - offset)
            ids = (torch.arange(env.num_envs) + offset).clamp_max(episodes - 1)
            env.launch_plan = {key: value[ids].to(env.device) for key, value in plan.items()}
            env.last_object[:] = -1
            obs, _ = env.reset(seed=seed + offset)
            alive = torch.arange(env.num_envs, device=env.device) < count
            success = torch.zeros_like(alive)
            fell = torch.zeros_like(alive)
            max_hold = torch.zeros(env.num_envs, device=env.device)
            for _ in range(env.max_episode_length + 2):
                actions = torch.zeros_like(env.actions) if policy is None else policy(TensorDict(obs, batch_size=[env.num_envs]))
                obs, reward, terminated, truncated, extras = env.step(actions)
                if not torch.isfinite(reward).all() or not torch.isfinite(obs["policy"]).all():
                    raise RuntimeError("评估出现非有限数值")
                metrics = extras["catch"]  # 重置前的快照，保留终止步上的结果
                success |= alive & metrics["success"]
                max_hold = torch.where(alive, torch.maximum(max_hold, metrics["stable_time"]), max_hold)
                fell |= alive & terminated
                alive &= ~(terminated | truncated)
                # 已结束的环境不再投放新物体；其他环境继续完成首次回合。
                env.next_throw[~alive] = float("inf")
                if not alive.any():
                    break
            if alive.any():
                raise RuntimeError("评估未完成整回合，请检查终止条件")
            for i in range(count):
                results.append({"episode": offset + i, "object": int(plan["object"][offset + i]),
                                "success": bool(success[i]), "fell": bool(fell[i]),
                                "max_hold_seconds": float(max_hold[i])})
    finally:
        env.launch_plan = previous_plan
    caught = sum(row["success"] for row in results)
    report = {"episodes": episodes, "seed": seed, "successes": caught, "success_rate": caught / episodes,
              "settings": asdict(env.settings),
              "falls": sum(row["fell"] for row in results),
              "held_one_second": sum(row["max_hold_seconds"] >= 1.0 for row in results),
              "by_object": {}, "results": results}
    for i, spec in enumerate(env.settings.objects):
        subset = [row for row in results if row["object"] == i]
        report["by_object"][spec.name] = {"episodes": len(subset), "successes": sum(row["success"] for row in subset)}
    return report
