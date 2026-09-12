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
    if env.settings.shelf_task:
        return evaluate_shelf(env, policy, episodes, seed)
    if env.settings.continuous or env.catch_controller is None:
        raise ValueError("固定回合评估需要单次抛掷与分层控制")
    plan = launch_dataset(env.settings, episodes, seed)
    results = []
    trace = []  # 首批每种物体一个环境，每秒记录姿态，定位抱持后的缓慢漂移。
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
            dropped = torch.zeros_like(alive)
            max_hold = torch.zeros(env.num_envs, device=env.device)
            end_hold = torch.zeros_like(max_hold)
            duration = torch.zeros_like(max_hold)
            leg_ids = [i for i, name in enumerate(env.robot.joint_names)
                       if any(part in name for part in ("hip", "knee", "ankle"))]
            joint_min = env.robot.data.joint_pos[:, leg_ids].clone()
            joint_max = joint_min.clone()
            for step in range(env.max_episode_length + 2):
                if offset == 0 and step % round(1 / env.step_dt) == 0:
                    for i in range(min(count, len(env.settings.objects))):
                        if alive[i]:
                            trace.append({"episode": i, "seconds": step * env.step_dt,
                                          "base_position": (env.robot.data.root_pos_w[i] - env.scene.env_origins[i]).tolist(),
                                          "gravity_b": env.robot.data.projected_gravity_b[i].tolist(),
                                          "object_heights": [float(o.data.root_pos_w[i, 2]) for o in env.objects],
                                          "support_error": env.catch_controller.motor.support_error[i, :2].tolist()})
                actions = torch.zeros_like(env.actions) if policy is None else policy(TensorDict(obs, batch_size=[env.num_envs]))
                obs, reward, terminated, truncated, extras = env.step(actions)
                if not torch.isfinite(reward).all() or not torch.isfinite(obs["policy"]).all():
                    raise RuntimeError("评估出现非有限数值")
                metrics = extras["catch"]  # 重置前的快照，保留终止步上的结果
                success |= alive & metrics["success"]
                max_hold = torch.where(alive, torch.maximum(max_hold, metrics["stable_time"]), max_hold)
                fell |= alive & metrics["fallen"]
                dropped |= alive & metrics["dropped"]
                finished = alive & (terminated | truncated)
                end_hold[finished] = metrics["stable_time"][finished]
                duration[finished] = (step + 1) * env.step_dt
                # 排除复位后的关节跳变，只累计真实运行时的下肢活动范围。
                moving = alive & ~finished
                joint_min = torch.where(moving[:, None], torch.minimum(joint_min, env.robot.data.joint_pos[:, leg_ids]), joint_min)
                joint_max = torch.where(moving[:, None], torch.maximum(joint_max, env.robot.data.joint_pos[:, leg_ids]), joint_max)
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
                                "dropped": bool(dropped[i]),
                                "duration_seconds": float(duration[i]),
                                "max_hold_seconds": float(max_hold[i]), "end_hold_seconds": float(end_hold[i]),
                                "leg_joint_excursion_rad": float((joint_max[i] - joint_min[i]).max())})
    finally:
        env.launch_plan = previous_plan
    caught = sum(row["success"] for row in results)
    report = {"episodes": episodes, "seed": seed, "successes": caught, "success_rate": caught / episodes,
              "settings": asdict(env.settings),
              "falls": sum(row["fell"] for row in results),
              "drops": sum(row["dropped"] for row in results),
              "failures": episodes - caught,
              "timeouts_without_hold": sum(not row["success"] and not row["fell"] and
                                           not row["dropped"] for row in results),
              "success_definition": "terminal_hold" if env.settings.strict_hug else "short_contact",
              "catch_and_survive": sum(row["success"] and not row["fell"] for row in results),
              "stable_catches": sum(not row["fell"] and not row["dropped"] and
                                    row["end_hold_seconds"] >= env.settings.required_hold_seconds for row in results),
              "mean_leg_joint_excursion_rad": sum(row["leg_joint_excursion_rad"] for row in results) / episodes,
              "held_one_second": sum(row["max_hold_seconds"] >= 1.0 for row in results),
              "by_object": {}, "results": results, "posture_trace": trace}
    for i, spec in enumerate(env.settings.objects):
        subset = [row for row in results if row["object"] == i]
        report["by_object"][spec.name] = {"episodes": len(subset), "successes": sum(row["success"] for row in subset)}
    return report


@torch.inference_mode()
def evaluate_shelf(env, policy, episodes, seed):
    """放架任务独立统计接住与完整成功，读取自动复位前的终止步快照。"""
    from dataclasses import asdict
    plan = launch_dataset(env.settings, episodes, seed)
    previous_plan = env.launch_plan
    results = []
    try:
        for offset in range(0, episodes, env.num_envs):
            count = min(env.num_envs, episodes - offset)
            ids = (torch.arange(env.num_envs) + offset).clamp_max(episodes - 1)
            env.launch_plan = {key: value[ids].to(env.device) for key, value in plan.items()}
            obs, _ = env.reset(seed=seed + offset)
            alive = torch.arange(env.num_envs, device=env.device) < count
            for step in range(env.max_episode_length + 2):
                actions = torch.zeros_like(env.actions) if policy is None else policy(TensorDict(obs, batch_size=[env.num_envs]))
                obs, reward, terminated, truncated, extras = env.step(actions)
                if not torch.isfinite(reward).all() or not torch.isfinite(obs["policy"]).all():
                    raise RuntimeError("放架评估出现非有限数值")
                metrics = extras["shelf"]
                finished = alive & (terminated | truncated)
                for i in finished.nonzero().flatten().tolist():
                    results.append({"episode": offset + i, "object": 0,
                                    "caught": bool(metrics["caught"][i]),
                                    "success": bool(metrics["success"][i]),
                                    "fell": bool(metrics["fallen"][i]),
                                    "dropped": bool(metrics["dropped"][i]),
                                    "timeout": bool(truncated[i]),
                                    "settle_seconds": float(metrics["settle_time"][i]),
                                    "duration_seconds": (step + 1) * env.step_dt})
                alive &= ~(terminated | truncated)
                env.next_throw[~alive] = float("inf")
                if not alive.any():
                    break
            if alive.any():
                raise RuntimeError("放架评估未完成整回合")
    finally:
        env.launch_plan = previous_plan
    results.sort(key=lambda row: row["episode"])
    successes = sum(row["success"] for row in results)
    return {"episodes": episodes, "seed": seed, "settings": asdict(env.settings),
            "success_definition": "caught_then_released_and_supported_on_upper_shelf",
            "successes": successes, "success_rate": successes / episodes,
            "caught": sum(row["caught"] for row in results),
            "falls": sum(row["fell"] for row in results),
            "drops": sum(row["dropped"] for row in results),
            "timeouts": sum(row["timeout"] for row in results),
            "failures": episodes - successes, "results": results}
