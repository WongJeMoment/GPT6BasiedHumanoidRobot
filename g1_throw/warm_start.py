"""把 152 维旧接物策略迁移到 173 维主动下肢策略，不恢复旧优化器。"""
import torch


def transfer_state(source, target):
    """只扩展输入层与观测归一化缓冲；其余形状不一致直接报错。"""
    output = {}
    for key, value in target.items():
        old = source[key]
        if old.shape == value.shape:
            output[key] = old.clone()
        elif (key in ("actor.0.weight", "critic.0.weight") or "obs_normalizer._" in key):
            if old.ndim != 2 or value.ndim != 2 or old.shape[0] != value.shape[0] or old.shape[1] >= value.shape[1]:
                raise ValueError(f"不支持的观测迁移: {key}: {old.shape} -> {value.shape}")
            expanded = value.clone()
            if key.endswith("weight"):
                expanded.zero_()
            expanded[:, :old.shape[1]] = old
            output[key] = expanded
        else:
            raise ValueError(f"不支持的网络迁移: {key}")
    return output


def warm_start(runner, checkpoint, env):
    policy = runner.alg.policy
    source = torch.load(checkpoint, map_location=env.device, weights_only=False)["model_state_dict"]
    policy.load_state_dict(transfer_state(source, policy.state_dict()))
    arms = {i for arm in env.catch_controller.motor.arms for i in arm}
    legs = [i for i in range(env.robot.num_joints) if i not in arms]
    final = [m for m in policy.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        # 保留已学到的双臂抱抓均值，下肢从新的稳定支撑控制器开始探索。
        final.weight[legs] = 0
        final.bias[legs] = 0
        policy.std[:] = 0.20
        policy.std[legs] = 0.12
    print(f"WARM START: {checkpoint}; 手臂迁移，下肢均值清零，输入扩展", flush=True)
