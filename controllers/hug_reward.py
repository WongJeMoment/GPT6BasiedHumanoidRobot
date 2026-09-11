"""主动抱抓任务：失败检测和奖励独立成纯张量函数，方便验证奖励漏洞。"""
import torch


def dropped_object(active, height, relative_position, drop_height):
    """低于抱抓高度或飞出可达范围即失败；未发射的备用物体不参与判定。

    高度按物体中心计算，0.5 m 是提前判掉物，不必等它完全触地。
    水平距离 2 m / 飞过身体后 0.6 m 用于快速结束漏接回合。
    """
    return active & ((height < drop_height) | (relative_position[:, :2].norm(dim=-1) > 2.0)
                     | (relative_position[:, 0] < -0.6))


def hug_reward(active, near, contact, upright, stable_time, required_hold, speed,
               vertical_speed, support_error, actions, action_change, elapsed,
               dropped, fallen, timeout, dt):
    """主奖励来自双臂抵胸持续抱持；短暂碰到不给一次性成功大奖。

    靠近和零散接触的辅助奖励在投放后 2 秒衰减，防止悬停刷分。
    掉物/跌倒步屏蔽所有正奖励，即使之前接住也判失败。
    """
    failed = dropped | fallen
    valid = active & (upright > 0.85) & ~failed
    both = contact[:, :2].all(-1).float()
    three = contact.all(-1).float()
    slow = torch.exp(-speed.square() / 0.5)
    curriculum = torch.exp(-elapsed.clamp_min(0) / 2.0)
    sustained = (stable_time / required_hold).clamp(0, 1)
    approach = curriculum * (2 * near + 2 * contact[:, :2].float().mean(-1) + 4 * both)
    holding = 12 * three * slow + 30 * sustained
    # 仅抱持中给小幅支撑奖励；空手站立不能积累高回报。
    balance = three * torch.exp(-support_error[:, :2].square().sum(-1) / 0.01)
    slipping = 4 * both * (-vertical_speed).clamp(0, 2)
    regularization = .02 * actions.square().sum(-1) + .015 * action_change.square().sum(-1)
    dense = (valid.float() * (approach + holding + balance - slipping)
             - .5 * active.float() - regularization) * dt
    completed = timeout & valid & (stable_time >= required_hold)
    # 超时但没有抱稳也是失败，而不是靠站满时长过关。
    event = (100 * completed.float() - 60 * dropped.float() - 80 * fallen.float()
             - 30 * (timeout & ~completed & ~failed).float())
    return dense + event
