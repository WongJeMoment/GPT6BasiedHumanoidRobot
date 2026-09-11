"""残差 PPO 的奖励与控制状态观测，不修改物理接触成功判据。"""
import torch

# 阶段5、接触3、计时2、收臂1、名义/发送目标46、物体姿态4、时间1、激活1、成功1。
EXTRA_OBSERVATIONS = 64


def task_reward(active, upright, proximity, contact, speed, stable_time, new_success,
                fallen, actions, action_change, dt):
    """逐步鼓励靠近→双臂承接→抵胸→稳定；成功奖励每次投放只发一次。

    所有接触奖励都要求物体在胸前范围且身体直立，地面碰撞无法获利。
    跌倒和成功属于事件奖励，不乘 dt，避免被 60 Hz 控制周期稀释。
    """
    both_arms = contact[:, :2].all(-1).float()
    torso_with_arm = (contact[:, 2] & contact[:, :2].any(-1)).float()
    three = contact.all(-1).float()
    slow = torch.exp(-speed.square() / 0.5)
    shaping = (1.5 * proximity + 1.0 * contact[:, :2].float().mean(-1)
               + 3.0 * both_arms + 5.0 * torso_with_arm + 8.0 * three
               + 8.0 * three * slow + 12.0 * (stable_time > 0).float())
    return ((0.5 * upright + active.float() * (upright > 0.85).float() * shaping
             - 0.02 * actions.square().sum(-1) - 0.015 * action_change.square().sum(-1)) * dt
            + 20.0 * new_success.float() - 5.0 * fallen.float())


def observation(env, object_state):
    """把控制器内部阶段与限速目标暴露给策略，减少部分可观测性。"""
    control, data = env.catch_controller, env.robot.data
    planner = control.planner
    active = env.active_object >= 0
    return torch.cat((
        torch.nn.functional.one_hot(planner.phase, 5).float(),
        planner.contact.float(), planner.elapsed.clamp_max(2.0)[:, None],
        planner.stable_time.clamp_max(2.0)[:, None], control.motor.draw_in[:, None],
        control.motor.target - data.default_joint_pos,
        env.joint_target - data.default_joint_pos,
        torch.where(active[:, None], object_state[:, 3:7], 0.0),
        (env.episode_length_buf * env.step_dt / env.settings.episode_seconds)[:, None],
        active.float()[:, None], planner.success.float()[:, None],
    ), dim=-1)
