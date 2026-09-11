"""身体抱抓 PPO 残差训练：继承单次抛掷来物分布和三方接触成功判据。"""
from copy import deepcopy
from env_configs.single_throw import CONFIG as SINGLE_THROW

CONFIG = deepcopy(SINGLE_THROW)
CONFIG.residual_rl = True
CONFIG.num_envs = 128
CONFIG.episode_seconds = 4.0  # 缩短失败后等待，提高有效接物样本占比
CONFIG.catch_control.residual_scale = 0.20  # 上肢残差 rad；下肢在控制适配层缩小至 40%
