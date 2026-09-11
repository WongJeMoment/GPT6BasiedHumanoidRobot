"""本地 PPO 主动抱抓：掉物立即失败并重置，站着不接物不能赚取奖励。"""
from copy import deepcopy
from env_configs.rl_stable_catch import CONFIG as BASE

CONFIG = deepcopy(BASE)
CONFIG.strict_hug = True
CONFIG.episode_seconds = 8.0
CONFIG.num_envs = 256
CONFIG.drop_height = 0.50
CONFIG.required_hold_seconds = 2.0
