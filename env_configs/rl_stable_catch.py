"""下肢主动支撑的身体抱抓：20 s 整回合，接到物体后继续保持站立。"""
from copy import deepcopy
from env_configs.rl_catch import CONFIG as CATCH

CONFIG = deepcopy(CATCH)
CONFIG.active_legs = True
CONFIG.episode_seconds = 20.0
CONFIG.num_envs = 128
