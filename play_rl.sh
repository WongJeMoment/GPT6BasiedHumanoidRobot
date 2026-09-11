#!/usr/bin/env bash
# 可视化回放已训练的身体抱抓策略；参数一可覆盖模型路径。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HEADLESS=0
export LIVESTREAM=0
play_python="${ISAACLAB_PYTHON:-/home/zhe/anaconda3/envs/env_isaaclab/bin/python}"
play_checkpoint="${1:-trained_models/body_hug_v1/model_best.pt}"
if [[ ! -f "$play_checkpoint" ]]; then
    echo "模型不存在：$play_checkpoint，请传入实际检查点路径。" >&2
    exit 1
fi
exec "$play_python" -u run.py --mode play --config env_configs/rl_catch.py --num_envs 1 --steps 0 --checkpoint "$play_checkpoint"
