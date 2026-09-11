#!/usr/bin/env bash
# 可视化新任务：掉物自动失败、复位、再次投放。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HEADLESS=0
export LIVESTREAM=0
play_python="${ISAACLAB_PYTHON:-/home/zhe/anaconda3/envs/env_isaaclab/bin/python}"
play_checkpoint="${1:-trained_models/strict_hug_v3/model_best.pt}"
if [[ ! -f "$play_checkpoint" ]]; then
    echo "模型不存在：$play_checkpoint，请先训练或指定检查点。" >&2
    exit 1
fi
exec "$play_python" -u run.py --mode play --config env_configs/rl_hug.py --num_envs 1 \
    --steps 0 --checkpoint "$play_checkpoint"
