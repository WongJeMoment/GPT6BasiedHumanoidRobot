#!/usr/bin/env bash
# 回放本地训练的主动下肢抱抓策略；第一个参数可指定其他同结构检查点。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HEADLESS=0
export LIVESTREAM=0
play_python="${ISAACLAB_PYTHON:-/home/zhe/anaconda3/envs/env_isaaclab/bin/python}"
play_checkpoint="${1:-trained_models/stable_body_hug_v2/model_best.pt}"
if [[ ! -f "$play_checkpoint" ]]; then
    echo "模型不存在：$play_checkpoint，请传入实际检查点路径。" >&2
    exit 1
fi
exec "$play_python" -u run.py --mode play --config env_configs/rl_stable_catch.py --num_envs 1 --steps 0 --checkpoint "$play_checkpoint"
