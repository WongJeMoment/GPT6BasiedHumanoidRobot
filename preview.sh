#!/usr/bin/env bash
# 在本机桌面打开单环境可视化；传入配置路径可切换场景。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HEADLESS=0
export LIVESTREAM=0
preview_python="${ISAACLAB_PYTHON:-/home/zhe/anaconda3/envs/env_isaaclab/bin/python}"
exec "$preview_python" -u run.py --mode preview --config "${1:-env_configs/catch_and_place.py}" --num_envs 1 --steps 0
