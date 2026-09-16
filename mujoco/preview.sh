#!/usr/bin/env bash
# 本机 MuJoCo 环境；参数直接传给 environment/run.py。
set -euo pipefail
mujoco_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mujoco_python="${MUJOCO_PYTHON:-/home/zhe/anaconda3/envs/mujoco/bin/python}"
cd -- "$mujoco_project_dir"
exec "$mujoco_python" -u mujoco/environment/run.py "$@"
