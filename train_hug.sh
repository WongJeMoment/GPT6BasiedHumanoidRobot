#!/usr/bin/env bash
# 在本机 GPU 前台训练；Ctrl+C 停止，检查点与 TensorBoard 日志保存在 logs/ 下。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
train_python="${ISAACLAB_PYTHON:-/home/zhe/anaconda3/envs/env_isaaclab/bin/python}"
export HEADLESS=1
export LIVESTREAM=0
train_init=()
# 从当前站稳策略迁移；续训/从零训练通过相应选项切换。
if [[ "${1:-}" == "--resume" ]]; then
    train_init=(--checkpoint "${2:?请提供续训检查点}")
    shift 2
elif [[ "${1:-}" == "--scratch" ]]; then
    shift
else
    train_init=(--init_policy trained_models/stable_body_hug_v2/model_best.pt)
fi
exec "$train_python" -u run.py --mode train --config env_configs/rl_hug.py \
    --num_envs "${HUG_ENVS:-256}" --iterations "${HUG_ITERATIONS:-1500}" \
    "${train_init[@]}" --headless "$@"
