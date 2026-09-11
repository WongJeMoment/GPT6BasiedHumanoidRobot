# 身体抱抓残差 PPO

策略在分层控制器的名义关节目标上学习 23 维修正。上肢修正上限 0.20 rad，下肢/腰部为 0.08 rad，最终仍经过速度及关节限位。训练使用原来的球、箱体和圆柱，速度保持 4–5 m/s；训练回合为 4 s，成功条件保持身体与双臂共同接触、低相对速度且直立持续 0.4 s。

`env_configs/rl_catch.py` 开启 `residual_rl`。152 维观测包含原有机器人和物体状态，以及高层阶段、接触、计时、收臂修正、名义/发送关节目标和物体姿态。原来的 88 维检查点不能用于这个配置。初始策略均值接近零修正，从现有控制器附近开始探索。

奖励逐步鼓励靠近来物、双臂承接、抵胸、三方接触和低速抱稳；成功每次抛掷额外奖励一次。跌倒为事件惩罚，动作幅度与变化受到惩罚。接触奖励要求物体处于胸前可达范围且机器人直立，成功判据未放宽。

## 训练、续训与回放

```bash
conda activate env_isaaclab
python run.py --mode train --config env_configs/rl_catch.py --num_envs 128 --iterations 200 --headless

# 替换为实际保存的模型路径；续训会创建新目录，保留原模型。
python run.py --mode train --config env_configs/rl_catch.py --num_envs 128 --iterations 500 --checkpoint logs/rl_catch/运行目录/model_199.pt --headless
python run.py --mode play --config env_configs/rl_catch.py --num_envs 1 --checkpoint logs/rl_catch/运行目录/model_199.pt
```

训练目录保存模型、合并后的环境参数、PPO 参数，以及当时的 `controllers/`、`env_configs/`、`g1_throw/` 和入口代码快照。默认每 50 次更新保存检查点，结束时也会保存。`--log_dir` 可指定尚不存在的输出目录。只回放检查点，不会重新开始训练。

## 同一组来物对照评估

```bash
python run.py --mode eval --config env_configs/rl_catch.py --num_envs 96 --eval_episodes 96 --eval_seed 2026 --checkpoint logs/rl_catch/运行目录/model_199.pt --compare_baseline --eval_output logs/rl_catch/comparison.json --headless
```

不传 `--checkpoint` 时只评估零残差。评估在独立随机源预采样来物，并按环境索引发射；某个环境提前跌倒不会改变其他环境的来物。每回合只计一次，自动重置后不重复统计；抛掷前跌倒也计入失败分母。报告包括总成功率、每类物体成功数、跌倒数、持续抱持至少 1 s 的次数与每回合明细。

模型选择可使用一组种子，最终对比需使用未参与选择的种子。短时间训练不保证提升；以固定测试集上的真实抱稳结果判断效果，不以训练奖励上升替代成功率。

## 本轮训练与独立测试结果

2026-09-10 完成 200 次 PPO 更新、819,200 步环境交互，128 个并行环境，训练计时约 9 分 46 秒。验证种子 2026 的 48 组来物上，检查点 50 / 100 / 150 / 199 分别成功 16 / 29 / 32 / 29 次，因此选用 `model_150.pt`，复制为 `model_best.pt`。没有使用独立测试结果重新选择模型。

2026-09-11 使用独立种子 2027、96 个并行环境和 96 个完整回合对照测试，每类物体各 32 次：

| 指标 | 零残差分层控制 | PPO 残差策略 |
| --- | --- | --- |
| 身体与双臂抱稳 ≥0.4 s | 8/96（8.3%） | 76/96（79.2%） |
| 连续抱稳 ≥1 s | 0/96 | 39/96 |
| 球体成功 | 8/32 | 32/32 |
| 箱体成功 | 0/32 | 14/32 |
| 圆柱成功 | 0/32 | 30/32 |
| 回合结束前跌倒 | 96/96 | 96/96 |

**提升是短时接住并抱稳的能力；接住后的持续站立仍未解决，箱体也明显弱于球和圆柱。** 这组有限样本的结果不代表其他速度、物体或长时间任务的成功率。

直接回放本轮模型：

```bash
./play_rl.sh
```

仓库内可直接回放的模型位于 `trained_models/body_hug_v1/model_best.pt`，并附带模型选择记录和独立测试摘要。完整训练过程仍保存在本地 `logs/rl_catch/ppo_body_hug_v1/`，包括中间检查点、训练日志、源代码快照、参数和每回合评估明细；这些大型过程文件不纳入 Git。14 项单元测试通过，并完成所选模型加载后的 120 步回放检查。
