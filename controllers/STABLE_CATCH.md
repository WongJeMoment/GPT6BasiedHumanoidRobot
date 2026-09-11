# 主动下肢与持续抱稳训练

`env_configs/rl_stable_catch.py` 开启 `active_legs`，默认回合为 20 s，运行 `./play_stable.sh` 回放本地最佳模型。机器人仍为浮动基座，脚与地面通过真实碰撞接触支撑，不锁定机器人或当前投放物体。

下肢控制根据捕获点相对双脚的位置配合调整髋、膝、踝。重心偏前时，髋部与踝部反向协同，将骨盆移回支撑区；独立的躯干倾斜反馈用于恢复直立。来物接触时屈膝缓冲，抱持后逐渐恢复支撑高度。这是双脚支撑范围内的主动姿态调整，尚未加入跨步步态规划。

`controllers/config.py` 的 `support_shift_gain` 与 `upright_tilt_gain` 控制下肢捕获点补偿及躯干俯仰恢复，默认分别为 1.8、0.8。

PPO 策略观测扩展为 173 维，增加基座高度、捕获点误差、双脚相对位置、速度与接触力。上肢残差上限仍为 0.20 rad，下肢/腰部提高到 0.30 rad；最后仍进行目标限速和关节软限位。通过 `--warm_start` 保留旧模型的手臂网络，将新观测列初始化为零，并清除旧下肢输出均值，使用新的平衡控制开始探索。

新增奖励鼓励维持高度、将重心保持在支撑区域、减少滑脚、持续抱稳，跌倒事件总惩罚由 5 提高到 50。仅在回合结束未跌倒且当前连续抱持至少 2 s 时获得终点抱持奖励。因此短暂接住然后跌倒不算本版本的“稳定抱抓成功”。

```bash
conda activate env_isaaclab
python run.py --mode train --config env_configs/rl_stable_catch.py --num_envs 256 --iterations 300 --warm_start trained_models/body_hug_v1/model_best.pt --headless

# 续训同结构的新模型使用 --checkpoint，不能再次用 --warm_start 清除下肢输出。
python run.py --mode train --config env_configs/rl_stable_catch.py --num_envs 256 --iterations 300 --checkpoint 新模型路径.pt --headless

python run.py --mode eval --config env_configs/rl_stable_catch.py --num_envs 96 --eval_episodes 96 --eval_seed 2031 --checkpoint 新模型路径.pt --eval_output logs/stable_evaluation.json --headless
```

评估报告除原短时接物率外，增加 `catch_and_survive`（曾接住且整个回合未跌倒）、`stable_catches`（回合末仍连续抱持 ≥2 s 且未跌倒）、最终抱持时长和下肢关节实际活动幅度。下肢活动幅度从仿真关节位置计算，排除复位跳变。

## 长回合物体池修复

旧环境仅在复位/抛掷时将备用物体移到地下。PhysX 的地面穿透恢复会将它们向上推：诊断日志中，未投放球体从第 1 秒的 -19 m 上升至第 10 秒的 -1 m，在第 11 秒冒出地面，撞翻正在接箱体或圆柱体的机器人。这不是当前来物导致的正常失稳。

`g1_throw/object_pool.py` 现在每个物理步持续停放**非当前物体**；当前物体即使接住或掉落也不会被停放逻辑改写，仍由 PhysX 自由积分。修复后 20 秒验证日志中备用物体最高位置约为 -19.98 m。单元测试验证活动物体状态保持不变，物理日志验证备用物体不会重新出现。

## 本地训练过程（2026-09-11）

在 RTX 5060 Ti / Isaac Lab 2.3.2 / Isaac Sim 5.1 上完成两轮各 300 次 PPO 更新，均为 256 个并行环境，共 4,915,200 个环境步。

1. 6 秒课程：由旧身体抱抓模型迁移到主动下肢结构。输出 `logs/rl_stable_catch/stable_v2_run1`，在 seed=2030 的 48 回合验证集上选中 `model_100.pt`。
2. 12 秒续训：从上述检查点继续 300 次更新，输出 `logs/rl_stable_catch/stable_v2_long`。检查点编号延续为 100～399。
3. 修复备用物体冒出问题后，在 seed=2033、48 回合、20 秒任务上重新比较。短课程模型接住 36 次、跌倒 0 次、终点稳定抱持 6 次；续训 `model_399.pt` 接住 45 次、跌倒 0 次、终点稳定抱持 11 次，选为最终模型。增大俯仰增益的实验未被采用。

两轮训练时仍存在备用物体冒出问题，最终评估与回放使用修复后的环境。训练参数保存在模型目录的 `training_settings.json`，回放参数保存在 `resolved_settings.json`；原始日志与源码快照保留在对应训练目录。复现课程时分别给训练命令增加 `--episode_seconds 6` 与 `--episode_seconds 12`，新训练默认使用修复后的环境。

最佳权重位于 `trained_models/stable_body_hug_v2/model_best.pt`。新模型为 173 维观测，须使用 `rl_stable_catch.py`；旧 `play_rl.sh` 的 152 维模型仍保留。

## 最终独立测试

选择模型后使用独立 seed=2035，96 个回合，每回合 20 秒，球/箱/圆柱各 32 次；速度 4～5 m/s，发射距离 1～1.4 m。两个模型均使用修复后的备用物体停放逻辑，关闭探索噪声。

| 指标 | 旧身体抱抓 v1 | 主动下肢 v2 |
| --- | ---: | ---: |
| 短时身体接住（连续 ≥0.4 s） | 71/96 | 88/96（91.7%） |
| 回合内跌倒 | 96/96 | 0/96 |
| 接住且全回合未跌倒 | 0/96 | 88/96 |
| 回合末仍连续抱稳 ≥2 s 且未跌倒 | 0/96 | 23/96 |

新模型每回合最大下肢关节实际活动幅度平均为 0.267 rad（约 15.3°），说明下肢确实参与了支撑调整。这是各回合中活动最大的腿部关节的幅度平均值，不表示每个关节都运动同样幅度。

本次在指定来物范围、20 秒回合内未观察到跌倒，不代表任意来物或无限时长保证。持续抱持仍有改进空间：接住且站稳与始终不掉物是不同指标。

原始逐回合结果、姿态及备用物体高度采样保存在 [新模型报告](../trained_models/stable_body_hug_v2/holdout_20s.json) 与 [旧模型对照](../trained_models/stable_body_hug_v2/old_v1_20s.json)。最终运行源码快照为 `logs/rl_stable_catch/final_runtime_source`。18 项单元测试通过，包括模型输入迁移、终点奖励、实际投放物体不受停放逻辑干预。
