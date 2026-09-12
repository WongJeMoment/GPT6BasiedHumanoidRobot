# GPT6BasiedHumanoidRobot

基于 Isaac Lab 的 Unitree G1 高动态物体分层具身抓取与抗扰动训练环境。

## 接箱后放架环境（2026-09-12）

新增独立配置 [catch_and_place.py](env_configs/catch_and_place.py)：**先接住箱子，再松手把箱子放稳在上层架面，才算完整成功**。本阶段实现环境、奖励、观测和验证接口；GPT-6 动作规划尚未接入，零动作预览不会自动演示完整任务。原有抱接配置、控制器和模型保留原用途。

| 项目 | 默认设置 |
| --- | --- |
| 箱体与投放 | 35 × 35 × 30 cm、0.70 kg；6–8 m/s，瞄准点左右随机偏移 ±0.65 m |
| 架子 | 机器人左前方；上层架面中心 `(0.45, 0.65, 0.75)` m，架面 60 × 55 cm，板厚 4 cm；下层高 25 cm；四根架腿 |
| 物理 | 架板/架腿为有碰撞和摩擦的固定运动学刚体；箱体保持自由刚体，通过接触承重，不吸附、不自动移动到目标 |
| 接住判据 | 双臂持续接触 0.20 s、相对速度低于 0.65 m/s、机器人直立，且箱体在接住前未接触架面；先落架再提起不能补算接住 |
| 放置判据 | 已接住；旋转后整个箱体投影位于架面内且留 2.5 cm 余量；底面贴近上层架面，姿态接近竖直；架面承担至少 70% 重量；所有机器人刚体都已松开 |
| 放稳判据 | 上述条件连续保持 0.75 s，箱体线速度低于 0.10 m/s、角速度低于 0.25 rad/s；成功立即结束并局部重置 |
| 失败 | 箱体底面降至离地 2.5 cm 以内、飞出活动范围或机器人跌倒立即失败；12 s 超时仍未完成也失败。只有接住、直接落架、悬空托住均不能成功 |

坐标以每个环境原点为基准，机器人初始前方为 +X、左方为 +Y。架面位置、尺寸和时间/速度阈值都在 `CONFIG.shelf` 中修改。

动作接口为 **23 维归一化关节目标偏移**：`默认姿态 + 1.5 × action`（rad），经过软限位和 6 rad/s 的目标变化率限制；手指仍为 0 自由度。该场景使用 `joint` 接口，方便后续规划器/策略驱动完整放置运动，不能直接混用原有小残差抱接策略。观测为 **152 维**，额外包含架面目标、箱体姿态/角速度、接触与承重、任务阶段、接住/放稳计时、历史最短距离、先碰架标记和实际发送的关节目标。

奖励包含接触辅助、一次性接住奖励、接近放置目标的进度、松手后的静置与完整任务成功奖励；跌倒、掉落及超时扣分。接近目标只奖励刷新最短距离，防止往返刷分。`extras["shelf"]` 保留重置前的成功/失败快照；训练日志分别记录 `Episode/box_caught` 和 `Episode/place_success`。

```bash
# 本机可视化，只预览环境
bash preview.sh env_configs/catch_and_place.py

conda activate env_isaaclab

# 检查实际碰撞、架面承重、判据、局部复位和重新投放
python run.py --mode smoke --config env_configs/catch_and_place.py --num_envs 2 --steps 180 --headless

# 从零训练该任务的本地 PPO 策略（此命令不会调用 GPT）
python run.py --mode train --config env_configs/catch_and_place.py --num_envs 64 --iterations 1500 --headless

# 固定来物评估；省略 --checkpoint 时评估零动作基线
python run.py --mode eval --config env_configs/catch_and_place.py --num_envs 4 --eval_episodes 8 --headless --eval_output logs/shelf_eval.json
```

现有 v1/v2/v3 模型的动作和观测与该场景不一致，不能直接作为放架策略加载。仿真检查会专门布置箱体、接住历史及机器人站姿以验证成功分支，**不代表已有策略自主完成了接箱放架**；实际训练和预览中没有这些测试辅助。

本机验证：34 项单元测试通过；新场景 2 个环境、180 步 PhysX 检查通过，覆盖真实架面承重、完整成功、掉落/超时、动作限幅限速与局部重置；4 个环境完成 2 次 PPO 迭代，生成检查点并成功加载，零动作与检查点各完成 5 个固定来物评估回合，均为 0 次完整成功。原 `rl_hug.py` 的 2 环境、180 步投放与掉物复位检查也通过。短训练产物只用于接口验证。

## 上一版本更新（2026-09-11）

本次将本地的主动下肢控制、严格抱持训练、物体池修复、测试及 v2/v3 模型同步到仓库。此前提交 `aa35110` 已将抛掷速度与横向目标范围调整为以下新默认值。

| 改动 | 当前行为与对应文件 |
| --- | --- |
| 更快、偏侧的来物 | 初速度由 4–5 提高到 **6–8 m/s**；`target_lateral_range=(-0.65, 0.65)` 随机偏移胸前高度的瞄准点。发射距离仍为 1.0–1.4 m，目标高度仍为 1.0 m。见 `env_configs/common.py`、`g1_throw/environment.py`。 |
| 主动下肢支撑 | `active_legs` 开启髋、膝、踝的支撑与姿态反馈，抱持后逐渐恢复支撑高度；增加双足接触观测，策略观测由 152 扩展到 173 维。见 `controllers/whole_body.py`、`controllers/residual_rl.py`、`env_configs/rl_stable_catch.py`。 |
| 严格抱持任务 | `rl_hug.py` 默认 8 秒回合；物体中心低于 0.50 m、飞出范围或机器人跌倒立即失败并局部复位。成功要求回合结束时仍连续抱稳至少 2 秒。见 `controllers/hug_reward.py`。 |
| 物体池修复 | 每个物理步持续停放备用物体，解决长回合中备用刚体被地面穿透恢复推回场景的问题；当前投放物体继续由 PhysX 自由积分。见 `g1_throw/object_pool.py`。 |
| 训练与迁移 | 增加 `--warm_start` 扩展旧策略输入、`--init_policy` 迁移动作网络、`--episode_seconds` 覆盖回合时长；严格任务使用对应 PPO 参数和有限时长回报。见 `run.py`、`g1_throw/warm_start.py`。 |
| 评估与检查 | 固定来物计划包含横向偏移；报告增加掉物、终点抱持、下肢实际活动幅度与姿态采样。新增奖励、策略迁移、物体池及实际掉物复位检查；投放检查同步适配随机偏移弹道。 |
| 模型与启动脚本 | 包含 `trained_models/stable_body_hug_v2/`、`trained_models/strict_hug_v3/` 的权重、参数和评估报告，以及 `play_stable.sh`、`play_hug.sh`、`train_hug.sh`。 |

本机快速运行：

```bash
bash preview.sh       # 当前 6–8 m/s、随机偏移场景，零残差分层控制
bash play_stable.sh   # 当前场景下回放主动下肢 v2，20 秒回合
bash play_hug.sh      # 当前场景下回放严格抱持 v3，8 秒回合
bash train_hug.sh     # 默认迁移 v2 动作网络，训练严格抱持任务
```

训练支持续训与从零初始化，详见 [主动抱抓训练](controllers/LOCAL_HUG_TRAINING.md)；下肢支撑与长回合设置见 [主动下肢说明](controllers/STABLE_CATCH.md)。

**模型成绩对应旧来物分布：4–5 m/s、瞄准机器人中心。** v2 的独立 96 个 20 秒回合中，短时接住 88 次、跌倒 0 次、终点连续抱稳至少 2 秒 23 次；v3 在严格判据的独立 96 个 8 秒回合中成功 47 次、掉物 13 次、跌倒 0 次、超时未抱稳 36 次。原始统计分别见 [v2 报告](trained_models/stable_body_hug_v2/holdout_20s.json) 和 [v3 报告](trained_models/strict_hug_v3/holdout.json)。两种任务的成绩不能直接比较。

模型参数快照保留实际训练/评估时的旧设置。回放入口使用命令行选定的当前配置，**不会自动恢复快照中的旧来物分布**；复现旧场景时需将 `speed_range=(4.0, 5.0)`、`target_lateral_range=(0.0, 0.0)` 写入对应配置副本。当前高速偏侧场景尚未重新训练或完成成功率评估；控制器仍为站定抱接，拦截点横向可达判断为 ±0.42 m，部分偏远来物可能不会触发拦截。

本次发布检查：22 项单元测试、全部五种场景配置校验、Python/启动脚本语法检查及 v2/v3 权重 SHA-256 校验通过。当前 `rl_hug.py` 在 2 个并行环境、180 个控制步的 PhysX 冒烟检查中通过投放速度/偏移弹道、掉物负奖励、局部自动复位、重新投放和有限数值检查；该短检查中 3 次抛掷、严格成功 0 次，仅验证运行流程。

## v0.1.0 首个公开版本

- 新增分层具身控制架构，高层执行 `READY → INTERCEPT → ABSORB → HOLD → RECOVER` 任务拆解。
- 新增低层全身协同动作生成，使用双臂阻尼逆运动学、屈膝缓冲、踝髋姿态反馈和质心捕获点反馈。
- 新增双掌及手指接触力检测，仅在双手持续接触、物体低相对速度且机器人保持直立时判定抱稳成功。
- 单次抛掷场景默认启用分层控制，并保留原有关节动作和 PPO 残差训练模式。
- 支持球体、箱体、圆柱体及 STL 自定义物体，包含单次和连续随机抛掷配置。
- 增加控制器单元测试、Isaac Sim 冒烟检查、局部环境重置检查和成功率统计。
- 本机两组短时验证共完成 40 次抛掷，其中 26 次达到持续 0.4 秒的抱稳判据。

## 环境说明

当前身体抱抓更新：拦截后将物体收向胸部，以躯干后方支撑、双臂两侧包围和双掌前下方托举共同抓取。成功判据现要求躯干、左臂、右臂共同接触；上方 v0.1.0 的 26/40 结果使用旧版双手接触判据，不适用于本次更新。具体控制参数见 [控制说明](controllers/README.md)。

新增残差强化学习：`env_configs/rl_catch.py` 配置使用 PPO 学习分层关节目标上的小幅修正，加入接物奖励和控制阶段观测。训练与固定来物对照评估命令见 [RL 训练说明](controllers/RL_TRAINING.md)。可运行 `./play_rl.sh 模型路径` 回放训练结果，`./preview.sh` 仍用于零残差控制。

旧版 v1 在原 4–5 m/s、瞄准中心场景中的独立测试：同样 96 组来物，短时身体抱稳成功数从 8 次提升到 76 次（8.3% → 79.2%），其中 39 次持续抱稳超过 1 秒；回合后续仍全部发生跌倒。仓库保留该模型，运行 `./play_rl.sh` 可在当前场景中回放。历史统计和适用范围见 [训练结果](controllers/RL_TRAINING.md#本轮训练与独立测试结果)。

基于本机 Isaac Lab **2.3.2**、Isaac Sim 和 RSL-RL。机器人为官方 Unitree G1：身体保留 23 个可动关节，双手共 14 个手指关节在加载时改为 **USD FixedJoint**，不进入动作空间。保留官方手部几何，固定在零角度手型，覆盖黑色非金属塑胶外观；不是通过高刚度电机模拟零自由度，也不是重新制作手的 CAD 外形。机器人底座不固定。

单次抛掷场景现默认使用**分层具身抱接控制**：高层拆解准备、拦截、缓冲、抱持和恢复任务，低层生成双臂与下肢协同关节目标。实现与运行说明见 [controllers/README.md](controllers/README.md)。这是需要仿真评估和调参的反馈控制基线，不代表已训练出稳定接球策略。连续场景仍默认使用原来的站立抗扰动任务。黑色塑胶目前是外观材质；质量和惯量沿用官方手部资产。

## 文件位置

```text
env_configs/                 ← 环境配置全部放在这个独立目录
  common.py                  # 参数定义、默认物体、校验
  single_throw.py            # 每回合抛一次
  continuous_throw.py        # 同一回合内连续抛掷
  rl_catch.py                # 基础身体抱抓残差训练
  rl_stable_catch.py         # 主动下肢与 20 秒持续支撑
  rl_hug.py                  # 掉物立即失败、终点连续抱稳任务
  catch_and_place.py         # 接箱后放到左前方架子的独立任务
controllers/                 ← 分层接物控制代码（含中文注释）
  config.py                  # 控制参数
  planner.py                 # 高层弹道预测、任务阶段及抱稳判据
  whole_body.py              # 低层双臂 IK、下肢平衡与缓冲
  hug_targets.py             # 胸前围合、掌肘目标及闭环收臂
  hierarchical.py            # 仿真状态/接触力适配
  residual_rl.py             # 残差策略观测、接物及支撑奖励
  hug_reward.py              # 严格抱持奖励与漏接/掉物判定
g1_throw/
  robot.py                   # 官方 G1 加载、固定手关节、黑色塑胶材质
  environment.py             # 场景搭建、物理抛掷、观测、奖励、重置
  shelf_task.py              # 架体几何、接箱/松手/承重/放稳判据及奖励
  ppo.py                     # PPO 超参数
  object_pool.py             # 备用物体持续隔离
  evaluation.py              # 固定来物评估与逐回合报告
  warm_start.py              # 旧模型输入扩展与动作迁移
tests/                       # 规划、奖励、迁移与物体池单元测试
trained_models/              # v1/v2/v3 权重及历史评估报告
run.py                       # 预览、训练、回放、评估、仿真检查入口
```

## 启动

本机直接执行 `./preview.sh` 打开可视化窗口，默认单次抛掷与分层抱接、单环境、实时速度持续运行，关闭窗口或按 Ctrl+C 退出。切换连续场景：`./preview.sh env_configs/continuous_throw.py`。只编辑配置文件不会自动启动仿真。

在这个项目目录运行。本机已有 `env_isaaclab` conda 环境：

```bash
conda activate env_isaaclab
cd /home/zhe/GPTControl

# 可视化预览（单次默认分层抱接；连续默认站立）
python run.py --config env_configs/single_throw.py --num_envs 1 --steps 3600
python run.py --config env_configs/continuous_throw.py --num_envs 1 --steps 3600

# 显式切换控制方式
python run.py --config env_configs/single_throw.py --controller joint --num_envs 1
python run.py --config env_configs/continuous_throw.py --controller hierarchical --num_envs 1

# 后台训练；按显存容量调整环境数量
python run.py --mode train --config env_configs/continuous_throw.py --num_envs 64 --headless --iterations 1500

# 回放训练结果；必须使用与检查点相同的物体列表和动作/观测结构
python run.py --mode play --config env_configs/continuous_throw.py --num_envs 1 --checkpoint logs/continuous_throw/时间目录/model_1499.pt --steps 3600

# 仿真检查
python run.py --mode smoke --num_envs 2 --steps 240 --headless
```

也可以直接使用 `/home/zhe/anaconda3/envs/env_isaaclab/bin/python`。换机器需先安装兼容的 Isaac Lab 2.3.2 / Isaac Sim 和 RSL-RL；本工程没有自动替换现有仿真软件。首次加载需要访问 NVIDIA 的官方 G1 USD 及其引用资产；也可在配置中设置 `robot_usd` 为同结构的本地资产路径。Isaac Lab 3.x API 不在此实现的适配范围内。

## 修改环境

当前默认采用大物体：单次场景为直径 40 cm 的球、35×35×30 cm 箱体、直径 32 cm / 高 40 cm 圆柱；连续场景 STL 最长边为 35–55 cm。物体质量为 0.4–0.7 kg，发射距离 1.0–1.4 m、初速度 6–8 m/s、目标高度 1.0 m，瞄准点相对机器人中心左右随机偏移最多 0.65 m；连续抛掷间隔为 4–6 s。基础分层模式在站立奖励上增加躯干与双臂共同接触且低速抱持的奖励；`rl_hug.py` 使用独立的严格终点抱持奖励。仓库包含旧分布训练的模型，当前来物分布的表现需重新评估。

连续抛掷场景现已使用 **8 个下载的 STL 模型**，文件位于 `assets/stl/`，来源和清单见 [模型说明](assets/stl/README.md)。运行 `./preview.sh env_configs/continuous_throw.py` 即可观看随机抛出这些模型。

加入自己的 STL：把文件放进 `assets/stl/`，在场景的 `objects` 列表中添加：

```python
ObjectSpec("my_object", "stl", (0.15,), 0.20, (0.8, 0.3, 0.1), "assets/stl/my_object.stl")
```

这里 `0.15` 表示最长边 15 cm，保持原模型比例；`0.20` 表示质量 0.20 kg。STL 自动转换为 USD 缓存并加入动态刚体和凸包碰撞；支持与球/方块等基础形状混合使用。孔洞和凹槽在凸包碰撞中会填实。导入依赖 `trimesh`，本机环境已安装。8 物体场景的观测维度为 93，之前 3 物体的旧检查点不能直接回放，应使用该配置重新训练。

编辑对应的 `env_configs/*.py` 后**重新启动进程**即可生效，不需要修改环境核心代码；不是运行中的热重载。新增场景可以复制现有文件，以 `--config env_configs/新场景.py` 选择，无需额外注册。

例如，把一个场景文件改为：

```python
from env_configs.common import ObjectSpec, Settings

CONFIG = Settings(
    continuous=True,
    speed_range=(6.0, 12.0),      # 每次初速度大小随机，单位 m/s
    interval_range=(2.0, 4.0),   # 两次抛掷间隔，单位 s
    distance_range=(1.5, 2.5),   # 机器人到发射点的水平距离，单位 m
    launch_height_range=(0.8, 1.3),
    azimuth_range=(-0.65, 0.65), # 前方扇区，单位 rad
    target_height=0.85,
    target_lateral_range=(-0.65, 0.65),  # 瞄准点相对机器人中心的 Y 偏移，m
    objects=[
        ObjectSpec("ball", "sphere", (0.06,), 0.2, (0.9, 0.2, 0.1)),
        ObjectSpec("box", "cuboid", (0.1, 0.1, 0.15), 0.3, (0.1, 0.3, 0.9)),
        ObjectSpec("can", "cylinder", (0.045, 0.14), 0.15, (0.9, 0.7, 0.1)),
    ],
)
```

`sphere.size` 是半径；`cuboid.size` 是 XYZ 全长；`cylinder.size` 是半径、高度。质量单位 kg。可在同一文件覆盖 `hand_color`、`hand_roughness`、`physics_dt`、`decimation`、回合长度和奖励权重。过低初速度、非法尺寸和过小环境间距会报错。

每次抛掷随机选择物体、速度、位置和横向目标偏移；物体种类超过一种时，相邻两次保证不同（包括跨回合）。速度是连续均匀采样，不保证有限精度数值永不重复。每个并行环境独立调度，`--seed` 控制随机种子。速度方向采用重力补偿的低弹道，瞄准发射时机器人水平位置加横向偏移、指定高度处的目标；这是胸前高度处的瞄准位置，不是地面落点，机器人移动后不保证命中。`target_lateral_range=(0.0, 0.0)` 可恢复瞄准中心。

每个环境预先创建物体池，每次只投放一个，其余在每个物理步持续停放于地下；下一次投放会回收上一物体。因此连续模式是依次抛掷，不是多物体同时飞行。不同并行环境启用碰撞隔离。高速小物体可能需要减小 `physics_dt` 以降低离散碰撞漏检。

策略观测包括基座速度、重力方向、关节状态、上一步动作、当前物体相对位置/速度、类型和采样初速度。训练目录保存模型、原始场景文件和合并后的参数快照。修改物体列表会改变观测维数，需要重新训练。

实现参考：本机 Isaac Lab 2.3.2 的 `direct/cartpole` 示例与 `isaaclab_assets/robots/unitree.py`。官方源码：[Isaac Lab](https://github.com/isaac-sim/IsaacLab)。

## 本机验证结果

2026-09-09，在 Isaac Lab 2.3.2 / Isaac Sim 5.1 / RTX 5060 Ti 上完成：

- 单次场景：2 个并行环境、240 步，发生 6 次跨回合抛掷，检查通过。
- 连续场景：2 个并行环境、360 步，发生 10 次抛掷，检查通过。
- 检查覆盖 14 个手部 FixedJoint、无手部角度驱动、掌部和指节黑色材质绑定、23 维动作、88 维有限观测、有限奖励、相邻物体切换、初速度大小、弹道目标高度、局部重置。
- PPO：4 个环境、2 次迭代完成，生成 `logs/continuous_throw/20260909_220643/model_1.pt`；加载该模型回放 20 步成功。
- Python 编译和配置校验通过。

上述检查验证环境和训练流程可运行，不代表策略收敛或已掌握抗扰动能力。零动作预览可能跌倒并自动重置；长回合稳定性需通过正式训练评估。

STL 扩展验证：8 个下载文件的 SHA-256 校验通过；2 个并行环境运行 240 步通过。检查覆盖全部 8 种物体的初速度、弹道、实际 PhysX 质量、网格居中/最长边尺寸及凸包碰撞。4 环境、2 次 PPO 迭代完成，新模型为 `logs/continuous_throw/20260909_221620/model_1.pt`（93 维观测，仅用于流程验证）。

大物体配置验证：单次场景和连续 STL 场景各运行 2 个环境、180 步通过；覆盖全部 3 / 8 种物体的速度、弹道、质量及 STL 缩放检查。之前的小物体模型未针对新尺寸重新训练。

2026-09-10 分层控制验证：单次场景两个随机种子合计 40 次抛掷，26 次达到持续 0.4 s 的双手接触、低相对速度及直立判据；成功后仍可能掉物或跌倒。六项单元测试及原关节模式回归检查通过。详细测试范围和日志位置见 [控制说明](controllers/README.md)。
