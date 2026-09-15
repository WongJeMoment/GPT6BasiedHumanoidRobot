# GPT6BasiedHumanoidRobot

基于 Isaac Lab 的 Unitree G1 抛物与接箱放架环境。当前仓库只包含环境、物理资产、场景配置、预览入口与环境检查；外部程序可通过 `reset()` / `step(actions)` 接入动作。

## 本次改动（2026-09-13）

- 清理旧分层控制器、PPO 训练器、策略迁移、策略评估与回放代码，以及对应脚本、配置、测试、训练记录和模型文件。
- 解除环境对上述模块的依赖；保留物理执行器、关节动作接口、任务奖励、观测、终止判据和局部复位。
- 默认运行场景改为 `env_configs/catch_and_place.py`。`run.py` 仅提供 `preview` 和 `smoke`；旧的训练、回放、模型参数已移除。
- 包含最新投放设置：X 随机 3–5 m，Y 独立随机 ±0.4 m，初速度 7.5–8 m/s；地面添加分米标尺与矩形投放范围。
- 固定来物采样保留在 `g1_throw/launch.py`，方便复现环境输入；STL 模型及其来源、许可证继续保留。

旧控制与训练版本可从 Git 历史中的提交 `c756810` 查阅。当前环境尚未接入 GPT 动作规划或自动接箱放架策略；零动作预览用于检查场景，不会自动完成任务。

## 运行

本机已使用 Isaac Lab 2.3.2 / Isaac Sim 5.1 的 `env_isaaclab` Python 环境验证；STL 场景另外使用 `trimesh`。环境代码不再直接依赖 RSL-RL 训练器。

```bash
# 本机桌面预览默认接箱放架场景
bash preview.sh

# 切换场景
bash preview.sh env_configs/single_throw.py
bash preview.sh env_configs/continuous_throw.py

conda activate env_isaaclab

# 直接启动可视化；关闭窗口或按 Ctrl+C 退出
python run.py --mode preview --num_envs 1

# 实际 PhysX 检查
python run.py --mode smoke --num_envs 2 --steps 180 --headless
python run.py --mode smoke --config env_configs/single_throw.py --num_envs 2 --steps 180 --headless
python run.py --mode smoke --config env_configs/continuous_throw.py --num_envs 2 --steps 180 --headless

# 不启动 Isaac Sim 的环境逻辑测试
python -m unittest discover -s tests -v
```

`preview.sh` 默认使用 `/home/zhe/anaconda3/envs/env_isaaclab/bin/python`，其他机器可通过 `ISAACLAB_PYTHON=/你的/python路径 bash preview.sh` 指定解释器。首次加载需能访问 NVIDIA 的官方 G1 USD 及其引用资产，也可在场景配置中把 `robot_usd` 设为同结构的本地资产路径。

可视化预览默认持续运行；`--steps 180` 可限制步数。无窗口运行默认 600 步；`--seed` 设置环境随机种子。配置修改后需要重新启动进程。

### 终端状态与手动保存 TXT

启动 `bash preview.sh` 后，机器人终端会显示箱子的位置、姿态、速度与接住/放架阶段，机器人的位置、姿态、速度与跌倒状态，以及左臂/手、右臂/手、躯干、架面的接触和接触力、接触刚体名称、承重情况、箱子是否掉落。位置相对各环境原点，单位 m；姿态为 wxyz 四元数。接触和掉落判据使用默认接箱放架场景的传感器与任务数据，其他场景会明确标注未启用的检测。

- 状态每 0.5 仿真秒输出一次；接触、任务阶段变化及掉落、跌倒、超时等回合结束事件立即输出。结束步保留自动复位前的物理状态。
- 点击运行机器人的终端，按 **`s`**（无需回车），将本次运行截至按键时已输出的全部状态保存为 UTF-8 `.txt`。
- 默认目录为项目下 `logs/robot_status/`，文件名为 `robot_status_日期_时间_微秒.txt`；终端会打印完整路径。再次按 `s` 会生成新的完整记录文件。
- **不按 `s` 不写状态文件，也不创建保存目录**；关闭窗口、Ctrl+C 或运行结束均不会自动保存。按键后产生的新记录需要再次按 `s` 才会写盘；记录在运行期间留在内存中。

可以用 `python run.py --mode preview --status_dir /你的/保存目录` 修改保存位置。按键功能需要交互终端，`smoke` 检查模式不启用手动导出。

## 默认接箱放架任务

**先在空中接住箱子，再松手把箱子放稳在上层架面，才算完整成功。**

| 项目 | 设置 |
| --- | --- |
| 箱体 | 35 × 35 × 30 cm，0.70 kg，自由刚体 |
| 发射位置 | 相对发射时的机器人基座，X 独立随机 3–5 m，Y 独立随机 −0.4～+0.4 m |
| 初速度与瞄准点 | 7.5–8 m/s，补偿重力的低弹道；瞄准点相对机器人左右随机偏移 ±0.65 m，高度 1.0 m |
| 地面标尺 | 小格和刻度为 1 dm = 0.1 m，每 5 dm 标数、每 10 dm 加粗；投放边界标注 X 为 30～50 dm、Y 为 −4～+4 dm |
| 架子 | 左前方，上层架面中心 `(0.45, 0.65, 0.75)` m，60 × 55 cm，板厚 4 cm；下层高 25 cm，四根固定架腿 |
| 接住 | 双臂持续接触 0.20 s，相对速度低于 0.65 m/s，机器人直立；箱体接住前不能先碰架子 |
| 放稳 | 箱体完整投影位于架面内并留 2.5 cm 余量，底部贴近架面，姿态接近竖直；架面承重至少 70%，所有机器人刚体松开 |
| 成功 | 放稳条件持续 0.75 s，线速度低于 0.10 m/s、角速度低于 0.25 rad/s；立即结束回合并局部复位 |
| 失败 | 箱体底部降至离地 2.5 cm 以内、离开活动范围或机器人跌倒；12 s 超时仍未完成也失败 |
| 并行空间 | 环境间距 12 m，箱体活动范围为环境原点周围半径 7 m |

地面标尺与架子固定在各环境坐标系中，机器人初始站位为 XY 零点，初始前方为 +X、左方为 +Y。机器人移动后，实际发射位置随发射时的基座位置平移；地面高亮区域仍表示初始站位下的投放范围。标尺仅用于显示，不参与碰撞或观测。

初速度下限 7.5 m/s 用于保证最远偏侧目标仍有有效弹道。只接住箱子、直接落到架上、悬空托住，或先碰架再提起，都不能算完整成功。

## 环境接口

`G1ThrowEnv` 继承 Isaac Lab 的 `DirectRLEnv`，须在 `AppLauncher` 启动后导入。保留标准接口，便于后续从仓库外接入规划器或训练程序：

```python
from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
env = None
try:
    import torch
    from g1_throw.config_loader import load_settings
    from g1_throw.environment import G1ThrowEnv, make_env_cfg

    settings = load_settings("env_configs/catch_and_place.py")
    settings.num_envs = 2
    env = G1ThrowEnv(make_env_cfg(settings))
    obs, info = env.reset()
    actions = torch.zeros((env.num_envs, 23), device=env.device)
    obs, reward, terminated, truncated, info = env.step(actions)
finally:
    if env is not None:
        env.close()
    launcher.app.close()
```

- 动作：23 维归一化关节目标偏移，裁剪至 `[-1, 1]`；手部为 0 自由度。接箱放架场景使用 `默认关节姿态 + 1.5 × action`（rad），施加软限位和 6 rad/s 的目标变化率限制。零动作对应默认姿态目标。
- 观测：通过 `obs["policy"]` 返回。接箱放架场景为 152 维，包含机器人状态、来物状态、架面目标、接触与承重、任务阶段、计时和实际关节目标。
- 奖励：接触辅助、一次性接住奖励、刷新最短放置距离的进度、松手静置、完整成功奖励，以及跌倒、掉落、超时惩罚。奖励属于环境任务定义。
- 终止信息：`info["shelf"]` 保留局部复位前的成功、接住、接触、承重、计时、跌倒和掉落快照；`info["log"]` 提供回合统计。

环境只执行输入动作并判断任务结果，没有内置抓取轨迹、规划器或学习算法。

## 场景与参数

| 配置 | 用途 |
| --- | --- |
| `env_configs/catch_and_place.py` | 默认接箱放架，远距离投放与分米标尺 |
| `env_configs/single_throw.py` | 单次抛掷球、箱体或圆柱，每回合一次；基础站立奖励 |
| `env_configs/continuous_throw.py` | 连续随机抛掷 8 个 STL 模型，间隔 4–6 s，相邻种类不重复；基础站立奖励 |

默认任务的发射位置配置为：

```python
distance_range=(3.0, 5.0),
launch_lateral_range=(-0.4, 0.4),
speed_range=(7.5, 8.0),
target_lateral_range=(-0.65, 0.65),
ground_ruler=True,
```

设置 `launch_lateral_range` 时，`distance_range` 表示 X 偏移，两轴独立采样；设为 `None` 时按水平距离和 `azimuth_range` 方位角采样。`target_lateral_range` 是瞄准点的横向偏移，与发射位置的 Y 范围分开设置。架面位置、尺寸和成功阈值在 `CONFIG.shelf` 中修改。

基础场景可使用球、长方体、圆柱或 STL。`ObjectSpec.size` 分别为球半径、长方体 XYZ 全长、圆柱半径/高度，或 STL 最长边长度；单位为 m，质量为 kg。STL 自动居中并转换到 `assets/usd_cache/`，使用凸包碰撞；孔洞和凹槽不会保留为可穿过的碰撞空间。资产来源与许可见 [STL 模型说明](assets/stl/README.md)。

## 目录

```text
env_configs/       # 三个环境场景、共享参数及校验
g1_throw/
  environment.py   # 场景、动作执行、观测、奖励、终止和复位
  shelf_task.py    # 架子物理与接箱放架任务判据
  robot.py         # 官方 G1、黑色固定手部、执行器参数
  ground_ruler.py  # 分米标尺与投放范围，仅视觉
  launch.py        # 可复现的固定来物采样
  object_pool.py   # 备用物体隔离
  stl_assets.py    # STL 转换、尺度和凸包碰撞
  config_loader.py # 加载与校验场景配置
  checks.py        # 实际 PhysX 环境检查
  terminal_status.py # 复位前状态快照、终端输出与按 s 保存 TXT
assets/stl/        # 模型、清单与许可证
tests/             # 放架任务、来物采样与物体池测试
run.py             # 预览和仿真检查入口
preview.sh         # 本机桌面预览
```

本次清理后验证：16 项环境单元测试通过，三个场景配置校验通过，三个场景各完成 2 环境、180 步的 PhysX 检查。

PhysX 检查涵盖投放位置/速度/弹道、手部固定关节、物体质量、STL 尺寸与碰撞、标尺、局部复位、架面承重及放稳/失败规则。放架规则检查使用明确注入的接住历史和机器人站姿夹具，验证环境判据，不代表机器人已学会完成任务。
