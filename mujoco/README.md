# MuJoCo G1 环境

环境代码位于 `mujoco/environment/`。复用仓库现有 `env_configs/`、`g1_throw/shelf_task.py` 和终端记录器，提供与当前 Isaac Lab 环境相同的场景参数、23 维动作、观测字段顺序、奖励与成功/失败规则。运行不需要启动或安装 Isaac Sim。

## 启动

在项目根目录运行：

```bash
# 本机已有 conda mujoco 环境，直接打开默认接箱放架窗口
bash mujoco/preview.sh

# 切换现有场景
bash mujoco/preview.sh --config env_configs/single_throw.py
bash mujoco/preview.sh --config env_configs/continuous_throw.py

# 无窗口真实物理检查
conda activate mujoco
python mujoco/environment/run.py --mode smoke --headless --num_envs 2 --steps 180

# MuJoCo 回归测试；不需要 Isaac Sim
python -m unittest discover -s mujoco/environment/tests -v
```

`preview.sh` 默认解释器为 `/home/zhe/anaconda3/envs/mujoco/bin/python`。在其他机器先安装 `pip install -r mujoco/requirements.txt`，再通过 `MUJOCO_PYTHON=/你的/python路径 bash mujoco/preview.sh` 指定解释器，或直接运行 `python mujoco/environment/run.py`。

可视化默认持续运行，关闭窗口或在终端按 Ctrl+C 退出；窗口空格暂停/继续。`--steps 180` 限制控制步数，`--steps 0` 持续运行，`--headless` 关闭可视化，`--seed` 设置投放随机种子。窗口显示环境 0，`--num_envs` 指定独立仿真环境数。

**与原 Isaac Lab 入口一样，这里是零动作预览，没有自动接住、站立平衡或放架控制策略。** 机器人可能跌倒并自动复位。物理测试中的接住历史和每步站姿是明确的测试夹具，不是运行中的隐形控制。

## 对齐的要求

| 项目 | MuJoCo 实现 |
| --- | --- |
| 机器人 | 与当前 Isaac Lab G1 同一套关节命名的旧版 G1 MJCF；23 个身体关节，14 个手指关节固定在零角度，手部黑色 |
| 动作顺序 | 从本机 Isaac Lab G1 实测取出的 23 维顺序；不直接使用 MJCF 内部的关节排列 |
| 执行器 | 默认姿态、位置增益、阻尼、力矩上限与当前 Isaac Lab 配置对应；关节软限位系数 0.9 |
| 接箱动作 | 默认姿态 + 1.5 × action（rad），输入裁剪至 [-1, 1]，目标变化率上限 6 rad/s |
| 仿真时间 | 物理步长 1/120 s，每个控制步 2 个物理步，即 60 Hz 控制 |
| 默认箱子 | 35 × 35 × 30 cm，0.70 kg，自由刚体 |
| 投放 | 相对发射时机器人基座，X 3–5 m，Y ±0.4 m；7.5–8 m/s；补偿重力的低弹道；目标左右偏移 ±0.65 m，高度 1.0 m |
| 架子 | 上层中心 (0.45, 0.65, 0.75) m，60 × 55 cm，板厚 4 cm；下层高 25 cm，四条固定架腿 |
| 地面 | 0.1 m 分米网格、数字、坐标轴和投放矩形；直接复用同一份标尺几何，不参与碰撞 |
| 接住 | 双臂持续接触 0.20 s，相对速度 <0.65 m/s，直立，接住前不能先碰架 |
| 放稳成功 | 完整箱体投影在架面内，余量 2.5 cm；底面高度、姿态达标；架面承重 ≥70%；机器人全部松开；低线/角速度持续 0.75 s |
| 失败 | 箱底离地 ≤2.5 cm、越出 7 m 范围、机器人跌倒或 12 s 超时；成功/物理失败优先于超时 |
| 观测和奖励 | 默认 152 维；单次抛物 88 维；连续 STL 93 维；复用同一个 `ShelfTask` 状态机与奖励 |
| 终止信息 | `info["shelf"]`、`info["terminal"]` 保留复位前快照；`info["log"]` 包含回合统计 |
| STL 场景 | 使用现有 8 个 STL，按最长边缩放、居中、指定质量，凸包碰撞，相邻抛掷不重复 |

所有场景参数继续在原有 `env_configs/*.py` 修改；两套仿真使用同一份配置。`robot_usd` 是 Isaac 专用选项，MuJoCo 会拒绝非空值。

## 终端按 s 保存 TXT

终端每 0.5 仿真秒打印箱子状态、机器人状态、接触部位/接触力、架面承重和是否掉落；接触变化和回合结束立即打印，并保留自动复位前的状态。

点击运行程序的终端，按 **`s`**（不用回车），把本次运行截至当前已输出的状态保存为 UTF-8 TXT。默认保存到项目的 `logs/mujoco_robot_status/robot_status_日期_时间_微秒.txt`，终端打印完整路径。再次按 `s` 生成新文件；用 `--status_dir /你的/目录` 改保存位置。

**不按 s 不创建状态文件或保存目录，关闭程序也不自动保存。** 保存后产生的新记录需要再次按 s 才会落盘。单次/连续抛物场景沿用原环境的检测能力，明确显示未启用的接触/掉落任务判据。`smoke` 模式不启动手动记录器。

## Python 接口

在项目根目录下，把 `mujoco/environment` 加入模块搜索路径：

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path("mujoco/environment").resolve()))
from g1_mujoco import G1MujocoEnv

env = G1MujocoEnv(num_envs=2, seed=42)
try:
    obs, info = env.reset()
    actions = torch.zeros((2, 23))
    obs, reward, terminated, truncated, info = env.step(actions)
    print(obs["policy"].shape)  # torch.Size([2, 152])
    env.reset(env_ids=[0])     # 只复位环境 0
finally:
    env.close()
```

`step` 也接受 NumPy 数组，返回 CPU torch 张量。终止后自动局部复位，返回的观测是复位后的观测，`info` 保留结束步的信息。`launch_plan` 支持 `g1_throw.launch.launch_dataset` 的固定投放样本。`model`、`datas` 可用于 MuJoCo 调试；默认可视化使用 `data = datas[0]`。

## 物理后端差异

任务要求和接口对齐不代表两个引擎每一帧的轨迹、接触力或成功率数值相同。MuJoCo 使用 `implicitfast`、MJCF 惯量和碰撞代理；PhysX 使用 USD 资产及另一套接触求解。静/动摩擦和恢复系数没有简单的一一映射，当前 MuJoCo 使用滑动摩擦 0.6、架面 0.8 和其原生软接触，未声称复现 PhysX 的 0.15 恢复系数。

批量环境共享模型结构、使用独立 `MjData`，在 CPU 上依次推进，彼此无物理交互；每个世界的坐标原点为零。Isaac Lab 的 12 m 间隔仅用于把多个场景摆在同一世界中，MuJoCo 无需这个物理间隔；配置校验仍保留原规则。这不是 GPU 并行训练后端，不能在多个线程同时调用同一个环境的 `step`。

模型为本地固定版本资产，运行时不下载。原始资产与许可证见 [模型来源](environment/assets/unitree_g1/SOURCE.md)。外层 `mujoco/` 故意不放 `__init__.py`，以免覆盖安装的 `mujoco` 库。

实现参考：[MuJoCo 接触力 API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce)、[Python passive viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。

## 目录

```text
mujoco/
  preview.sh                  # 本机可视化入口
  requirements.txt
  environment/
    run.py                    # preview / smoke 命令行入口
    g1_mujoco/
      model.py                # 固定手部、执行器、架子、物体、标尺 MJCF 构建
      environment.py          # MuJoCo 物理、动作/观测、真实接触和局部复位
      viewer.py               # 可视化生命周期与线程清理
      checks.py               # 真实物理检查
    assets/unitree_g1/        # 固定版本模型、网格、LICENSE 和来源
    tests/test_environment.py
```
