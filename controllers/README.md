# 分层具身接物控制

`single_throw.py` 默认选择 `controller="hierarchical"`。运行：

```bash
/home/zhe/anaconda3/envs/env_isaaclab/bin/python run.py --config env_configs/single_throw.py --num_envs 1
```

在其他场景启用：`--controller hierarchical`；恢复原来的站立关节控制：`--controller joint`。控制实现包含中文注释。

| 文件 | 职责 |
| --- | --- |
| `config.py` | 预测时域、缓冲时间、接触阈值、限速和残差幅度 |
| `planner.py` | 根据弹道、可达范围、接触及平衡状态拆解任务 |
| `whole_body.py` | 双臂位置逆运动学、屈膝缓冲、踝髋姿态反馈、关节限位限速 |
| `hierarchical.py` | 读取仿真状态和双掌接触力，连接上下层 |

高层顺序为 `READY → INTERCEPT → ABSORB → HOLD`，错过物体、失去接触或倾倒时进入 `RECOVER`。每个并行环境独立维护阶段，新抛掷和局部重置都会清除对应任务状态。预测在基座偏航坐标系中进行，并补偿世界重力。

低层让双臂跟踪同一个预测接物点；接触后顺来球方向后撤，双腿同步屈膝，踝髋根据基座倾斜、速度和质心捕获点相对双脚的位置反馈调整。模型复位采用抬臂预备姿态。关节目标经过限速和软限位，再由原有 PD 执行器产生力矩；执行器保留官方力矩上限。此实现是站定抱接基线，没有跨步控制、质心动力学优化或全身 QP。

双手的掌部及七个指节分别设置传感器，过滤当前物体的接触力，再按左右手汇总。双侧接触、低相对速度、物体位于胸前且身体直立持续 `hold_seconds`，才会锁存本次 `success`。仅距离接近或碰到躯干不计成功。不会瞬移物体、固定物体到手上或创建吸附约束。

零动作预览会执行完整分层控制。训练/回放时，23 维动作作为关节残差叠加到分层目标上；抱稳过程增加奖励。原来 `joint` 模式的模型动作含义不同，即使维度相同也不应直接用于分层模式。现有手指为固定关节，目标是双臂抱接大物体，不能执行手指闭合抓取。

修改控制参数示例：

```python
from controllers.config import CatchControlCfg

# 在 Settings(...) 中设置
# controller="hierarchical",
# catch_control=CatchControlCfg(intercept_x=0.36, hold_seconds=0.4),
```

测试：

```bash
/home/zhe/anaconda3/envs/env_isaaclab/bin/python -m unittest discover -s tests -v
/home/zhe/anaconda3/envs/env_isaaclab/bin/python run.py --mode smoke --config env_configs/single_throw.py --num_envs 2 --steps 240 --headless
```

`SMOKE PASSED` 只表示接口与数值检查通过。接物统计单独输出成功抱稳次数，不将程序运行成功等同于已掌握高动态抓取。实际成功率还取决于物体形状、手型、控制参数和策略训练。

## 本机验证（2026-09-10）

Isaac Lab 2.3.2 / Isaac Sim 5.1 / RTX 5060 Ti，使用单次场景原有 4–5 m/s 球、箱体、圆柱，零 PPO 残差：

| 随机种子 | 并行环境 | 控制步数 | 抛掷次数 | 满足抱稳判据 |
| --- | --- | --- | --- | --- |
| 42 | 2 | 480 | 8 | 7 |
| 7 | 8 | 480 | 32 | 19 |

两轮合计 26/40 次达到 0.4 s 连续双手接触、低相对速度且直立的判据，五个任务阶段均出现。该样本包含跌倒后的自动复位，成功后仍可能掉物或跌倒，因此 **65% 仅为这些短时测试中的判据达成比例，不代表完整回合抱持率或泛化成功率**。

六项纯张量单元测试通过，覆盖弹道、远离物体、无接触不报成功、接触中断、局部复位、坐标变换与奇异 IK。仿真检查通过物体投放、关节软限位、有限观测/奖励、控制状态局部复位。`joint` 模式的两环境 120 步回归检查也通过。原始日志保存在 `logs/hierarchical_validation/`。
