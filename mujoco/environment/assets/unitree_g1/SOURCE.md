# G1 模型来源

- 仓库：[Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/cddb24f0dba18db8a330944edd0621adb91bdad0/unitree_g1)
- 固定提交：`cddb24f0dba18db8a330944edd0621adb91bdad0`（2024-07-19）。
- 本目录保留该提交的原始 `g1.xml`、`assets/` 和 BSD-3-Clause `LICENSE`。
- 选择旧版是为了匹配当前 Isaac Lab `G1_CFG` 的 `torso_joint`、`elbow_pitch_joint`、`elbow_roll_joint` 等 23 个身体关节。新版 Menagerie G1 已改为另一套 29 关节结构。
- 原始模型含 14 个手指关节。`g1_mujoco/model.py` 在内存中删除这些关节及驱动，保留零角度的网格、质量和惯量；将掌部接触网格分为固定子刚体，掌部惯量仍保留在原模型肘部中，避免重复质量。
- 原始第三方文件未修改。运行时另行配置黑色固定手部、Isaac Lab 对应的默认姿态/增益/力矩上限、地面、架子、投放物和标尺。
- MuJoCo 使用此 MJCF 的碰撞几何；它与 NVIDIA USD 模型的碰撞近似、PhysX 的接触求解并不逐项相同。
