# 下载的 STL 模型

这 8 个文件直接下载自 [mikedh/trimesh 的 models 目录](https://github.com/mikedh/trimesh/tree/fcf660feb0a14c68fd3945789e8ed77e260f9167/models)，保留原始文件内容，仅把扩展名统一为小写。仓库许可证原文保存在 `LICENSE.trimesh.md`；逐文件下载地址、固定提交、SHA-256、三角面数量和原始包围盒尺寸保存在 `manifest.json`。

| 文件 | 模型 | 默认最长边 | 默认质量 |
|---|---|---|---|
| teapot.stl | 茶壶 | 45 cm | 0.60 kg |
| torus.stl | 圆环 | 45 cm | 0.40 kg |
| angle_block.stl | 角块 | 40 cm | 0.60 kg |
| plate_holes.stl | 带孔板 | 50 cm | 0.40 kg |
| featuretype.stl | 特征块 | 45 cm | 0.60 kg |
| octagonal_pocket.stl | 八边形凹槽件 | 45 cm | 0.40 kg |
| 20mm-xyz-cube.stl | XYZ 标记方块 | 35 cm | 0.70 kg |
| cylinder.stl | 圆柱杆 | 55 cm | 0.50 kg |

尺寸、质量是训练用设置，不是扫描测量值。所有模型已加入 `env_configs/continuous_throw.py`；修改那里即可控制参与抛掷的模型。

显示使用原始三角网格，碰撞使用凸包，所以圆环的孔洞、茶壶手柄和凹槽不会保留为可穿过的碰撞空间。STL 不携带可靠的单位与颜色：导入时自动居中并按最长边缩放，颜色由配置提供。缓存写入 `assets/usd_cache/`，源 STL 内容或尺寸变化后自动更新。
