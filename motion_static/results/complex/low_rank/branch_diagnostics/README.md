# Midpoint 分支诊断

使用已保存的未见中间时刻预测，分别检查 intensity 残余运动与 motion 输出。

| 实验 | 帧数 | Intensity 最大平移 (px) | 最大 x/y 尺寸倍率 | Motion RMS (px) | 真实 motion RMS (px) | Motion RMSE (px) |
|---|---:|---:|---:|---:|---:|---:|
| rank_2 | 63 | 0.23 | 1.16 / 1.17 | 4.84 | 6.07 | 4.78 |
| rank_3 | 63 | 0.38 | 1.16 / 1.18 | 4.84 | 6.07 | 4.82 |

Intensity 的平移和尺寸相对于首个 midpoint，是允许亮度独立变化的已知形状拟合估计。它们与 motion 的 RMS 不是同一统计量，不能相除作为运动分离百分比。

## 分支替换检查

所有图像 RRMSE 在完整 midpoint 图像上计算，使用原始预测。

| 实验 | 完整模型 | 关闭 motion | 预测 motion + 真实 intensity | 真实 motion + 预测 intensity |
|---|---:|---:|---:|---:|
| rank_2 | 5.16% | 81.61% | 26.20% | 47.79% |
| rank_3 | 5.31% | 81.28% | 30.93% | 55.12% |

## 图和诊断质量

### rank_2

- Intensity 单独输出 GIF (historical reference; artifact not retained) · 残余平移、膨胀曲线 (historical reference; artifact not retained)
- 几何拟合检查 GIF (historical reference; artifact not retained)：中位拟合 RRMSE 30.31%，用于检查几何估计是否可靠。
- ux/uy、幅度与误差曲线 (historical reference; artifact not retained)
- 预测 motion + 真实 intensity GIF (historical reference; artifact not retained)
- [完整指标及源文件校验](rank_2/metrics.json) · 几何拟合数组 (historical reference; artifact not retained)

控制结果：静止参考图最大伪平移 2e-08 px；真实动态图的最大平移估计误差 1.7e-08 px。

### rank_3

- Intensity 单独输出 GIF (historical reference; artifact not retained) · 残余平移、膨胀曲线 (historical reference; artifact not retained)
- 几何拟合检查 GIF (historical reference; artifact not retained)：中位拟合 RRMSE 33.32%，用于检查几何估计是否可靠。
- ux/uy、幅度与误差曲线 (historical reference; artifact not retained)
- 预测 motion + 真实 intensity GIF (historical reference; artifact not retained)
- [完整指标及源文件校验](rank_3/metrics.json) · 几何拟合数组 (historical reference; artifact not retained)

控制结果：静止参考图最大伪平移 2e-08 px；真实动态图的最大平移估计误差 1.7e-08 px。

## 指标含义

- 几何拟合依赖 toy 的已知形状，并非通用光流估计器。Complex 允许主体与亮点有两个独立亮度系数；simple 使用一个系数。
- Motion 指标在全部未见帧的真实前景上计算；RMS 表示幅度，RMSE 表示误差。整体向量余弦为负不代表每个像素都反向。
- 固定位置的 ux/uy 曲线不是组织点轨迹。图像显示范围固定 0–1，指标不裁剪。
- diagnostics.npz 中 intensity_geometry 的列为 `[tx,ty,sx,sy,brightness gains...]`。tx/ty 是坐标单位，像素单位由保存的 x/y 间隔换算。
- 所有诊断来自现有预测，不更新模型、不增加正则项。

运行入口：`python -m motion_static.evaluation.diagnose_branches RUN_DIR [RUN_DIR ...] --out OUTPUT_DIR`。实现与复用说明见 `motion_static/evaluation/README.md`。
