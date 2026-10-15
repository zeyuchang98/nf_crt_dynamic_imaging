# Intensity 残余运动与 motion 分支诊断

使用 complex block split 的两个既有 checkpoint 对应的保存预测。没有重新训练、改变网络、增加正则项或修改训练代码。

## 先看什么

建议先看嵌套 Hash NF：intensity 单独输出 GIF (historical reference; artifact not retained)、motion 输出曲线 (historical reference; artifact not retained)、motion 单独作用的图像 GIF (historical reference; artifact not retained)。

主要发现：intensity 输出中仍保留了与真实物体高度一致的平移轨迹，并出现明显膨胀；motion 输出幅度较小，整体方向对应也较差。motion 对最终拟合有帮助，但结果支持它在修补自身 intensity 输出的解释，尚未恢复 phantom 定义的真实运动。

## 1. Intensity 单独输出中还剩多少运动

固定同一套参考坐标，只查询 `C_pred(q,t)`，完全绕过 motion 网络。GIF 从左到右为：真实动态图像、intensity 单独输出、真实参考强度图。最后一栏形状固定，但保留真实亮度变化。

| 模型 | 最大平移距离，距首帧 (px) | x 尺寸范围，首帧=1 | y 尺寸范围，首帧=1 | x/y 平移轨迹与真实物体的相关系数 | 几何拟合中位 RRMSE |
|---|---:|---:|---:|---:|---:|
| nested_mlp | 8.74 | 1.00–1.38 | 1.00–1.60 | 0.9966 / 0.9993 | 14.56% |
| nested_hash_nf | 7.41 | 1.00–1.40 | 1.00–1.59 | 0.9993 / 0.9994 | 13.86% |

作为对照，真实物体的最大中心平移是 7.01 px，x/y 最大尺寸倍率为 1.30/1.45；真实参考强度图的平移应为 0、尺寸倍率应为 1。轨迹相关系数只比较时间趋势，不是“恢复了多少运动”的百分比。

如何避免把亮度变化算成运动：逐帧用 phantom 已知的椭圆及两个亮点形状拟合 intensity 输出，同时拟合六个量：x/y 平移、x/y 尺寸、主体亮度系数、亮点亮度系数。两个亮度系数彼此独立，因此可以解释本 toy 的整体和局部亮度变化。

这些是已知形状下的整体几何估计，不是稠密光流，也不是通用 PACT 运动估计器。预测图像不完全符合这个六参数模型，拟合剩余约 14% 的相对误差，因此平移与尺寸只作近似诊断。`geometry_fit.gif` 显示拟合本身的误差，以便检查该近似。

估计器先经过两个控制：对亮度随时间变化但几何固定的真实参考图，检测到的最大平移小于 0.000001 px；对真实动态图像，平移和尺寸恢复误差也低于 0.000001（分别为 px 和倍率）。这验证了它能在本 phantom 的亮度变化下恢复指定的几何量。

## 2. Motion network 实际输出了什么

直接读取网络输出的 backward displacement `ux,uy`，通过固定位置的时间曲线及前景上的幅度、误差检查输出，没有借助光流估计。

| 模型 | 预测位移 RMS (px) | 真实位移 RMS (px) | 位移 endpoint RMSE (px) | ux / uy RMSE (px) | 整体向量余弦 |
|---|---:|---:|---:|---:|---:|
| nested_mlp | 1.36 | 5.97 | 6.79 | 5.51 / 3.97 | -0.529 |
| nested_hash_nf | 1.15 | 5.97 | 6.48 | 5.08 / 4.01 | -0.358 |

上表所有量均在“测试像素 ∩ 真实物体前景”上计算。位移 RMS 是输出幅度，不是误差；真实位移 RMS 同时等于全零预测的 endpoint RMSE。整体向量余弦是将选中的所有二维位移拼接后算出的相似度，负数表示总体内积为负，不表示每一个位置都反向。

`motion_outputs.png` 上排直接显示固定空间点的 ux/uy 随时间变化；下排显示每帧完整真实前景上的幅度和误差。下排是全前景诊断，与上表的测试子集范围不同。固定点曲线也不是跟踪某个组织点的轨迹。

## 3. Motion 对最终图像究竟有没有贡献

将预测 motion 接到已知真实 intensity：`C_true(r + u_pred(r,t),t)`，并与真实动态图像对比。这隔离了学习到的 intensity 误差；真实参考强度的亮度变化仍被保留。

| 模型 | 完整模型 RRMSE | 关闭 motion，只用 C_pred(r,t) | 预测 motion + 真实 intensity | 真实 motion + 预测 intensity |
|---|---:|---:|---:|---:|
| nested_mlp | 16.72% | 30.17% | 83.06% | 99.44% |
| nested_hash_nf | 11.09% | 28.12% | 76.74% | 94.66% |

上表只统计测试像素，保留原始预测。以嵌套 Hash NF 为例，关闭 motion 后误差从 11.09% 增加到 28.12%，所以 motion 确实参与改善了拟合；但它与真实 intensity 配合时误差达到 76.74%。结合小幅且对应不准的位移、intensity 中保留的平移和过大的膨胀，现有证据更支持“两个分支互相补偿”，而非正确分离。

## 图和数组入口

### nested_mlp

- Intensity 残余运动 GIF (historical reference; artifact not retained) · 平移和尺寸曲线 (historical reference; artifact not retained)
- 几何估计的拟合检查 GIF (historical reference; artifact not retained)
- ux/uy、幅度、误差曲线 (historical reference; artifact not retained)
- 预测 motion + 真实 intensity GIF (historical reference; artifact not retained)
- [数值指标](nested_mlp/metrics.json) · 几何参数与拟合数组 (historical reference; artifact not retained)

### nested_hash_nf

- Intensity 残余运动 GIF (historical reference; artifact not retained) · 平移和尺寸曲线 (historical reference; artifact not retained)
- 几何估计的拟合检查 GIF (historical reference; artifact not retained)
- ux/uy、幅度、误差曲线 (historical reference; artifact not retained)
- 预测 motion + 真实 intensity GIF (historical reference; artifact not retained)
- [数值指标](nested_hash_nf/metrics.json) · 几何参数与拟合数组 (historical reference; artifact not retained)

`diagnostics.npz` 的 `intensity_geometry` 列顺序为 `[tx, ty, sx, sy, base_gain, spot_gain]`；tx/ty 是输入坐标单位，像素位移为坐标位移乘以 63。图中的平移扣除了首帧位置，尺寸除以首帧尺寸。亮度系数分别乘在主体加第二亮点、第一亮点上。

[summary.json](summary.json) 保存两组结果、估计器控制结果和源 checkpoint/预测数组的 SHA-256。分析前后这些源文件完全一致。所有 GIF 使用固定强度范围 0–1，计算指标不裁剪。
