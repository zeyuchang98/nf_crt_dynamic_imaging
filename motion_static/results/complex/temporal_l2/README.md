# Complex：Intensity 时间差分 L2

与已有 nested Hash NF 的原始 MSE 实验比较。模型、数据、seed、训练步数相同，只增加 intensity 时间惩罚。

`loss = image_mse + weight * mean(((C(q,t+delta)-C(q,t))/delta)^2)`

- 权重：0.0001；时间间隔：0.015873016（输入时间单位）。
- q 由训练坐标经过当前 motion 得到；两次查询保持同一个 q，正则项不向 motion 反传梯度。
- 仅用训练图像监督，无真实 motion、参考强度图或 midpoint 标签。使用一个预先选定的权重，没有按测试结果搜索参数。
- 该项也会惩罚真实亮度变化，因此同时检查几何、motion 与亮度。单个 phantom、单个 seed 不证明泛化。

| 指标 | 原始 MSE | 加时间 L2 |
|---|---:|---:|
| 图像 RRMSE | 3.40% | 5.79% |
| 图像 SSIM | 0.999436 | 0.998183 |
| ROI RRMSE | 3.40% | 5.78% |
| 参考强度图 RRMSE | 98.99% | 85.72% |
| Motion 前景 endpoint RMSE (px) | 6.421 | 5.925 |
| 前景非正形变 Jacobian 比例 | 0.15% | 9.94% |
| Intensity 最大残余平移 (px) | 7.511 | 6.061 |
| Intensity 几何拟合中位 RRMSE | 11.45% | 22.88% |
| 主体亮度系数 RRMSE（拟合估计） | 1.63% | 7.81% |
| 亮点亮度系数 RRMSE（拟合估计） | 12.81% | 28.34% |
| 主体亮度时间标准差 / 真值 | 0.965 | 0.762 |
| 亮点亮度时间标准差 / 真值 | 0.795 | 0.580 |

亮度由允许平移、膨胀和两个独立亮度系数的已知形状拟合估计，仅用于评估。时间标准差比值为 1 表示波动幅度相当，不代表时间曲线完全正确；需结合 RRMSE 和曲线。

分工与亮度对比 (historical artifact removed)

## 结果入口

- **Original MSE**：完整图像指标 (historical reference; artifact not retained) · 图像 GIF (historical reference; artifact not retained)
  Intensity 单独输出 (historical reference; artifact not retained) · 残余几何 (historical reference; artifact not retained) · Motion 输出曲线 (historical reference; artifact not retained) · Motion + 真实 intensity (historical reference; artifact not retained)
- **Temporal L2**：完整图像指标 (historical reference; artifact not retained) · 图像 GIF (historical reference; artifact not retained)
  Intensity 单独输出 (historical reference; artifact not retained) · 残余几何 (historical reference; artifact not retained) · Motion 输出曲线 (historical reference; artifact not retained) · Motion + 真实 intensity (historical reference; artifact not retained)

[完整对比数字](comparison.json)。原有逐帧指标、ROI、时间曲线和 average adjacent truth frames 对照保留在各实验目录。
