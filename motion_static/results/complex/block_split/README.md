# Complex phantom：时空块留出实验

先看 划分示意 (historical reference; artifact not retained) 和 模型对比 (historical reference; artifact not retained)，再看下面各模型的逐帧指标与 GIF。

## 划分与训练

- 原始数据：64×64×64，按 `(y,x,t)` 切成 8×8×2 的块。共 2048 块，其中 410 块（20.02%）留作测试。
- 训练点 209,664，测试点 52,480；三个模型复用同一份 seed 0 划分。
- 同一空间块不连续留出时间块，首末时间块保留。最多连续缺 2 帧；两侧已知帧最多相隔 3 个采样间隔（0.047619 时间坐标单位）。
- 这是部分时空区域缺失的补全实验。每个时刻仍有训练像素，与旧实验的整帧 unseen midpoint 问题不同。
- 训练设置：Hash NF：seed 0、8,000 步、batch 4,096；Nested MLP：seed 0、8,000 步、batch 4,096；Nested Hash NF：seed 0、8,000 步、batch 4,096。仅图像 MSE；模型结构和学习率沿用各自配置。
- 真实 motion、参考强度图和测试像素强度均不参与训练；没有新增惩罚项。

## 图像结果

RRMSE 使用小数表示（0.1 = 10%），越低越好；SSIM 越高越好。所有 Test 列只统计测试像素。

| 模型 | Train RRMSE | Test RRMSE | Test SSIM | ROI RRMSE | ROI SSIM | Test PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Temporal interpolation | — | 0.154926 | 0.980698 | 0.154926 | 0.939255 | 37.364 |
| Hash NF | 0.000727 | 0.136073 | 0.989760 | 0.136044 | 0.968235 | 38.491 |
| Nested MLP | 0.141535 | 0.167206 | 0.978502 | 0.166919 | 0.937883 | 36.702 |
| Nested Hash NF | 0.000505 | 0.110916 | 0.992320 | 0.110872 | 0.977135 | 40.267 |

插值基线逐像素寻找前后最近的可见训练帧，按时间距离线性插值。连续缺两帧时使用 1/3、2/3 权重；不读取留出真值。

SSIM 使用完全落在评分区域内的 7×7 空间窗口，再对所有有效窗口中心取平均；ROI 窗口还必须完全落在 ROI 内。它与旧实验对整个时空数组计算的 SSIM 不同，不能直接横比。
RRMSE 和 PSNR 对所有选中像素一次性计算，不平均逐帧分数。首末保护帧没有测试像素，其测试曲线留空。ROI 使用完整真值的空间包围盒，仅用于评估。

逐帧测试区域随时间改变，所以曲线起伏同时受到区域组成影响，不能单独解释为模型不平滑。

## 嵌套模型的分工检查

Motion 在测试像素与真实物体前景的交集上评分；参考强度图在完整参考网格上单独诊断。

| 模型 | 前景 motion RMSE (px) | 零 motion 基线 (px) | 参考强度图 RRMSE | 非正 Jacobian 比例 |
|---|---:|---:|---:|---:|
| Nested MLP | 6.7920 | 5.9736 | 1.012234 | 0.0000% |
| Nested Hash NF | 6.4753 | 5.9736 | 0.981962 | 0.0404% |

参考强度图的数值是分解诊断，不是 observation train/test 分数。最终图像准确也不能证明 motion 与 intensity 恢复了真实分工；请同时查看替换分量的结果。

## 图和原始结果

### Hash NF

- 图像 GIF (historical reference; artifact not retained) · 最差测试帧 (historical reference; artifact not retained)
- 逐帧 RRMSE / SSIM / PSNR (historical reference; artifact not retained) · 时间曲线 (historical reference; artifact not retained)
- [完整指标](hash_nf/summary.json) · [配置](hash_nf/config.json) · 划分数组 (historical reference; artifact not retained)

### Nested MLP

- 图像 GIF (historical reference; artifact not retained) · 最差测试帧 (historical reference; artifact not retained)
- 逐帧 RRMSE / SSIM / PSNR (historical reference; artifact not retained) · 时间曲线 (historical reference; artifact not retained)
- [完整指标](nested_mlp/summary.json) · [配置](nested_mlp/config.json) · 划分数组 (historical reference; artifact not retained)
- Motion 误差 (historical reference; artifact not retained)
- ux 指标 (historical reference; artifact not retained) · uy 指标 (historical reference; artifact not retained)
- 参考强度图 GIF (historical reference; artifact not retained) · 参考强度图指标 (historical reference; artifact not retained) · [替换分量检查](nested_mlp/metrics/part_checks.json)

### Nested Hash NF

- 图像 GIF (historical reference; artifact not retained) · 最差测试帧 (historical reference; artifact not retained)
- 逐帧 RRMSE / SSIM / PSNR (historical reference; artifact not retained) · 时间曲线 (historical reference; artifact not retained)
- [完整指标](nested_hash_nf/summary.json) · [配置](nested_hash_nf/config.json) · 划分数组 (historical reference; artifact not retained)
- Motion 误差 (historical reference; artifact not retained)
- ux 指标 (historical reference; artifact not retained) · uy 指标 (historical reference; artifact not retained)
- 参考强度图 GIF (historical reference; artifact not retained) · 参考强度图指标 (historical reference; artifact not retained) · [替换分量检查](nested_hash_nf/metrics/part_checks.json)

图像 GIF 的误差栏中，灰色像素属于训练集；误差亮度只显示测试像素。

固定点时间曲线显示全部时刻，但仅带标记的测试样本参与曲线评分；ROI 测试均值的像素集合随时间变化，不应当作固定组织的生理曲线。

每个模型目录保存 `config.json`、`model.pt`、`loss.csv`、`split.npz`、`summary.json`、`arrays/`、`metrics/`、`plots/`；原始预测不做裁剪。旧 midpoint 实验保留在上一层对应模型目录。

## 复现

在仓库根目录运行；已有输出目录不会被覆盖，重跑时加 `--out 新目录`。

```bash
python -m motion_static.run --phantom complex --model hash_nf --split blocks
python -m motion_static.run --phantom complex --model nested_mlp --split blocks
python -m motion_static.run --phantom complex --model nested_hash_nf --split blocks
python -m motion_static.evaluation.summarize_blocks
```

划分配置在 `motion_static/experiments/splits/blocks.json`，划分函数在 `motion_static/splits.py`。`--seed` 只改变模型与训练采样；划分 seed 独立固定在划分配置中。

这是一组固定 seed 的表示实验；模型参数量不同，不是等参数量比较。

## 两个分支实际在做什么

[Intensity 残余运动与 motion 分支诊断](branch_diagnostics/README.md)：现有模型的独立诊断，包含 intensity 残余平移/膨胀、motion 输出曲线和分支替换实验。
