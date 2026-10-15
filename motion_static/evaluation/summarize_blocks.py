"""Collect completed block experiments into one table, plot, and reading guide."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .metrics import save_metrics
from .plots import pyplot


MODELS = {"hash_nf": "Hash NF", "nested_mlp": "Nested MLP", "nested_hash_nf": "Nested Hash NF"}


def summarize(root):
    reports = {name: json.loads((root / name / "summary.json").read_text())
               for name in MODELS if (root / name / "summary.json").exists()}
    if not reports:
        raise ValueError(f"No completed block experiments in {root}")
    first = next(iter(reports.values()))
    split = first["split"]
    phantom = first["config"]["phantom"]
    config = first["config"]
    block_text = "×".join(str(n) for n in split["block_shape_yxt"])
    with np.load(root / next(iter(reports)) / "split.npz") as arrays:
        time_step = float(arrays["t"][1] - arrays["t"][0])
    for report in reports.values():
        if report["split"] != split:
            raise ValueError("Models must use exactly the same split")
        if report["image"]["temporal_interpolation"] != first["image"]["temporal_interpolation"]:
            raise ValueError("Interpolation baselines differ between models")
    rows = [dict(model="temporal_interpolation", label="Temporal interpolation", train_rrmse=None,
                 test=first["image"]["temporal_interpolation"])]
    for name, report in reports.items():
        rows.append(dict(model=name, label=MODELS[name], parameters=report["parameters"],
                         train_rrmse=report["image"]["train_pixels"]["rrmse"],
                         test=report["image"]["test_pixels"]))
    save_metrics(root / "comparison.json", dict(split=split, models=rows))

    plt = pyplot(root)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    colors = ["#888888", "#0072b2", "#d55e00", "#009e73"][:len(rows)]
    for row_index, scope in enumerate((None, "roi")):
        for col, (key, title) in enumerate((("rrmse", "RRMSE (lower is better)"), ("ssim", "Spatial SSIM (higher is better)"))):
            scores = [row["test"] if scope is None else row["test"][scope] for row in rows]
            values = [score[key] for score in scores]
            ax = axes[row_index, col]
            bars = ax.bar(np.arange(len(rows)), values, color=colors)
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=9)
            ax.set_xticks(np.arange(len(rows)), [row["label"].replace(" ", "\n", 1) for row in rows])
            ax.set(title=f"{'Full field of view' if scope is None else 'Spatial ROI'}: {title}", ylim=(0, max(values) * 1.2))
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    fig.suptitle(f"{phantom.capitalize()} phantom: held-out blocks, same mask for all models\n"
                 f"{block_text} blocks; interpolation reads training values only", fontsize=14)
    fig.savefig(root / "comparison.png", dpi=170)
    plt.close(fig)
    first_name = next(iter(reports))
    shutil.copyfile(root / first_name / "plots" / "split.png", root / "split.png")

    lines = [f"# {phantom.capitalize()} phantom：时空块留出实验", "",
             "先看 [划分示意](split.png) 和 [模型对比](comparison.png)，再看下面各模型的逐帧指标与 GIF。", "",
             "## 划分与训练", "",
             f"- 原始数据：{config['size']}×{config['size']}×{config['frames']}，按 `(y,x,t)` 切成 {block_text} 的块。共 {split['total_blocks']} 块，"
             f"其中 {split['test_blocks']} 块（{split['actual_test_fraction']:.2%}）留作测试。",
             f"- 训练点 {split['train_points']:,}，测试点 {split['test_points']:,}；三个模型复用同一份 seed {split['seed']} 划分。",
             f"- 同一空间块不连续留出时间块，首末时间块保留。最多连续缺 {split['max_missing_frames']} 帧；"
             f"两侧已知帧最多相隔 {split['max_bracketing_gap_in_frame_steps']} 个采样间隔"
             f"（{split['max_bracketing_gap_in_frame_steps'] * time_step:.6f} 时间坐标单位）。",
             "- 这是部分时空区域缺失的补全实验。每个时刻仍有训练像素，与旧实验的整帧 unseen midpoint 问题不同。",
             "- 训练设置：" + "；".join(
                 f"{MODELS[name]}：seed {r['config']['seed']}、{r['config']['steps']:,} 步、batch {r['config']['batch_size']:,}"
                 for name, r in reports.items()) + "。仅图像 MSE；模型结构和学习率沿用各自配置。",
             "- 真实 motion、参考强度图和测试像素强度均不参与训练；没有新增惩罚项。", "",
             "## 图像结果", "",
             "RRMSE 使用小数表示（0.1 = 10%），越低越好；SSIM 越高越好。所有 Test 列只统计测试像素。", "",
             "| 模型 | Train RRMSE | Test RRMSE | Test SSIM | ROI RRMSE | ROI SSIM | Test PSNR (dB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        score = row["test"]
        training = "—" if row["train_rrmse"] is None else f"{row['train_rrmse']:.6f}"
        lines.append(f"| {row['label']} | {training} | {score['rrmse']:.6f} | {score['ssim']:.6f} | "
                     f"{score['roi']['rrmse']:.6f} | {score['roi']['ssim']:.6f} | {score['psnr_db']:.3f} |")
    lines += ["", "插值基线逐像素寻找前后最近的可见训练帧，按时间距离线性插值。"
              "连续缺两帧时使用 1/3、2/3 权重；不读取留出真值。", "",
              "SSIM 使用完全落在评分区域内的 7×7 空间窗口，再对所有有效窗口中心取平均；"
              "ROI 窗口还必须完全落在 ROI 内。它与旧实验对整个时空数组计算的 SSIM 不同，不能直接横比。",
              "RRMSE 和 PSNR 对所有选中像素一次性计算，不平均逐帧分数。"
              "首末保护帧没有测试像素，其测试曲线留空。ROI 使用完整真值的空间包围盒，仅用于评估。", "",
              "逐帧测试区域随时间改变，所以曲线起伏同时受到区域组成影响，不能单独解释为模型不平滑。", "",
              "## 嵌套模型的分工检查", "",
              "Motion 在测试像素与真实物体前景的交集上评分；参考强度图在完整参考网格上单独诊断。", "",
              "| 模型 | 前景 motion RMSE (px) | 零 motion 基线 (px) | 参考强度图 RRMSE | 非正 Jacobian 比例 |",
              "|---|---:|---:|---:|---:|"]
    for name, report in reports.items():
        if "motion" not in report:
            continue
        motion = report["motion"]["foreground"]
        reference = report["reference_intensity_full_grid"]
        lines.append(f"| {MODELS[name]} | {motion['endpoint_rmse_pixels']:.4f} | "
                     f"{motion['zero_motion_endpoint_rmse_pixels']:.4f} | {reference['rrmse']:.6f} | "
                     f"{report['motion']['foreground_nonpositive_jacobian_fraction']:.4%} |")
    lines += ["", "参考强度图的数值是分解诊断，不是 observation train/test 分数。"
              "最终图像准确也不能证明 motion 与 intensity 恢复了真实分工；请同时查看替换分量的结果。", "",
              "## 图和原始结果", ""]
    for name in reports:
        lines += [f"### {MODELS[name]}", "",
                  f"- [图像 GIF]({name}/plots/image.gif) · [最差测试帧]({name}/plots/image.png)",
                  f"- [逐帧 RRMSE / SSIM / PSNR]({name}/plots/image/framewise_metrics.png) · "
                  f"[时间曲线]({name}/plots/image/temporal_curves.png)",
                  f"- [完整指标]({name}/summary.json) · [配置]({name}/config.json) · [划分数组]({name}/split.npz)"]
        if "motion" in reports[name]:
            lines += [f"- [Motion 误差]({name}/plots/motion_error.png)",
                      f"- [ux 指标]({name}/plots/ux/framewise_metrics.png) · [uy 指标]({name}/plots/uy/framewise_metrics.png)",
                      f"- [参考强度图 GIF]({name}/plots/reference_intensity.gif) · "
                      f"[参考强度图指标]({name}/plots/reference_intensity/framewise_metrics.png) · "
                      f"[替换分量检查]({name}/metrics/part_checks.json)"]
        lines.append("")
    lines += ["图像 GIF 的误差栏中，灰色像素属于训练集；误差亮度只显示测试像素。", "",
              "固定点时间曲线显示全部时刻，但仅带标记的测试样本参与曲线评分；"
              "ROI 测试均值的像素集合随时间变化，不应当作固定组织的生理曲线。", "",
              "每个模型目录保存 `config.json`、`model.pt`、`loss.csv`、`split.npz`、`summary.json`、"
              "`arrays/`、`metrics/`、`plots/`；原始预测不做裁剪。旧 midpoint 实验保留在上一层对应模型目录。", "",
              "## 复现", "", "在仓库根目录运行；已有输出目录不会被覆盖，重跑时加 `--out 新目录`。", "", "```bash"]
    for name in reports:
        lines.append(f"python -m motion_static.run --phantom {phantom} --model {name} --split blocks")
    lines += ["python -m motion_static.evaluation.summarize_blocks", "```", "",
              "划分配置在 `motion_static/experiments/splits/blocks.json`，划分函数在 `motion_static/splits.py`。"
              "`--seed` 只改变模型与训练采样；划分 seed 独立固定在划分配置中。", "",
              "这是一组固定 seed 的表示实验；模型参数量不同，不是等参数量比较。"]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(root / "README.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/complex/block_split")
    summarize(parser.parse_args().results)


if __name__ == "__main__":
    main()
