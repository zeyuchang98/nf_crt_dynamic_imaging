"""Build a small results index from completed experiments; no retraining."""
import argparse
import json
from pathlib import Path


def experiment_folders(results, phantom):
    """Original baselines plus completed low-rank runs in their own folders."""
    folders = [results / phantom / name for name in ("hash_nf", "nested_mlp", "nested_hash_nf")]
    folders += sorted((results / phantom / "low_rank").glob("rank_*"))
    return [folder for folder in folders if (folder / "summary.json").is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    rows = []
    lines = ["# 实验结果总览", "", "所有数字来自未参与训练的中间时刻；具体网格和 seed 见各自 config.json。RRMSE 和 motion 误差越低越好。", "",
             "| Phantom | Model | Image RRMSE | Image SSIM | ROI RRMSE | Motion RMSE（前景，像素） | 参考强度图 RRMSE |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for phantom in ("simple", "complex"):
        for folder in experiment_folders(args.results, phantom):
            report = json.loads((folder / "summary.json").read_text())
            model = report["config"]["model"]
            label = model + (f" (R={report['config']['rank']})" if "rank" in report["config"] else "")
            image = report["image"]["unseen_time_midpoints"]
            row = dict(phantom=phantom, model=model, image_rrmse=image["rrmse"], image_ssim=image["ssim"],
                       roi_rrmse=image["roi"]["rrmse"], parameters=report["parameters"], seed=report["config"]["seed"])
            motion_value, reference_intensity_value = "—", "—"
            if "rank" in report["config"]:
                row["rank"] = report["config"]["rank"]
            if "motion" in report:
                row["motion_endpoint_rmse_pixels"] = report["motion"]["midpoint"]["foreground"]["endpoint_rmse_pixels"]
                row["reference_intensity_rrmse"] = report["reference_intensity"]["unseen_time_midpoints"]["rrmse"]
                motion_value = f"{row['motion_endpoint_rmse_pixels']:.3f}"
                reference_intensity_value = f"{row['reference_intensity_rrmse']:.2%}"
            rows.append(row)
            lines.append(f"| {phantom} | {label} | {image['rrmse']:.2%} | {image['ssim']:.6f} | "
                         f"{image['roi']['rrmse']:.2%} | {motion_value} | {reference_intensity_value} |")
    lines += ["", "直接 Hash NF（hash_nf）没有独立的 motion 或参考强度图，因此对应项留空。",
              "图像拟合与分解准确性必须分开判断；单次 seed 结果不代表所有初始化。", "", "## 图在哪里", ""]
    for phantom in ("simple", "complex"):
        lines.append(f"- [{phantom} phantom]({phantom}/phantom/phantom.gif)")
        for folder in experiment_folders(args.results, phantom):
            base = folder.relative_to(args.results).as_posix()
            lines.append(f"- {base}：[图像 GIF]({base}/plots/image.gif)、[完整图像指标]({base}/metrics/image.json)")
            if (folder / "metrics/motion.json").is_file():
                lines.append(f"- {base} 分工检查：[参考强度图 GIF]({base}/plots/reference_intensity.gif)、"
                             f"[Motion 误差曲线]({base}/plots/motion_error.png)、[替换分量指标]({base}/metrics/part_checks.json)")
            if (folder / "plots/motion_only.gif").is_file():
                lines.append(f"- {base}：[Motion 单独作用 GIF]({base}/plots/motion_only.gif)")
    for phantom in ("simple", "complex"):
        branch_results = Path(phantom) / "midpoint_branch_diagnostics" / "README.md"
        if (args.results / branch_results).exists():
            lines += ["", f"## {phantom}：Midpoint 分支诊断", "",
                      f"[Intensity 残余运动、motion 输出和分支替换检查]({branch_results.as_posix()})。"]
        block_results = Path(phantom) / "block_split" / "README.md"
        if (args.results / block_results).exists():
            lines += ["", f"## {phantom}：新增时空块留出", "",
                      f"[块划分、三模型指标与 GIF]({block_results.as_posix()})。"
                      "每帧留出部分区域，和上面的整帧中间时刻实验分别报告。"]
        temporal_results = Path(phantom) / "temporal_l2" / "README.md"
        if (args.results / temporal_results).exists():
            lines += ["", f"## {phantom}：Intensity 时间差分 L2", "",
                      f"[与原始 loss 的图像、分工和亮度对比]({temporal_results.as_posix()})。"]
        low_rank_diagnostics = Path(phantom) / "low_rank" / "branch_diagnostics" / "README.md"
        if (args.results / low_rank_diagnostics).exists():
            lines += ["", f"## {phantom}：Low-rank intensity", "",
                      f"[R=2、3 的分支诊断与 GIF]({low_rank_diagnostics.as_posix()})。"
                      "模型面向 PACT 的连续对象表示；当前仍是图像监督 toy 验证，尚未验证真实 PACT 重建。"]
        multicycle = Path(phantom) / "low_rank_multicycle" / "README.md"
        if (args.results / multicycle).exists():
            lines += ["", f"## {phantom}：新多周期 prototype 与训练预算", "",
                      f"[3 周期、379 帧，8k / 32k steps 的 loss curve、分支指标与 GIF]({multicycle.as_posix()})。"
                      "与旧单周期模型的比较另用共同未见时刻，不混入上表。"]
    (args.results / "README.md").write_text("\n".join(lines) + "\n")
    (args.results / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
