"""Diagnose saved midpoint predictions; no retraining or checkpoint changes.

Run: python -m motion_static.evaluation.diagnose_branches RUN_DIR [RUN_DIR ...] --out OUTPUT_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .branch_metrics import (calibrate_geometry, fit_intensity_geometry, geometry_statistics,
                             motion_statistics, toy_geometry_template, true_geometry)
from .branch_plots import save_branch_plots
from .metrics import error_metrics, save_metrics
from ..phantoms.data import sample


def diagnose_run(run, out):
    """Read one completed experiment and save its independent branch diagnostics."""
    run, out = Path(run).resolve(), Path(out).resolve()
    if run == out:
        raise ValueError("Use a separate diagnostic output folder")
    prediction_path = run / "predictions.npz"
    if not prediction_path.exists():
        prediction_path = run / "arrays/predictions.npz"
    sources = (run / "config.json", prediction_path if prediction_path.exists() else run / "model.pt")
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    config, data, prediction = load_midpoints(run)
    out.mkdir(parents=True, exist_ok=True)
    phantom = config["phantom"]
    spacing = np.array([data[axis][1] - data[axis][0] for axis in ("x", "y")])

    print(f"{run.name}: checking known-geometry controls", flush=True)
    template = toy_geometry_template(phantom, data["x"], data["y"])
    calibration, stationary, moving = calibrate_geometry(data, phantom, template)
    print(f"{run.name}: fitting residual intensity geometry", flush=True)
    fit = fit_intensity_geometry(prediction["reference_intensity"], data["x"], data["y"], *template)
    translation, _ = true_geometry(phantom, data["t"])
    intensity = geometry_statistics(fit, spacing, translation)
    motion = motion_statistics(data["u_true"], prediction["motion"], data["foreground_mask"], spacing)
    variants = {label: error_metrics(data["truth"], prediction[key], data_range=1) for label, key in (
        ("full_model", "image"), ("intensity_only_zero_motion", "reference_intensity"),
        ("predicted_motion_true_intensity", "learned_motion_true_intensity"),
        ("true_motion_predicted_intensity", "true_motion_learned_intensity"))}
    report = dict(source=str(run), phantom=phantom, model=config["model"], frame_count=len(data["t"]),
                  evaluation_scope="Complete unseen midpoint frames; motion metrics use true foreground.",
                  geometry_time_reference=dict(kind="first_unseen_midpoint", time=float(data["t"][0])),
                  intensity=intensity, motion=motion, image_variants_unseen_midpoints=variants,
                  calibration=calibration, source_sha256=hashes)
    save_metrics(out / "metrics.json", report)
    np.savez_compressed(out / "diagnostics.npz", t=data["t"], x=data["x"], y=data["y"],
                        intensity_geometry=fit["parameters"], intensity_template_fit=fit["images"],
                        intensity_template_fit_rrmse=fit["rrmse"],
                        stationary_control_geometry=stationary["parameters"], moving_control_geometry=moving["parameters"])
    save_branch_plots(out, data, prediction, fit, stationary, motion, phantom, run.name)
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise RuntimeError(f"Source changed during diagnosis: {path}")
    return report


def load_midpoints(run):
    """Read times and predictions from the experiment, not from fixed defaults."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    if config.get("evaluation_split", "midpoints") != "midpoints":
        raise ValueError("This diagnosis requires midpoint evaluation")
    archive = run / "predictions.npz"
    if not archive.exists():
        archive = run / "arrays/predictions.npz"
    if not archive.exists():
        from .render import load_run, saved_evaluation_data
        from .evaluate import predict_fields
        model, config = load_run(run)
        data, _ = saved_evaluation_data(run, config)
        if not hasattr(model, "motion"):
            raise ValueError("Branch diagnosis requires a nested model")
        return config, data, predict_fields(model, data, config)
    with np.load(archive, allow_pickle=False) as arrays:
        if "midpoint_motion" not in arrays:
            raise ValueError(f"{run}: requires a nested model with saved midpoint predictions")
        times = arrays["midpoint_t"]
        prediction = {key.removeprefix("midpoint_"): arrays[key] for key in arrays.files
                      if key.startswith("midpoint_") and key != "midpoint_t"}
    bounds = config.get("coordinate_bounds", [[-.5, .5]] * 3)[2]
    data = sample(config["phantom"], config["size"], times, bounds)
    return config, data, prediction


def write_summary(out, reports):
    """Regenerate a compact index alongside the figures; numbers come from reports."""
    save_metrics(out / "summary.json", dict(models=reports))
    lines = ["# Midpoint 分支诊断", "",
             "使用已保存的未见中间时刻预测，分别检查 intensity 残余运动与 motion 输出。", "",
             "| 实验 | 帧数 | Intensity 最大平移 (px) | 最大 x/y 尺寸倍率 | Motion RMS (px) | 真实 motion RMS (px) | Motion RMSE (px) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, report in reports.items():
        intensity, motion = report["intensity"], report["motion"]
        lines.append(f"| {name} | {report['frame_count']} | {intensity['max_translation_from_first_midpoint_pixels']:.2f} | "
                     f"{intensity['x_scale_range'][1]:.2f} / {intensity['y_scale_range'][1]:.2f} | "
                     f"{motion['predicted_displacement_rms_pixels']:.2f} | {motion['true_displacement_rms_pixels']:.2f} | "
                     f"{motion['midpoint_foreground_endpoint_rmse_pixels']:.2f} |")
    lines += ["", "Intensity 的平移和尺寸相对于首个 midpoint，是允许亮度独立变化的已知形状拟合估计。"
              "它们与 motion 的 RMS 不是同一统计量，不能相除作为运动分离百分比。", "",
              "## 分支替换检查", "", "所有图像 RRMSE 在完整 midpoint 图像上计算，使用原始预测。", "",
              "| 实验 | 完整模型 | 关闭 motion | 预测 motion + 真实 intensity | 真实 motion + 预测 intensity |",
              "|---|---:|---:|---:|---:|"]
    keys = ("full_model", "intensity_only_zero_motion", "predicted_motion_true_intensity", "true_motion_predicted_intensity")
    for name, report in reports.items():
        variants = report["image_variants_unseen_midpoints"]
        lines.append(f"| {name} | " + " | ".join(f"{variants[key]['rrmse']:.2%}" for key in keys) + " |")
    lines += ["", "## 图和诊断质量", ""]
    for name, report in reports.items():
        fit_error = report["intensity"]["template_fit_median_rrmse"]
        calibration = report["calibration"]
        lines += [f"### {name}", "",
                  f"- [Intensity 单独输出 GIF]({name}/intensity_residual.gif) · [残余平移、膨胀曲线]({name}/intensity_geometry.png)",
                  f"- [几何拟合检查 GIF]({name}/geometry_fit.gif)：中位拟合 RRMSE {fit_error:.2%}，用于检查几何估计是否可靠。",
                  f"- [ux/uy、幅度与误差曲线]({name}/motion_outputs.png)",
                  f"- [预测 motion + 真实 intensity GIF]({name}/motion_effect.gif)",
                  f"- [完整指标及源文件校验]({name}/metrics.json) · [几何拟合数组]({name}/diagnostics.npz)", "",
                  f"控制结果：静止参考图最大伪平移 {calibration['stationary_max_shift_pixels']:.2g} px；"
                  f"真实动态图的最大平移估计误差 {calibration['moving_max_translation_error_pixels']:.2g} px。", ""]
    lines += ["## 指标含义", "",
              "- 几何拟合依赖 toy 的已知形状，并非通用光流估计器。Complex 允许主体与亮点有两个独立亮度系数；simple 使用一个系数。",
              "- Motion 指标在全部未见帧的真实前景上计算；RMS 表示幅度，RMSE 表示误差。整体向量余弦为负不代表每个像素都反向。",
              "- 固定位置的 ux/uy 曲线不是组织点轨迹。图像显示范围固定 0–1，指标不裁剪。",
              "- diagnostics.npz 中 intensity_geometry 的列为 `[tx,ty,sx,sy,brightness gains...]`。tx/ty 是坐标单位，像素单位由保存的 x/y 间隔换算。",
              "- 所有诊断来自现有预测，不更新模型、不增加正则项。", "",
              "运行入口：`python -m motion_static.evaluation.diagnose_branches RUN_DIR [RUN_DIR ...] --out OUTPUT_DIR`。"
              "实现与复用说明见 `motion_static/evaluation/README.md`。"]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Completed nested-model experiment directories")
    parser.add_argument("--out", type=Path, required=True, help="Directory for generated diagnostics")
    args = parser.parse_args()
    names = [run.name for run in args.runs]
    if len(set(names)) != len(names):
        parser.error("Run directories must have distinct names to keep their outputs separate")
    reports = {run.name: diagnose_run(run, args.out / run.name) for run in args.runs}
    write_summary(args.out, reports)
    print(f"Saved diagnostics: {(args.out / 'README.md').resolve()}")


if __name__ == "__main__":
    main()
