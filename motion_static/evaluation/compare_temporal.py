"""Compare two saved complex midpoint diagnoses, including brightness fidelity.

Run: python -m motion_static.evaluation.compare_temporal BASELINE_DIAG NEW_DIAG --out OUTPUT_DIR
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .branch_metrics import relative_geometry
from .metrics import error_metrics, save_metrics
from .plots import pyplot


def load_result(directory):
    """Read completed diagnostics and the original image evaluation."""
    directory = Path(directory).resolve()
    diagnosis = json.loads((directory / "metrics.json").read_text())
    if diagnosis["phantom"] != "complex":
        raise ValueError("This comparison uses the complex phantom's two brightness components")
    run = Path(diagnosis["source"])
    report = json.loads((run / "summary.json").read_text())
    with np.load(directory / "diagnostics.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    gains = arrays["intensity_geometry"][:, 4:]
    true_gains = arrays["stationary_control_geometry"][:, 4:]
    # These are estimates from the geometry fit, not direct tissue measurements.
    brightness = dict(
        component_rrmse=[error_metrics(a, b, data_range=1)["rrmse"] for a, b in zip(true_gains.T, gains.T)],
        temporal_std_ratio=(gains.std(axis=0) / true_gains.std(axis=0)).tolist())
    image = report["image"]["unseen_time_midpoints"]
    scores = dict(source=str(run), diagnostic_directory=str(directory), config=report["config"],
                  image_rrmse=image["rrmse"], image_ssim=image["ssim"], roi_rrmse=image["roi"]["rrmse"],
                  reference_intensity_rrmse=report["reference_intensity"]["unseen_time_midpoints"]["rrmse"],
                  nonpositive_jacobian_fraction=report["motion"]["midpoint"]["foreground_nonpositive_jacobian_fraction"],
                  intensity=diagnosis["intensity"], motion=diagnosis["motion"], brightness_fit=brightness)
    return scores, arrays


def plot_comparison(out, results):
    plt = pyplot(out)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for (label, (scores, arrays)), color in zip(results.items(), ("#0072b2", "#d55e00")):
        times = arrays["t"]
        spacing = [arrays[axis][1] - arrays[axis][0] for axis in ("x", "y")]
        shift, _ = relative_geometry(arrays["intensity_geometry"], spacing)
        axes[0, 0].plot(times, np.linalg.norm(shift, axis=1), label=label, color=color)
        axes[0, 1].plot(times, scores["motion"]["per_frame_full_foreground_endpoint_rmse_pixels"], label=label, color=color)
        for i, ax in enumerate(axes[1]):
            ax.plot(times, arrays["intensity_geometry"][:, 4 + i], label=label, color=color)
    axes[0, 0].axhline(0, color="gray", linestyle="--", label="No residual shift")
    axes[0, 1].plot(times, scores["motion"]["per_frame_full_foreground_zero_baseline_pixels"],
                    color="gray", linestyle="--", label="Zero-motion prediction")
    for i, ax in enumerate(axes[1]):
        ax.plot(times, arrays["stationary_control_geometry"][:, 4 + i], color="black", linestyle="--", label="True brightness gain")
    titles = ("Intensity: residual translation", "Motion: foreground error",
              "Intensity: body brightness gain", "Intensity: first spot brightness gain")
    units = ("Shift from first midpoint (pixels)", "Endpoint RMSE (pixels)", "Fitted gain", "Fitted gain")
    for ax, title, unit in zip(axes.flat, titles, units):
        ax.set(title=title, xlabel="Time", ylabel=unit)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Complex / {scores['config']['model']} / unseen midpoints\n"
                 "Brightness gains are known-shape fit estimates; check fit RRMSE in the report")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def compare(baseline, current, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {"Original MSE": load_result(baseline), "Temporal L2": load_result(current)}
    original, regularized = results.values()
    for name in ("t", "x", "y", "stationary_control_geometry"):
        np.testing.assert_allclose(original[1][name], regularized[1][name])
    for key in ("phantom", "model", "size", "frames", "seed", "steps", "batch_size"):
        if original[0]["config"][key] != regularized[0]["config"][key]:
            raise ValueError(f"Different experiment setting: {key}")
    save_metrics(out / "comparison.json", {label: scores for label, (scores, _) in results.items()})
    plot_comparison(out, results)
    config = regularized[0]["config"]
    lines = ["# Complex：Intensity 时间差分 L2", "",
             "与已有 nested Hash NF 的原始 MSE 实验比较。模型、数据、seed、训练步数相同，只增加 intensity 时间惩罚。", "",
             "`loss = image_mse + weight * mean(((C(q,t+delta)-C(q,t))/delta)^2)`", "",
             f"- 权重：{config['temporal_weight']:g}；时间间隔：{config['temporal_step']:.8g}（输入时间单位）。",
             "- q 由训练坐标经过当前 motion 得到；两次查询保持同一个 q，正则项不向 motion 反传梯度。",
             "- 仅用训练图像监督，无真实 motion、参考强度图或 midpoint 标签。使用一个预先选定的权重，没有按测试结果搜索参数。",
             "- 该项也会惩罚真实亮度变化，因此同时检查几何、motion 与亮度。单个 phantom、单个 seed 不证明泛化。", "",
             "| 指标 | 原始 MSE | 加时间 L2 |", "|---|---:|---:|"]
    rows = [
        ("图像 RRMSE", lambda s: s["image_rrmse"], ".2%"),
        ("图像 SSIM", lambda s: s["image_ssim"], ".6f"),
        ("ROI RRMSE", lambda s: s["roi_rrmse"], ".2%"),
        ("参考强度图 RRMSE", lambda s: s["reference_intensity_rrmse"], ".2%"),
        ("Motion 前景 endpoint RMSE (px)", lambda s: s["motion"]["midpoint_foreground_endpoint_rmse_pixels"], ".3f"),
        ("前景非正形变 Jacobian 比例", lambda s: s["nonpositive_jacobian_fraction"], ".2%"),
        ("Intensity 最大残余平移 (px)", lambda s: s["intensity"]["max_translation_from_first_midpoint_pixels"], ".3f"),
        ("Intensity 几何拟合中位 RRMSE", lambda s: s["intensity"]["template_fit_median_rrmse"], ".2%"),
        ("主体亮度系数 RRMSE（拟合估计）", lambda s: s["brightness_fit"]["component_rrmse"][0], ".2%"),
        ("亮点亮度系数 RRMSE（拟合估计）", lambda s: s["brightness_fit"]["component_rrmse"][1], ".2%"),
        ("主体亮度时间标准差 / 真值", lambda s: s["brightness_fit"]["temporal_std_ratio"][0], ".3f"),
        ("亮点亮度时间标准差 / 真值", lambda s: s["brightness_fit"]["temporal_std_ratio"][1], ".3f")]
    for label, value, formatting in rows:
        lines.append(f"| {label} | " + " | ".join(format(value(scores), formatting) for scores, _ in results.values()) + " |")
    lines += ["", "亮度由允许平移、膨胀和两个独立亮度系数的已知形状拟合估计，仅用于评估。"
              "时间标准差比值为 1 表示波动幅度相当，不代表时间曲线完全正确；需结合 RRMSE 和曲线。", "",
              "![分工与亮度对比](comparison.png)", "", "## 结果入口", ""]
    for label, (scores, _) in results.items():
        run = Path(os.path.relpath(scores["source"], out)).as_posix()
        diagnostic = Path(os.path.relpath(scores["diagnostic_directory"], out)).as_posix()
        lines += [f"- **{label}**：[完整图像指标]({run}/plots/image/framewise_metrics.png) · [图像 GIF]({run}/plots/image.gif)",
                  f"  [Intensity 单独输出]({diagnostic}/intensity_residual.gif) · [残余几何]({diagnostic}/intensity_geometry.png) · "
                  f"[Motion 输出曲线]({diagnostic}/motion_outputs.png) · [Motion + 真实 intensity]({diagnostic}/motion_effect.gif)"]
    lines += ["", "[完整对比数字](comparison.json)。原有逐帧指标、ROI、时间曲线和 average adjacent truth frames 对照保留在各实验目录。"]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(out / "README.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.current, args.out)


if __name__ == "__main__":
    main()
