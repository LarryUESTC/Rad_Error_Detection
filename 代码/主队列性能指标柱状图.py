# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

DEFAULT_WORKBOOK = Path("exact_metrics_full_panels.xlsx")
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260420
METRICS = [
    ("Sensitivity", "Sensitivity"),
    ("Specificity", "Specificity"),
    ("PPV", "PPV"),
    ("NPV", "NPV"),
    ("F1-score", "F1-score"),
    ("Balanced accuracy", "Balanced accuracy"),
    ("MCC", "MCC"),
]
CLASS_ORDER = ["1a", "1b", "2a", "2b", "2c", "N"]
MODALITY_GROUPS = ["Total", "X-ray", "CT", "MRI"]
BODY_GROUPS = ["头部", "胸部", "脊柱", "腹部", "膝关节"]
BODY_DISPLAY = {
    "头部": "Head",
    "胸部": "Chest",
    "脊柱": "Spine",
    "腹部": "Abdomen",
    "膝关节": "Knee",
}
MODALITY_BODY_GROUPS = {
    "Total": ["Total_头部", "Total_胸部", "Total_脊柱", "Total_腹部", "Total_膝关节"],
    "X-ray": ["X-ray_胸部", "X-ray_脊柱", "X-ray_腹部", "X-ray_膝关节"],
    "CT": ["CT_头部", "CT_胸部", "CT_脊柱", "CT_腹部", "CT_膝关节"],
    "MRI": ["MRI_头部", "MRI_脊柱", "MRI_腹部", "MRI_膝关节"],
}
MODALITY_COLORS = {
    "Total": "#E7B88F",
    "X-ray": "#B9DDEA",
    "CT": "#5C83A1",
    "MRI": "#C7DDB3",
}
BODY_COLORS = {
    "Head": "#163A5F",
    "Chest": "#B9DDEA",
    "Spine": "#5C83A1",
    "Abdomen": "#C7DDB3",
    "Knee": "#657CDE",
}


def configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for preferred in ["Arial", "Microsoft YaHei", "Calibri", "DejaVu Sans"]:
        if preferred in available_fonts:
            plt.rcParams["font.family"] = preferred
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["axes.linewidth"] = 0.8


def percentile_ci(values: list[float]) -> tuple[float | None, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None, None
    low, high = np.percentile(array, [2.5, 97.5])
    return float(low), float(high)


def safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom != 0)


def multiclass_mcc(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=float)
    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    n_correct = float(np.trace(cm))
    n_samples = float(cm.sum())
    cov_ytyp = n_correct * n_samples - float(np.dot(t_sum, p_sum))
    cov_ypyp = n_samples * n_samples - float(np.dot(p_sum, p_sum))
    cov_ytyt = n_samples * n_samples - float(np.dot(t_sum, t_sum))
    denom = math.sqrt(max(cov_ytyt, 0.0) * max(cov_ypyp, 0.0))
    if denom == 0:
        return 0.0
    return cov_ytyp / denom


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict[str, float]:
    cm = np.asarray(cm, dtype=float)
    total = float(cm.sum())
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    fn = row_sum - tp
    fp = col_sum - tp
    tn = total - tp - fn - fp
    active = row_sum > 0

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    ppv = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    balanced = (sensitivity + specificity) / 2

    if not np.any(active):
        return {metric: 0.0 for metric, _ in METRICS}

    return {
        "Sensitivity": float(np.mean(sensitivity[active])),
        "Specificity": float(np.mean(specificity[active])),
        "PPV": float(np.mean(ppv[active])),
        "NPV": float(np.mean(npv[active])),
        "F1-score": float(np.mean(f1[active])),
        "Balanced accuracy": float(np.mean(balanced[active])),
        "MCC": float(multiclass_mcc(cm)),
    }


def bootstrap_metrics(cm: np.ndarray, seed: int) -> dict[str, tuple[float | None, float | None]]:
    cm = np.asarray(cm, dtype=int)
    row_sum = cm.sum(axis=1)
    rng = np.random.default_rng(seed)
    collect = {metric: [] for metric, _ in METRICS}

    for _ in range(BOOTSTRAP_REPS):
        sampled_rows = []
        for row, n in zip(cm, row_sum):
            if n <= 0:
                sampled_rows.append(np.zeros_like(row))
                continue
            probs = row / n
            sampled_rows.append(rng.multinomial(int(n), probs))
        boot_cm = np.vstack(sampled_rows)
        metrics = metrics_from_confusion_matrix(boot_cm)
        for metric, value in metrics.items():
            collect[metric].append(value)

    return {metric: percentile_ci(values) for metric, values in collect.items()}


def normalize_sheet_prefix(group: str) -> str:
    return (
        group.replace("X-ray", "Xray")
        .replace("头部", "Head")
        .replace("胸部", "Chest")
        .replace("腹部", "Abdomen")
        .replace("脊柱", "Spine")
        .replace("膝关节", "Knee")
    )


def display_group_name(group: str) -> str:
    if group in MODALITY_GROUPS:
        return group
    if group in BODY_DISPLAY:
        return BODY_DISPLAY[group]
    if "_" in group:
        _, body = group.split("_", 1)
        return BODY_DISPLAY.get(body, body)
    return group


def load_confusion_matrix(workbook: Path, group: str) -> pd.DataFrame:
    prefix = normalize_sheet_prefix(group)
    cm = pd.read_excel(workbook, sheet_name=f"{prefix}_cm_count", index_col=0)
    rows = [index.replace("True ", "") for index in cm.index.astype(str)]
    cols = [column.replace("Pred ", "") for column in cm.columns.astype(str)]
    cm.index = rows
    cm.columns = cols
    return cm.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)


def build_metrics_frame(workbook: Path, groups: list[str], start_seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, group in enumerate(groups):
        cm_df = load_confusion_matrix(workbook, group)
        cm = cm_df.to_numpy(dtype=int)
        metrics = metrics_from_confusion_matrix(cm)
        cis = bootstrap_metrics(cm, start_seed + offset)
        row: dict[str, object] = {"Group": group, "DisplayGroup": display_group_name(group), "TotalInstances": int(cm.sum())}
        for metric, _ in METRICS:
            low, high = cis[metric]
            row[metric] = round(metrics[metric], 6)
            row[f"{metric}_CI_low"] = None if low is None else round(low, 6)
            row[f"{metric}_CI_high"] = None if high is None else round(high, 6)
            row[f"{metric}_95CI"] = None if low is None or high is None else f"[{low:.6f}, {high:.6f}]"
        row["AUROC"] = None
        row["AUPRC"] = None
        row["Note"] = "AUROC/AUPRC cannot be derived from this hard-label workbook without probabilistic scores."
        rows.append(row)
    return pd.DataFrame(rows)


def calc_xlim(plot_frames: list[pd.DataFrame]) -> float:
    upper = 1.06
    for frame in plot_frames:
        if frame.empty:
            continue
        for metric, _ in METRICS:
            finite = pd.to_numeric(frame[f"{metric}_CI_high"], errors="coerce").dropna()
            if not finite.empty:
                upper = max(upper, float(finite.max()) + 0.08)
    return min(1.16, upper)


def add_value_labels(ax: plt.Axes, centers: np.ndarray, values: np.ndarray, highs: np.ndarray, x_limit: float) -> None:
    placed_y: list[float] = []
    for y, value, high in zip(centers, values, highs):
        x = min(max(value, high) + 0.008, x_limit - 0.015)
        adjusted_y = y
        if placed_y and abs(adjusted_y - placed_y[-1]) < 0.08:
            adjusted_y += 0.03 if len(placed_y) % 2 == 0 else -0.03
        placed_y.append(adjusted_y)
        ax.text(x, adjusted_y, f"{value:.2f}", va="center", ha="left", fontsize=5.9, color="#333333")


def draw_grouped_horizontal_bars(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    colors: dict[str, str],
    title: str,
    panel_letter: str,
    x_limit: float,
    show_y_labels: bool = True,
) -> None:
    metrics_raw = [metric for metric, _ in METRICS]
    metrics_display = [label for _, label in METRICS]
    y_base = np.arange(len(metrics_raw))
    n_groups = max(len(plot_df), 1)
    bar_height = 1.00 / n_groups
    offsets = (np.arange(len(plot_df)) - (len(plot_df) - 1) / 2) * bar_height if len(plot_df) else np.array([])

    ax.set_xlim(0.0, x_limit)
    for idx, row in plot_df.iterrows():
        centers = y_base + offsets[idx]
        display = row["DisplayGroup"]
        color = colors[display]
        values = row[metrics_raw].astype(float).to_numpy()
        lows = np.array([float(row[f"{metric}_CI_low"]) for metric in metrics_raw], dtype=float)
        highs = np.array([float(row[f"{metric}_CI_high"]) for metric in metrics_raw], dtype=float)
        xerr = np.vstack([values - lows, highs - values])
        ax.barh(
            centers,
            values,
            height=bar_height * 0.90,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.errorbar(
            values,
            centers,
            xerr=xerr,
            fmt="none",
            ecolor="#333333",
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=4,
        )
        add_value_labels(ax, centers, values, highs, x_limit)

    ax.set_yticks(y_base)
    ax.set_yticklabels(metrics_display if show_y_labels else [])
    ax.invert_yaxis()
    ax.set_title(title, pad=8, fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="x", length=3, color="#BBBBBB")
    ax.tick_params(axis="y", length=0)
    ax.text(-0.12, 1.03, panel_letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance (error bars indicate 95% CI)")
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))


def export_plot_values(writer: pd.ExcelWriter, sheet_name: str, plot_df: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for _, row in plot_df.iterrows():
        for metric, display_metric in METRICS:
            rows.append(
                {
                    "Group": row["DisplayGroup"],
                    "Metric": display_metric,
                    "Value": float(row[metric]),
                    "CI_low": float(row[f"{metric}_CI_low"]),
                    "CI_high": float(row[f"{metric}_CI_high"]),
                    "95% CI": row[f"{metric}_95CI"],
                }
            )
    pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)


def save_combined_figure(
    output_dir: Path,
    modality_df: pd.DataFrame,
    body_df: pd.DataFrame,
    subgroup_frames: dict[str, pd.DataFrame],
) -> None:
    x_limit = calc_xlim([modality_df, body_df, *subgroup_frames.values()])
    fig = plt.figure(figsize=(18.8, 11.8), dpi=220)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.40, wspace=0.60)
    axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]
    draw_grouped_horizontal_bars(axes[0], modality_df, MODALITY_COLORS, "Performance by Imaging Modality", "A", x_limit)
    draw_grouped_horizontal_bars(axes[1], body_df, BODY_COLORS, "Performance by Body Region", "B", x_limit)
    draw_grouped_horizontal_bars(axes[2], subgroup_frames["X-ray"], BODY_COLORS, "X-ray", "C", x_limit, show_y_labels=True)
    draw_grouped_horizontal_bars(axes[3], subgroup_frames["CT"], BODY_COLORS, "CT", "D", x_limit, show_y_labels=True)
    draw_grouped_horizontal_bars(axes[4], subgroup_frames["MRI"], BODY_COLORS, "MRI", "E", x_limit, show_y_labels=True)

    modality_handles = [Patch(facecolor=MODALITY_COLORS[g], edgecolor="none", label=g) for g in MODALITY_GROUPS]
    body_labels = [BODY_DISPLAY[g] for g in BODY_GROUPS]
    body_handles = [Patch(facecolor=BODY_COLORS[g], edgecolor="none", label=g) for g in body_labels]
    fig.legend(
        handles=modality_handles,
        labels=MODALITY_GROUPS,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.4,
    )
    fig.legend(
        handles=body_handles,
        labels=body_labels,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.995),
        ncol=5,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.4,
    )
    fig.savefig(output_dir / "Figure3_like_binary_metrics_main.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Figure3_like_binary_metrics_main.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_total_subgroup_figure(output_dir: Path, total_df: pd.DataFrame) -> None:
    x_limit = calc_xlim([total_df])
    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=220)
    draw_grouped_horizontal_bars(ax, total_df, BODY_COLORS, "Total", "F", x_limit, show_y_labels=True)
    body_labels = [BODY_DISPLAY[g] for g in BODY_GROUPS]
    body_handles = [Patch(facecolor=BODY_COLORS[g], edgecolor="none", label=g) for g in body_labels]
    fig.legend(
        handles=body_handles,
        labels=body_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "Figure3_like_binary_metrics_total_subgroups.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Figure3_like_binary_metrics_total_subgroups.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_notes(output_dir: Path) -> None:
    note = (
        "This figure was derived from hard-label confusion-matrix sheets in exact_metrics_full_panels.xlsx. "
        "Sensitivity, Specificity, PPV, NPV, F1-score, Balanced accuracy, and MCC were recalculated from the "
        "count matrices with stratified bootstrap 95% CIs. AUROC and AUPRC are not available from this workbook "
        "because no probabilistic scores were provided."
    )
    (output_dir / "Figure3_like_binary_metrics_README.txt").write_text(note, encoding="utf-8")


def main() -> None:
    configure_plot_style()
    workbook = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")

    output_dir = workbook.parent

    modality_df = build_metrics_frame(workbook, MODALITY_GROUPS, BOOTSTRAP_SEED)
    body_df = build_metrics_frame(workbook, BODY_GROUPS, BOOTSTRAP_SEED + 1000)
    subgroup_frames = {
        modality: build_metrics_frame(workbook, groups, BOOTSTRAP_SEED + 2000 + idx * 100)
        for idx, (modality, groups) in enumerate(MODALITY_BODY_GROUPS.items())
    }

    save_combined_figure(output_dir, modality_df, body_df, subgroup_frames)
    save_total_subgroup_figure(output_dir, subgroup_frames["Total"])
    save_notes(output_dir)

    with pd.ExcelWriter(output_dir / "Figure3_like_binary_metrics_values.xlsx", engine="openpyxl") as writer:
        export_plot_values(writer, "Modality", modality_df)
        export_plot_values(writer, "BodyRegion", body_df)
        export_plot_values(writer, "Total_Subgroups", subgroup_frames["Total"])
        export_plot_values(writer, "Xray_Subgroups", subgroup_frames["X-ray"])
        export_plot_values(writer, "CT_Subgroups", subgroup_frames["CT"])
        export_plot_values(writer, "MRI_Subgroups", subgroup_frames["MRI"])
        pd.DataFrame(
            [
                {
                    "Item": "AUROC/AUPRC",
                    "Value": "Not derived. The workbook only contains hard-label confusion matrices and no probabilistic scores.",
                }
            ]
        ).to_excel(writer, sheet_name="Notes", index=False)

    print(f"Workbook: {workbook}")
    print(f"Output directory: {output_dir}")
    print(f"Created: {output_dir / 'Figure3_like_binary_metrics_main.png'}")


if __name__ == "__main__":
    main()
