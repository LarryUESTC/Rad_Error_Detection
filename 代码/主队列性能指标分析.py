# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import analyze_final_included_lower_type1_acc as base
from exclude_ai_error_from_raw import ROOT, ensure_unique_dir


OUTPUT_ROOT_NAME = "最终纳入"
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260419
ALL_LABELS = ["a", "b", "c", "d", "e", "f"]
ALL_DISPLAY = {"a": "1a", "b": "1b", "c": "2a", "d": "2b", "e": "2c", "f": "N"}
MODALITY_GROUPS = {"Total": None, "X-ray": "DR", "CT": "CT", "MRI": "MRI"}
BODY_GROUPS = ["头部", "胸部", "腹部", "脊柱", "膝关节"]
BODY_DISPLAY = {"头部": "Head", "胸部": "Chest", "腹部": "Abdomen", "脊柱": "Spine", "膝关节": "Knee"}
TYPE_GROUPS = {
    "Type 1": {"labels": ["a", "b"], "display": {"a": "1a", "b": "1b"}},
    "Type 2": {"labels": ["c", "d", "e"], "display": {"c": "2a", "d": "2b", "e": "2c"}},
}
SECTION_COLORS = {
    "Total": "#163A5F",
    "X-ray": "#4F83C2",
    "CT": "#7BA6D6",
    "MRI": "#D88C5A",
    "头部": "#163A5F",
    "胸部": "#B9DDEA",
    "腹部": "#C7DDB3",
    "脊柱": "#5C83A1",
    "膝关节": "#C77D7D",
    "Type 1": "#4F83C2",
    "Type 2": "#D88C5A",
}
SUMMARY_METRICS = ["ACC", "Balanced accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1", "Weighted-F1"]


def configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for preferred in ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]:
        if preferred in available_fonts:
            plt.rcParams["font.family"] = preferred
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def format_metric(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return round(float(value), 6)


def format_ci(low: float | None, high: float | None) -> str | None:
    if low is None or high is None:
        return None
    return f"[{low:.6f}, {high:.6f}]"


def display_group_name(group: str) -> str:
    label = group.replace("头部", "Head").replace("胸部", "Chest").replace("腹部", "Abdomen").replace("脊柱", "Spine").replace("膝关节", "Knee")
    return BODY_DISPLAY.get(label, label)


def percentile_ci(values: list[float]) -> tuple[float | None, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None, None
    low, high = np.percentile(array, [2.5, 97.5])
    return float(low), float(high)


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_sums, out=np.zeros_like(values, dtype=float), where=row_sums != 0)
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], display_map: dict[str, str]) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    active_labels = [label for label in labels if np.any(y_true == label)]
    if y_true.size == 0 or not active_labels:
        summary = {metric: 0.0 for metric in SUMMARY_METRICS}
        per_class_df = pd.DataFrame(
            [
                {
                    "ClassRaw": label,
                    "ClassPaper": display_map[label],
                    "Support": 0,
                    "Predicted": 0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F1": 0.0,
                }
                for label in labels
            ]
        )
        cm_df = pd.DataFrame(
            0,
            index=[f"True {display_map[label]}" for label in labels],
            columns=[f"Pred {display_map[label]}" for label in labels],
        )
        return summary, per_class_df, cm_df

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=active_labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=active_labels,
        average="weighted",
        zero_division=0,
    )
    recall_map = {label: float(r) for label, r in zip(labels, recall) if label in active_labels}
    summary = {
        "ACC": float(np.mean(y_true == y_pred)),
        "Balanced accuracy": float(np.mean([recall_map[label] for label in active_labels])),
        "Macro-Precision": float(macro_precision),
        "Macro-Recall": float(macro_recall),
        "Macro-F1": float(macro_f1),
        "Weighted-F1": float(weighted_f1),
    }
    per_class_df = pd.DataFrame(
        [
            {
                "ClassRaw": label,
                "ClassPaper": display_map[label],
                "Support": int(cls_support),
                "Predicted": int(np.sum(y_pred == label)),
                "Precision": float(cls_precision),
                "Recall": float(cls_recall),
                "F1": float(cls_f1),
            }
            for label, cls_precision, cls_recall, cls_f1, cls_support in zip(labels, precision, recall, f1, support)
        ]
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {display_map[label]}" for label in labels],
        columns=[f"Pred {display_map[label]}" for label in labels],
    )
    return summary, per_class_df, cm_df


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], display_map: dict[str, str], seed: int) -> tuple[dict[str, tuple[float | None, float | None]], dict[tuple[str, str], tuple[float | None, float | None]]]:
    active_labels = [label for label in labels if np.any(y_true == label)]
    if y_true.size == 0 or not active_labels:
        return {}, {}

    rng = np.random.default_rng(seed)
    class_indices = {label: np.where(y_true == label)[0] for label in active_labels}
    summary_collect = {metric: [] for metric in SUMMARY_METRICS}
    per_class_collect = {(label, metric): [] for label in active_labels for metric in ["Precision", "Recall", "F1"]}

    for _ in range(BOOTSTRAP_REPS):
        sampled_blocks = [rng.choice(indices, size=indices.size, replace=True) for indices in class_indices.values()]
        sampled_idx = np.concatenate(sampled_blocks)
        boot_true = y_true[sampled_idx]
        boot_pred = y_pred[sampled_idx]
        summary, per_class_df, _ = compute_metrics(boot_true, boot_pred, labels, display_map)
        for metric, value in summary.items():
            summary_collect[metric].append(float(value))
        for _, row in per_class_df.iterrows():
            if row["ClassRaw"] not in active_labels:
                continue
            for metric in ["Precision", "Recall", "F1"]:
                per_class_collect[(row["ClassRaw"], metric)].append(float(row[metric]))

    summary_ci = {metric: percentile_ci(values) for metric, values in summary_collect.items()}
    per_class_ci = {key: percentile_ci(values) for key, values in per_class_collect.items()}
    return summary_ci, per_class_ci


def finalize_group_outputs(group_name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], display_map: dict[str, str], seed: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, per_class_df, cm_df = compute_metrics(y_true, y_pred, labels, display_map)
    summary_ci, per_class_ci = bootstrap_metrics(y_true, y_pred, labels, display_map, seed)

    summary_row = {"Group": group_name, "TotalInstances": int(y_true.size)}
    for metric in SUMMARY_METRICS:
        low, high = summary_ci.get(metric, (None, None))
        summary_row[metric] = format_metric(summary[metric])
        summary_row[f"{metric}_CI_low"] = format_metric(low)
        summary_row[f"{metric}_CI_high"] = format_metric(high)
        summary_row[f"{metric}_95CI"] = format_ci(low, high)

    per_class_df = per_class_df.copy()
    for metric in ["Precision", "Recall", "F1"]:
        lows = []
        highs = []
        cis = []
        for label in per_class_df["ClassRaw"]:
            low, high = per_class_ci.get((label, metric), (None, None))
            lows.append(format_metric(low))
            highs.append(format_metric(high))
            cis.append(format_ci(low, high))
        per_class_df[metric] = per_class_df[metric].map(format_metric)
        per_class_df[f"{metric}_CI_low"] = lows
        per_class_df[f"{metric}_CI_high"] = highs
        per_class_df[f"{metric}_95CI"] = cis

    return summary_row, per_class_df, cm_df, normalize_rows(cm_df)


def build_modality_outputs(instance_records):
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    count_map: dict[str, pd.DataFrame] = {}
    norm_map: dict[str, pd.DataFrame] = {}

    for idx, (group_name, modality_code) in enumerate(MODALITY_GROUPS.items()):
        subset = [
            record
            for record in instance_records
            if record.y_true in base.LABEL_ORDER and (modality_code is None or record.modality == modality_code)
        ]
        y_true = np.array([record.y_true for record in subset], dtype=object)
        y_pred = np.array([record.y_pred if record.y_pred in base.LABEL_ORDER else "f" for record in subset], dtype=object)
        summary_row, per_class_df, count_df, norm_df = finalize_group_outputs(group_name, y_true, y_pred, ALL_LABELS, ALL_DISPLAY, BOOTSTRAP_SEED + idx)
        summary_rows.append(summary_row)
        per_class_df.insert(0, "Group", group_name)
        per_class_rows.extend(per_class_df.to_dict("records"))
        count_map[group_name] = count_df
        norm_map[group_name] = norm_df

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows), count_map, norm_map


def build_body_outputs(instance_records):
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    count_map: dict[str, pd.DataFrame] = {}
    norm_map: dict[str, pd.DataFrame] = {}

    for idx, group_name in enumerate(BODY_GROUPS, start=100):
        subset = [record for record in instance_records if record.y_true in base.LABEL_ORDER and record.body_part == group_name]
        y_true = np.array([record.y_true for record in subset], dtype=object)
        y_pred = np.array([record.y_pred if record.y_pred in base.LABEL_ORDER else "f" for record in subset], dtype=object)
        summary_row, per_class_df, count_df, norm_df = finalize_group_outputs(group_name, y_true, y_pred, ALL_LABELS, ALL_DISPLAY, BOOTSTRAP_SEED + idx)
        summary_rows.append(summary_row)
        per_class_df.insert(0, "Group", group_name)
        per_class_rows.extend(per_class_df.to_dict("records"))
        count_map[group_name] = count_df
        norm_map[group_name] = norm_df

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows), count_map, norm_map


def build_modality_body_outputs(instance_records):
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    count_map: dict[str, pd.DataFrame] = {}
    norm_map: dict[str, pd.DataFrame] = {}

    combos: list[tuple[str, str, str | None]] = []
    for body in BODY_GROUPS:
        combos.append((f"Total_{body}", body, None))
    for modality_name, modality_code in [("X-ray", "DR"), ("CT", "CT"), ("MRI", "MRI")]:
        for body in BODY_GROUPS:
            combos.append((f"{modality_name}_{body}", body, modality_code))

    for idx, (group_name, body_name, modality_code) in enumerate(combos, start=300):
        subset = [
            record
            for record in instance_records
            if record.y_true in base.LABEL_ORDER
            and record.body_part == body_name
            and (modality_code is None or record.modality == modality_code)
        ]
        y_true = np.array([record.y_true for record in subset], dtype=object)
        y_pred = np.array([record.y_pred if record.y_pred in base.LABEL_ORDER else "f" for record in subset], dtype=object)
        summary_row, per_class_df, count_df, norm_df = finalize_group_outputs(group_name, y_true, y_pred, ALL_LABELS, ALL_DISPLAY, BOOTSTRAP_SEED + idx)
        summary_rows.append(summary_row)
        per_class_df.insert(0, "Group", group_name)
        per_class_rows.extend(per_class_df.to_dict("records"))
        count_map[group_name] = count_df
        norm_map[group_name] = norm_df

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows), count_map, norm_map


def build_type_outputs(instance_records):
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    count_map: dict[str, pd.DataFrame] = {}
    norm_map: dict[str, pd.DataFrame] = {}

    for idx, (group_name, config) in enumerate(TYPE_GROUPS.items(), start=500):
        labels = config["labels"]
        subset = [record for record in instance_records if record.y_true in labels]
        y_true = np.array([record.y_true for record in subset], dtype=object)
        y_pred = np.array([record.y_pred if record.y_pred in base.LABEL_ORDER else "f" for record in subset], dtype=object)
        summary_row, per_class_df, count_df, norm_df = finalize_group_outputs(group_name, y_true, y_pred, labels, config["display"], BOOTSTRAP_SEED + idx)
        summary_rows.append(summary_row)
        per_class_df.insert(0, "Group", group_name)
        per_class_rows.extend(per_class_df.to_dict("records"))
        count_map[group_name] = count_df
        norm_map[group_name] = norm_df

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows), count_map, norm_map


def plot_summary_panel(summary_df: pd.DataFrame, groups: list[str], title: str, output_png: Path, output_pdf: Path) -> None:
    metrics = ["ACC", "Balanced accuracy", "Macro-F1"]
    x = np.arange(len(metrics))
    width = min(0.18, 0.8 / max(len(groups), 1))

    fig, ax = plt.subplots(figsize=(max(8.5, len(groups) * 1.2), 5.8))
    for idx, group in enumerate(groups):
        row = summary_df[summary_df["Group"] == group].iloc[0]
        vals = np.array([float(row[metric]) for metric in metrics], dtype=float)
        lows = np.array([float(row[f"{metric}_CI_low"]) if pd.notna(row[f"{metric}_CI_low"]) else np.nan for metric in metrics], dtype=float)
        highs = np.array([float(row[f"{metric}_CI_high"]) if pd.notna(row[f"{metric}_CI_high"]) else np.nan for metric in metrics], dtype=float)
        offset = (idx - (len(groups) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=SECTION_COLORS.get(group, "#7BA6D6"),
            edgecolor="black",
            linewidth=0.4,
            label=display_group_name(group),
        )
        lower = np.where(np.isfinite(lows), vals - lows, np.nan)
        upper = np.where(np.isfinite(highs), highs - vals, np.nan)
        valid = np.isfinite(lower) & np.isfinite(upper)
        if valid.any():
            ax.errorbar(
                (x + offset)[valid],
                vals[valid],
                yerr=np.vstack([lower[valid], upper[valid]]),
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=2,
            )
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, min(1.02, value + 0.01), f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=min(4, len(groups)), frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=500, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_grid(confusion_map: dict[str, pd.DataFrame], groups: list[str], titles: dict[str, str], output_png: Path, output_pdf: Path) -> None:
    cols = 2 if len(groups) > 1 else 1
    rows = int(np.ceil(len(groups) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    im = None

    for ax, group, panel_idx in zip(axes, groups, range(len(groups))):
        cm_norm = confusion_map[group].iloc[::-1]
        im = ax.imshow(cm_norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(np.arange(cm_norm.shape[1]))
        ax.set_yticks(np.arange(cm_norm.shape[0]))
        ax.set_xticklabels(cm_norm.columns, rotation=25, ha="right")
        ax.set_yticklabels(cm_norm.index)
        ax.set_title(titles[group], fontweight="bold", pad=10)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                value = cm_norm.iloc[i, j]
                ax.text(j, i, f"{value:.1%}", ha="center", va="center", color="white" if value > 0.5 else "black", fontsize=8)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.text(-0.12, 1.04, chr(ord("A") + panel_idx), transform=ax.transAxes, fontsize=16, fontweight="bold")

    for ax in axes[len(groups):]:
        ax.axis("off")

    fig.subplots_adjust(right=0.90, top=0.92, wspace=0.28, hspace=0.32)
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Row-normalized percentage")
    fig.savefig(output_png, dpi=500, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not base.FINAL_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {base.FINAL_DIR}")

    positive_dir = base.FINAL_DIR / "AI真假阳性"
    negative_dir = base.FINAL_DIR / "AI真阴性"
    if not positive_dir.exists():
        raise FileNotFoundError(f"Missing directory: {positive_dir}")
    if not negative_dir.exists():
        raise FileNotFoundError(f"Missing directory: {negative_dir}")

    base.POSITIVE_DIR = positive_dir
    base.NEGATIVE_DIR = negative_dir

    configure_plot_style()
    output_root = ensure_unique_dir(ROOT / OUTPUT_ROOT_NAME)
    output_root.mkdir(parents=True, exist_ok=False)

    _, _, report_records, instance_records, file_stats_df = base.load_specified_records()
    modality_summary_df, modality_per_class_df, modality_counts, modality_norm = build_modality_outputs(instance_records)
    body_summary_df, body_per_class_df, body_counts, body_norm = build_body_outputs(instance_records)
    modality_body_summary_df, modality_body_per_class_df, modality_body_counts, modality_body_norm = build_modality_body_outputs(instance_records)
    type_summary_df, type_per_class_df, type_counts, type_norm = build_type_outputs(instance_records)

    workbook_path = output_root / "exact_metrics_full_panels.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {"Item": "Positive directory", "Value": str(positive_dir)},
                {"Item": "Negative directory", "Value": str(negative_dir)},
                {"Item": "Total reports", "Value": len(report_records)},
                {"Item": "Total label instances", "Value": len(instance_records)},
                {"Item": "True error reports", "Value": sum(1 for record in report_records if record.true_has_error)},
                {"Item": "True error instances", "Value": sum(1 for record in instance_records if record.y_true in base.ERROR_LABELS)},
                {"Item": "Evaluation rule", "Value": "Only exact subtype match is correct; true a and pred b is incorrect even though both are Type 1."},
                {"Item": "Instance rule", "Value": "Error1 / Error2 / Error3 are all counted as separate label instances."},
            ]
        ).to_excel(writer, sheet_name="overview", index=False)
        file_stats_df.to_excel(writer, sheet_name="file_stats", index=False)
        modality_summary_df.to_excel(writer, sheet_name="modality_summary", index=False)
        modality_per_class_df.to_excel(writer, sheet_name="modality_per_class", index=False)
        body_summary_df.to_excel(writer, sheet_name="bodypart_summary", index=False)
        body_per_class_df.to_excel(writer, sheet_name="bodypart_per_class", index=False)
        modality_body_summary_df.to_excel(writer, sheet_name="modality_bodypart_summary", index=False)
        modality_body_per_class_df.to_excel(writer, sheet_name="modality_bodypart_per_class", index=False)
        type_summary_df.to_excel(writer, sheet_name="type_summary", index=False)
        type_per_class_df.to_excel(writer, sheet_name="type_per_class", index=False)

        for group, df in modality_counts.items():
            safe = group.replace("X-ray", "Xray")
            df.to_excel(writer, sheet_name=f"{safe}_cm_count"[:31])
            modality_norm[group].to_excel(writer, sheet_name=f"{safe}_cm_norm"[:31])
        for group, df in body_counts.items():
            safe = BODY_DISPLAY.get(group, group)
            df.to_excel(writer, sheet_name=f"{safe}_cm_count"[:31])
            body_norm[group].to_excel(writer, sheet_name=f"{safe}_cm_norm"[:31])
        for group, df in modality_body_counts.items():
            safe = group.replace("X-ray", "Xray").replace("头部", "Head").replace("胸部", "Chest").replace("腹部", "Abdomen").replace("脊柱", "Spine").replace("膝关节", "Knee")
            df.to_excel(writer, sheet_name=f"{safe}_cm_count"[:31])
            modality_body_norm[group].to_excel(writer, sheet_name=f"{safe}_cm_norm"[:31])
        for group, df in type_counts.items():
            safe = group.replace(" ", "")
            df.to_excel(writer, sheet_name=f"{safe}_cm_count"[:31])
            type_norm[group].to_excel(writer, sheet_name=f"{safe}_cm_norm"[:31])

    plot_summary_panel(modality_summary_df, list(MODALITY_GROUPS), "Exact-Label Main Metrics by Modality", output_root / "SCI_modality_exact_main_metrics.png", output_root / "SCI_modality_exact_main_metrics.pdf")
    plot_summary_panel(body_summary_df, BODY_GROUPS, "Exact-Label Main Metrics by Body Region", output_root / "SCI_bodypart_exact_main_metrics.png", output_root / "SCI_bodypart_exact_main_metrics.pdf")
    plot_summary_panel(modality_body_summary_df, modality_body_summary_df["Group"].tolist(), "Exact-Label Main Metrics by Modality and Body Region", output_root / "SCI_modality_bodypart_exact_main_metrics.png", output_root / "SCI_modality_bodypart_exact_main_metrics.pdf")
    plot_summary_panel(type_summary_df, list(TYPE_GROUPS), "Exact-Label Main Metrics by Type Group", output_root / "SCI_type_exact_main_metrics.png", output_root / "SCI_type_exact_main_metrics.pdf")

    plot_confusion_grid(modality_norm, list(MODALITY_GROUPS), {group: group for group in MODALITY_GROUPS}, output_root / "SCI_modality_confusion_matrices.png", output_root / "SCI_modality_confusion_matrices.pdf")
    plot_confusion_grid(body_norm, BODY_GROUPS, {group: BODY_DISPLAY[group] for group in BODY_GROUPS}, output_root / "SCI_bodypart_confusion_matrices.png", output_root / "SCI_bodypart_confusion_matrices.pdf")
    plot_confusion_grid(modality_body_norm, modality_body_summary_df["Group"].tolist(), {group: display_group_name(group) for group in modality_body_summary_df["Group"].tolist()}, output_root / "SCI_modality_bodypart_confusion_matrices.png", output_root / "SCI_modality_bodypart_confusion_matrices.pdf")
    plot_confusion_grid(type_norm, list(TYPE_GROUPS), {group: group for group in TYPE_GROUPS}, output_root / "SCI_type_confusion_matrices.png", output_root / "SCI_type_confusion_matrices.pdf")

    print(f"Output root: {output_root}")
    print(f"Workbook: {workbook_path}")


if __name__ == "__main__":
    main()
