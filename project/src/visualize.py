"""
visualize.py

ROC-AUC centered visualization (primary metric)
Secondary: F1 @ Youden's J threshold

Output: artifacts/results{suffix}/figures/

Usage:
  python visualize.py
  python visualize.py --augment smote
  python visualize.py --augment gan
  python visualize.py --augment wgan_gp
  python visualize.py --augment wcgan_gp
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# --augment argument
# =========================================================
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--augment",
    type=str,
    default="none",
    choices=["none", "smote", "gan", "wgan_gp", "wcgan_gp"],
)
AUGMENT = _parser.parse_args().augment


# =========================================================
# Paths
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent
_ROOT     = _PROJECT.parent

_SUFFIX    = f"_{AUGMENT}" if AUGMENT != "none" else ""
RESULT_DIR = _ROOT / "artifacts" / f"results{_SUFFIX}"
FIGURE_DIR = RESULT_DIR / "total_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PATH  = RESULT_DIR / "eval_results.json"


# =========================================================
# Style
# =========================================================
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})

MODELS    = ["RF", "XGBoost", "CNN-LSTM", "GRU", "CNN-GRU"]
MODEL_KEY = ["rf", "xgb", "cnn_lstm", "gru", "cnn_gru"]

COLORS = {
    "RF":       "#4C72B0",
    "XGBoost":  "#DD8452",
    "CNN-LSTM": "#55A868",
    "GRU":      "#C44E52",
    "CNN-GRU":  "#8172B2",
}

AUGMENT_LABEL = {
    "none":     "Baseline",
    "smote":    "SMOTE",
    "gan":      "GAN",
    "wgan_gp":  "WGAN-GP",
    "wcgan_gp": "WCGAN-GP",
}.get(AUGMENT, AUGMENT)

DATASET_COLORS = {
    "CIC2017": "#4878D0",
    "CIC2018": "#EE854A",
    "CTU-13":  "#6ACC65",
}


# =========================================================
# Data loading
# =========================================================
def load_results() -> dict:
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_stage(results: dict, stage_key: str) -> dict:
    stage = results.get(stage_key, {})
    name_map = {
        "rf": "RF", "xgb": "XGBoost",
        "cnn_lstm": "CNN-LSTM", "gru": "GRU", "cnn_gru": "CNN-GRU",
    }
    return {name_map[k]: v for k, v in stage.items() if k in name_map}


def get_val(data: dict, model: str, metric: str, default=0.0):
    return data.get(model, {}).get(metric) or default


# =========================================================
# 1. ROC-AUC Bar Comparison (Primary Chart)
# =========================================================
def plot_roc_auc_comparison(
    cic17_data: dict,
    cic18_data: dict,
    ctu_data:   dict,
    filename:   str,
) -> None:
    x     = np.arange(len(MODELS))
    width = 0.25

    cic17_auc = [get_val(cic17_data, m, "roc_auc") for m in MODELS]
    cic18_auc = [get_val(cic18_data, m, "roc_auc") for m in MODELS]
    ctu_auc   = [get_val(ctu_data,   m, "roc_auc") for m in MODELS]

    fig, ax = plt.subplots(figsize=(13, 6))

    b1 = ax.bar(x - width, cic17_auc, width,
                label="CIC-IDS2017 (Internal)", color=DATASET_COLORS["CIC2017"], alpha=0.85)
    b2 = ax.bar(x,         cic18_auc, width,
                label="CIC-IDS2018 (Cross)",   color=DATASET_COLORS["CIC2018"], alpha=0.85)
    b3 = ax.bar(x + width, ctu_auc,   width,
                label="CTU-13 (Cross)",         color=DATASET_COLORS["CTU-13"],  alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8
            )

    ax.axhline(0.5, color="red",  linestyle="--", linewidth=1.2, label="Random (0.5)")
    ax.axhline(0.7, color="gray", linestyle=":",  linewidth=1.0, label="Good (0.7)")
    ax.axhline(0.9, color="gray", linestyle="-.", linewidth=1.0, label="Excellent (0.9)")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        f"ROC-AUC Comparison (Primary Metric) — [{AUGMENT_LABEL}]\n"
        f"CIC-IDS2017 (Internal) vs CIC-IDS2018 / CTU-13 (Cross-Dataset)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 2. F1 Bar Comparison (Secondary Chart, Youden's J)
# =========================================================
def plot_f1_comparison(
    cic17_data: dict,
    cic18_data: dict,
    ctu_data:   dict,
    filename:   str,
) -> None:
    x     = np.arange(len(MODELS))
    width = 0.25

    cic17_f1 = [get_val(cic17_data, m, "f1") for m in MODELS]
    cic18_f1 = [get_val(cic18_data, m, "f1") for m in MODELS]
    ctu_f1   = [get_val(ctu_data,   m, "f1") for m in MODELS]

    fig, ax = plt.subplots(figsize=(13, 6))

    b1 = ax.bar(x - width, cic17_f1, width,
                label="CIC-IDS2017 (Internal)", color=DATASET_COLORS["CIC2017"], alpha=0.85)
    b2 = ax.bar(x,         cic18_f1, width,
                label="CIC-IDS2018 (Cross)",   color=DATASET_COLORS["CIC2018"], alpha=0.85)
    b3 = ax.bar(x + width, ctu_f1,   width,
                label="CTU-13 (Cross)",         color=DATASET_COLORS["CTU-13"],  alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1-Score")
    ax.set_title(
        f"F1-Score Comparison (Secondary Metric, Youden's J Threshold) — [{AUGMENT_LABEL}]"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 3. ROC-AUC Heatmap
# =========================================================
def plot_roc_auc_heatmap(
    cic17_data: dict,
    cic18_data: dict,
    ctu_data:   dict,
    filename:   str,
) -> None:
    datasets = ["CIC-IDS2017\n(Internal)", "CIC-IDS2018\n(Cross)", "CTU-13\n(Cross)"]
    matrix   = np.array([
        [get_val(d, m, "roc_auc") for m in MODELS]
        for d in [cic17_data, cic18_data, ctu_data]
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)

    for i in range(len(datasets)):
        for j in range(len(MODELS)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="black" if v < 0.75 else "white")

    plt.colorbar(im, ax=ax, label="ROC-AUC")
    ax.set_title(
        f"ROC-AUC Heatmap [{AUGMENT_LABEL}]  "
        f"(Green=Excellent / Red=Random-level)"
    )
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 4. ROC-AUC Drop Line Chart (Domain Shift Visualization)
# =========================================================
def plot_roc_auc_drop(
    cic17_data: dict,
    cic18_data: dict,
    ctu_data:   dict,
    filename:   str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = ["CIC-IDS2017\n(Internal)", "CIC-IDS2018\n(Cross)", "CTU-13\n(Cross)"]
    x_pos    = [0, 1, 2]

    for model in MODELS:
        c17 = get_val(cic17_data, model, "roc_auc")
        c18 = get_val(cic18_data, model, "roc_auc")
        ctu = get_val(ctu_data,   model, "roc_auc")

        vals = [c17, c18, ctu]
        ax.plot(x_pos, vals, marker="o", linewidth=2,
                label=model, color=COLORS[model])
        for xi, v in zip(x_pos, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8,
                    color=COLORS[model])

    ax.axhline(0.5, color="red",  linestyle="--", linewidth=1.2, alpha=0.7,
               label="Random baseline (0.5)")
    ax.axhline(0.7, color="gray", linestyle=":",  linewidth=1.0, alpha=0.7,
               label="Good threshold (0.7)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        f"ROC-AUC Degradation: Internal -> Cross-Dataset [{AUGMENT_LABEL}]\n"
        f"(Domain Shift Impact Visualization)"
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 5. Confusion Matrix
# =========================================================
def plot_confusion_matrices(data: dict, title_prefix: str, filename: str) -> None:
    models = [m for m in MODELS if m in data]
    n      = len(models)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        cm    = np.array(data[model]["confusion_matrix"])
        total = cm.sum()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Botnet"])
        ax.set_yticklabels(["Normal", "Botnet"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{title_prefix}\n{model}")
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                ax.text(j, i, f"{v:,}\n({v / total * 100:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if v > cm.max() * 0.5 else "black",
                        fontweight="bold")

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 6. Recall & Precision Comparison (Supplementary)
# =========================================================
def plot_recall_precision(
    cic17_data: dict,
    cic18_data: dict,
    ctu_data:   dict,
    filename:   str,
) -> None:
    x     = np.arange(len(MODELS))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in zip(
        axes,
        ["recall",    "precision"],
        ["Recall (Botnet Detection Rate)", "Precision"],
        [f"Recall [{AUGMENT_LABEL}]",      f"Precision [{AUGMENT_LABEL}]"],
    ):
        c17 = [get_val(cic17_data, m, metric) for m in MODELS]
        c18 = [get_val(cic18_data, m, metric) for m in MODELS]
        ctu = [get_val(ctu_data,   m, metric) for m in MODELS]

        ax.bar(x - width, c17, width, label="CIC-IDS2017",
               color=DATASET_COLORS["CIC2017"], alpha=0.85)
        ax.bar(x,         c18, width, label="CIC-IDS2018",
               color=DATASET_COLORS["CIC2018"], alpha=0.85)
        ax.bar(x + width, ctu, width, label="CTU-13",
               color=DATASET_COLORS["CTU-13"],  alpha=0.85)

        for vals, offset in [(c17, -width), (c18, 0), (ctu, width)]:
            for i, v in enumerate(vals):
                ax.text(i + offset, v + 0.01, f"{v:.2f}",
                        ha="center", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        f"Botnet Detection: Recall & Precision — [{AUGMENT_LABEL}]",
        fontsize=12
    )
    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# Summary print
# =========================================================
def print_summary(cic17_data: dict, cic18_data: dict, ctu_data: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  ROC-AUC Summary [{AUGMENT_LABEL}]")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'CIC2017':>9} {'CIC2018':>9} {'CTU-13':>9}")
    print("-" * 42)
    for model in MODELS:
        c17 = get_val(cic17_data, model, "roc_auc")
        c18 = get_val(cic18_data, model, "roc_auc")
        ctu = get_val(ctu_data,   model, "roc_auc")
        c18_s = f"{c18:.4f}" if c18 else "  N/A"
        print(f"{model:<12} {c17:>9.4f} {c18_s:>9} {ctu:>9.4f}")
    print("=" * 60)


# =========================================================
# Main
# =========================================================
def main():
    print(f"=== visualize.py  [augment={AUGMENT}] ===")
    print(f"[LOAD] {EVAL_PATH}")
    print(f"[SAVE] {FIGURE_DIR}")

    results    = load_results()
    cic17_data = extract_stage(results, "stage1_cic_test")
    cic18_data = extract_stage(results, "stage2_cic2018_cross")
    ctu_data   = extract_stage(results, "stage3_ctu13_cross")

    print_summary(cic17_data, cic18_data, ctu_data)

    # 1. ROC-AUC bar chart (primary)
    plot_roc_auc_comparison(
        cic17_data, cic18_data, ctu_data,
        filename="01_roc_auc_comparison.png",
    )

    # 2. ROC-AUC heatmap
    plot_roc_auc_heatmap(
        cic17_data, cic18_data, ctu_data,
        filename="02_roc_auc_heatmap.png",
    )

    # 3. ROC-AUC drop line chart
    plot_roc_auc_drop(
        cic17_data, cic18_data, ctu_data,
        filename="03_roc_auc_drop.png",
    )

    # 4. F1 bar chart (secondary)
    plot_f1_comparison(
        cic17_data, cic18_data, ctu_data,
        filename="04_f1_comparison.png",
    )

    # 5. Recall & Precision (supplementary)
    plot_recall_precision(
        cic17_data, cic18_data, ctu_data,
        filename="05_recall_precision.png",
    )

    # 6. Confusion Matrix — CIC-IDS2017
    plot_confusion_matrices(
        cic17_data,
        title_prefix=f"CIC-IDS2017 [{AUGMENT_LABEL}]",
        filename="06_cm_cic17.png",
    )

    # 7. Confusion Matrix — CTU-13
    plot_confusion_matrices(
        ctu_data,
        title_prefix=f"CTU-13 [{AUGMENT_LABEL}]",
        filename="07_cm_ctu13.png",
    )

    print(f"\n[Done] {FIGURE_DIR}")
    print("  01_roc_auc_comparison.png  <- Primary chart")
    print("  02_roc_auc_heatmap.png     <- ROC-AUC heatmap")
    print("  03_roc_auc_drop.png        <- Domain shift visualization")
    print("  04_f1_comparison.png       <- F1 secondary chart")
    print("  05_recall_precision.png    <- Recall & Precision")
    print("  06_cm_cic17.png            <- Confusion Matrix")
    print("  07_cm_ctu13.png            <- Confusion Matrix")


if __name__ == "__main__":
    main()