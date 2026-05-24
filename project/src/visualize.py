"""
visualize.py

F1 / Recall 중심 시각화 (주 지표)
보조: ROC-AUC

각 데이터셋의 eval_results.json을 개별 로드하여 통합 시각화

Output: artifacts/figures_{augment}/

Usage:
  python visualize.py
  python visualize.py --augment smote
  python visualize.py --augment gan
  python visualize.py --augment wcgan_gp
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 인자 파싱
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
# 경로 설정
# =========================================================
_SRC_DIR   = Path(__file__).resolve().parent
_PROJECT   = _SRC_DIR.parent
_ROOT      = _PROJECT.parent

_SUFFIX    = f"_{AUGMENT}" if AUGMENT != "none" else ""

# 데이터셋별 결과 파일 경로
EVAL_PATHS = {
    "cicids2017": _ROOT / "artifacts" / f"results_cicids2017{_SUFFIX}" / "eval_results.json",
    "cicids2018": _ROOT / "artifacts" / f"results_cicids2018{_SUFFIX}" / "eval_results.json",
    "ctu13":      _ROOT / "artifacts" / f"results_ctu13{_SUFFIX}"      / "eval_results.json",
}

# 시각화 저장 경로
FIGURE_DIR = _ROOT / "artifacts" / f"figures{_SUFFIX}"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 스타일
# =========================================================
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})

MODELS = ["RF", "XGBoost", "CNN-LSTM", "GRU", "CNN-GRU"]

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
    "cicids2017": "#4878D0",
    "cicids2018": "#EE854A",
    "ctu13":      "#6ACC65",
}

DATASET_LABELS = {
    "cicids2017": "CIC-IDS2017",
    "cicids2018": "CSE-CIC-IDS2018",
    "ctu13":      "CTU-13 Sc.9",
}


# =========================================================
# 데이터 로드
# =========================================================
def load_dataset_results(dataset: str) -> dict:
    """
    각 데이터셋의 eval_results.json 로드
    없으면 빈 딕셔너리 반환
    """
    path = EVAL_PATHS[dataset]
    if not path.exists():
        print(f"[SKIP] {dataset} 결과 없음: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # evaluate.py 출력 형식: {"test_results": {"rf": {...}, ...}}
    raw = data.get("test_results", {})

    name_map = {
        "rf": "RF", "xgb": "XGBoost",
        "cnn_lstm": "CNN-LSTM", "gru": "GRU", "cnn_gru": "CNN-GRU",
    }
    return {name_map[k]: v for k, v in raw.items() if k in name_map}


def get_val(data: dict, model: str, metric: str, default: float = 0.0) -> float:
    v = data.get(model, {}).get(metric)
    return v if v is not None else default


# =========================================================
# 1. F1 Bar Comparison (★ 주 지표 ①)
# =========================================================
def plot_f1_comparison(results: dict, filename: str) -> None:
    x     = np.arange(len(MODELS))
    width = 0.25
    datasets = ["cicids2017", "cicids2018", "ctu13"]
    offsets  = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(13, 6))

    for ds, offset in zip(datasets, offsets):
        vals = [get_val(results[ds], m, "f1") for m in MODELS]
        bars = ax.bar(x + offset, vals, width,
                      label=DATASET_LABELS[ds],
                      color=DATASET_COLORS[ds], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1-Score")
    ax.set_title(f"★ F1-Score Comparison (Primary Metric) — [{AUGMENT_LABEL}]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {FIGURE_DIR / filename}")


# =========================================================
# 2. Recall Bar Comparison (★ 주 지표 ②)
# =========================================================
def plot_recall_comparison(results: dict, filename: str) -> None:
    x     = np.arange(len(MODELS))
    width = 0.25
    datasets = ["cicids2017", "cicids2018", "ctu13"]
    offsets  = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(13, 6))

    for ds, offset in zip(datasets, offsets):
        vals = [get_val(results[ds], m, "recall") for m in MODELS]
        bars = ax.bar(x + offset, vals, width,
                      label=DATASET_LABELS[ds],
                      color=DATASET_COLORS[ds], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.8, color="orange", linestyle="--", linewidth=1.2,
               label="Target Recall ≥ 0.8")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall (Botnet Detection Rate)")
    ax.set_title(f"★ Recall Comparison (Primary Metric) — [{AUGMENT_LABEL}]")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {FIGURE_DIR / filename}")


# =========================================================
# 3. F1 / Recall / Precision 3-panel
# =========================================================
def plot_metrics_panel(results: dict, filename: str) -> None:
    x        = np.arange(len(MODELS))
    width    = 0.25
    datasets = ["cicids2017", "cicids2018", "ctu13"]
    offsets  = [-width, 0, width]

    metrics_info = [
        ("f1",        "F1-Score",  "★ Primary"),
        ("recall",    "Recall",    "★ Primary"),
        ("precision", "Precision", "Secondary"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (metric, ylabel, tag) in zip(axes, metrics_info):
        for ds, offset in zip(datasets, offsets):
            vals = [get_val(results[ds], m, metric) for m in MODELS]
            ax.bar(x + offset, vals, width,
                   label=DATASET_LABELS[ds],
                   color=DATASET_COLORS[ds], alpha=0.85)
            for i, v in enumerate(vals):
                ax.text(i + offset, v + 0.01, f"{v:.2f}",
                        ha="center", fontsize=7)

        if metric == "recall":
            ax.axhline(0.8, color="orange", linestyle="--",
                       linewidth=1.0, label="Target ≥ 0.8")

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} [{tag}]")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Detection Metrics — [{AUGMENT_LABEL}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {FIGURE_DIR / filename}")


# =========================================================
# 4. ROC-AUC Bar (보조 지표)
# =========================================================
def plot_roc_auc_comparison(results: dict, filename: str) -> None:
    x        = np.arange(len(MODELS))
    width    = 0.25
    datasets = ["cicids2017", "cicids2018", "ctu13"]
    offsets  = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(13, 6))

    for ds, offset in zip(datasets, offsets):
        vals = [get_val(results[ds], m, "roc_auc") for m in MODELS]
        bars = ax.bar(x + offset, vals, width,
                      label=DATASET_LABELS[ds],
                      color=DATASET_COLORS[ds], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="red",  linestyle="--", linewidth=1.2, label="Random (0.5)")
    ax.axhline(0.7, color="gray", linestyle=":",  linewidth=1.0, label="Good (0.7)")
    ax.axhline(0.9, color="gray", linestyle="-.", linewidth=1.0, label="Excellent (0.9)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"ROC-AUC Comparison (Secondary Metric) — [{AUGMENT_LABEL}]")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {FIGURE_DIR / filename}")


# =========================================================
# 5. F1 / Recall Heatmap (★ 주 지표)
# =========================================================
def plot_f1_recall_heatmap(results: dict, filename: str) -> None:
    ds_labels = ["CIC-IDS2017", "CSE-CIC-IDS2018", "CTU-13 Sc.9"]
    ds_keys   = ["cicids2017",  "cicids2018",       "ctu13"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, metric, title in zip(
        axes,
        ["f1",              "recall"],
        ["F1-Score (★)",    "Recall (★)"],
    ):
        matrix = np.array([
            [get_val(results[ds], m, metric) for m in MODELS]
            for ds in ds_keys
        ])
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.set_yticks(range(len(ds_labels)))
        ax.set_yticklabels(ds_labels, fontsize=10)
        for i in range(len(ds_labels)):
            for j in range(len(MODELS)):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if v > 0.6 else "black")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

    plt.suptitle(f"F1 / Recall Heatmap [{AUGMENT_LABEL}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {FIGURE_DIR / filename}")


# =========================================================
# 6. ROC-AUC Heatmap (보조)
# =========================================================
def plot_roc_auc_heatmap(results: dict, filename: str) -> None:
    ds_labels = ["CIC-IDS2017", "CSE-CIC-IDS2018", "CTU-13 Sc.9"]
    ds_keys   = ["cicids2017",  "cicids2018",       "ctu13"]

    matrix = np.array([
        [get_val(results[ds], m, "roc_auc") for m in MODELS]
        for ds in ds_keys
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_yticks(range(len(ds_labels)))
    ax.set_yticklabels(ds_labels)
    for i in range(len(ds_labels)):
        for j in range(len(MODELS)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if v < 0.75 else "white")
    plt.colorbar(im, ax=ax, label="ROC-AUC")
    ax.set_title(f"ROC-AUC Heatmap (Secondary) [{AUGMENT_LABEL}]")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {FIGURE_DIR / filename}")


# =========================================================
# 7. Confusion Matrix
# =========================================================
def plot_confusion_matrices(data: dict, title_prefix: str, filename: str) -> None:
    if not data:
        return
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

        f1  = data[model].get("f1",     0)
        rec = data[model].get("recall", 0)
        ax.set_title(f"{title_prefix}\n{model}\nF1={f1:.3f} / Rec={rec:.3f}")

        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                ax.text(j, i, f"{v:,}\n({v / total * 100:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if v > cm.max() * 0.5 else "black",
                        fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {FIGURE_DIR / filename}")


# =========================================================
# 요약 출력
# =========================================================
def print_summary(results: dict) -> None:
    print(f"\n{'='*75}")
    print(f"  ★ 주 지표 요약 — F1 / Recall  [{AUGMENT_LABEL}]")
    print(f"{'='*75}")
    print(
        f"{'Model':<12} {'CIC2017 F1':>11} {'CIC2017 Rec':>12} "
        f"{'CIC2018 F1':>11} {'CIC2018 Rec':>12} "
        f"{'CTU-13 F1':>10} {'CTU-13 Rec':>11}"
    )
    print("-" * 73)
    for model in MODELS:
        def _s(ds, m): return f"{get_val(results[ds], model, m):.4f}"
        print(
            f"{model:<12} "
            f"{_s('cicids2017', 'f1'):>11} {_s('cicids2017', 'recall'):>12} "
            f"{_s('cicids2018', 'f1'):>11} {_s('cicids2018', 'recall'):>12} "
            f"{_s('ctu13',      'f1'):>10} {_s('ctu13',      'recall'):>11}"
        )
    print("=" * 75)

    print(f"\n{'='*55}")
    print(f"  보조 지표 — ROC-AUC  [{AUGMENT_LABEL}]")
    print(f"{'='*55}")
    print(f"{'Model':<12} {'CIC2017':>9} {'CIC2018':>9} {'CTU-13':>9}")
    print("-" * 42)
    for model in MODELS:
        c17 = get_val(results["cicids2017"], model, "roc_auc")
        c18 = get_val(results["cicids2018"], model, "roc_auc")
        ctu = get_val(results["ctu13"],      model, "roc_auc")
        print(f"{model:<12} {c17:>9.4f} {c18:>9.4f} {ctu:>9.4f}")
    print("=" * 55)


# =========================================================
# main
# =========================================================
def main():
    print(f"=== visualize.py  [augment={AUGMENT}] ===")
    print(f"[SAVE] {FIGURE_DIR}")

    # 3개 데이터셋 결과 로드
    results = {ds: load_dataset_results(ds) for ds in ["cicids2017", "cicids2018", "ctu13"]}

    loaded = [ds for ds, d in results.items() if d]
    missing = [ds for ds, d in results.items() if not d]
    print(f"[LOAD] 로드 성공: {loaded}")
    if missing:
        print(f"[SKIP] 결과 없음: {missing}")

    if not any(results.values()):
        print("[ERROR] 로드된 결과가 없습니다. evaluate.py 먼저 실행하세요.")
        return

    print_summary(results)

    # ── 주 지표 ───────────────────────────────────────────
    plot_f1_comparison(results,    "01_f1_comparison.png")
    plot_recall_comparison(results,"02_recall_comparison.png")
    plot_metrics_panel(results,    "03_metrics_panel.png")
    plot_f1_recall_heatmap(results,"04_f1_recall_heatmap.png")

    # ── 보조 지표 ─────────────────────────────────────────
    plot_roc_auc_comparison(results, "05_roc_auc_comparison.png")
    plot_roc_auc_heatmap(results,    "06_roc_auc_heatmap.png")

    # ── Confusion Matrix ──────────────────────────────────
    plot_confusion_matrices(
        results["cicids2017"],
        title_prefix=f"CIC-IDS2017 [{AUGMENT_LABEL}]",
        filename="07_cm_cic2017.png",
    )
    plot_confusion_matrices(
        results["cicids2018"],
        title_prefix=f"CSE-CIC-IDS2018 [{AUGMENT_LABEL}]",
        filename="08_cm_cic2018.png",
    )
    plot_confusion_matrices(
        results["ctu13"],
        title_prefix=f"CTU-13 Sc.9 [{AUGMENT_LABEL}]",
        filename="09_cm_ctu13.png",
    )

    print(f"\n[Done] {FIGURE_DIR}")
    print("  ★ 주 지표")
    print("    01_f1_comparison.png")
    print("    02_recall_comparison.png")
    print("    03_metrics_panel.png")
    print("    04_f1_recall_heatmap.png")
    print("  보조 지표")
    print("    05_roc_auc_comparison.png")
    print("    06_roc_auc_heatmap.png")
    print("  Confusion Matrix")
    print("    07_cm_cic2017.png")
    print("    08_cm_cic2018.png")
    print("    09_cm_ctu13.png")


if __name__ == "__main__":
    main()