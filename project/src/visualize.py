"""
visualize.py

eval_results.json을 읽어 성능 비교 시각화

1단계: CIC-IDS2017 내부 test 성능
2단계: CTU-13 시나리오 9 교차검증 성능 (Adaptive threshold, Safety 2025)
비교: Recall(Bot 탐지율) 중심

저장 위치: artifacts/results/figures/
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR   = Path(__file__).resolve().parent
_PROJECT   = _SRC_DIR.parent
_ROOT      = _PROJECT.parent

RESULT_DIR = _ROOT / "artifacts" / "results"
FIGURE_DIR = RESULT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PATH  = RESULT_DIR / "eval_results.json"


# =========================================================
# 스타일 설정
# =========================================================
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})

COLORS = {
    "RF":       "#4C72B0",
    "XGBoost":  "#DD8452",
    "CNN-LSTM": "#55A868",
    "GRU":      "#C44E52",
    "CNN-GRU":  "#8172B2",
}

METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1-Score",
    "roc_auc":   "ROC-AUC",
}


# =========================================================
# 데이터 로드 및 파싱
# =========================================================
def load_results() -> dict:
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_models(results: dict, stage_key: str) -> dict:
    stage = results[stage_key]
    name_map = {
        "rf":       "RF",
        "xgb":      "XGBoost",
        "cnn_lstm": "CNN-LSTM",
        "gru":      "GRU",
        "cnn_gru":  "CNN-GRU",
    }
    return {name_map[k]: v for k, v in stage.items() if k in name_map}


# =========================================================
# 1. 막대 그래프 — 단계별 지표 비교
# =========================================================
def plot_bar_comparison(
    data:     dict,
    title:    str,
    filename: str,
    metrics:  list[str] = None,
) -> None:
    if metrics is None:
        metrics = ["precision", "recall", "f1", "roc_auc"]

    models      = list(data.keys())
    n_models    = len(models)
    x           = np.arange(len(metrics))
    group_width = 0.8
    width       = group_width / n_models
    offsets     = np.linspace(
        -group_width / 2 + width / 2,
        group_width / 2 - width / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, model in enumerate(models):
        values = [data[model].get(m, 0) or 0 for m in metrics]
        bars   = ax.bar(
            x + offsets[i], values,
            width=width,
            color=COLORS.get(model, "#999999"),
            label=model,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=6.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 2. 히트맵 — CIC-IDS2017 vs CTU-13 비교
# =========================================================
def plot_heatmap_comparison(
    cic17_data: dict,
    ctu_data:   dict,
    filename:   str,
) -> None:
    models  = list(cic17_data.keys())
    metrics = ["precision", "recall", "f1", "roc_auc"]

    vals17  = np.array([[cic17_data[m].get(mt, 0) or 0 for mt in metrics] for m in models])
    vals_ctu = np.array([[ctu_data[m].get(mt, 0) or 0 for mt in metrics] for m in models])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, vals, title in zip(
        axes,
        [vals17, vals_ctu],
        [
            "Stage 1: CIC-IDS2017 Internal Test",
            "Stage 2: CTU-13 Cross-Dataset (Adaptive)",
        ],
    ):
        im = ax.imshow(vals, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title(title)

        for i in range(len(models)):
            for j in range(len(metrics)):
                v = vals[i, j]
                ax.text(
                    j, i, f"{v:.3f}",
                    ha="center", va="center",
                    fontsize=10,
                    color="black" if v < 0.6 else "white",
                    fontweight="bold",
                )

    plt.colorbar(im, ax=axes[-1], label="Score")
    plt.suptitle(
        "Model Performance: CIC-IDS2017 vs CTU-13 (Safety 2025)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 3. Recall / F1 중심 비교 — 봇넷 탐지율 강조
# =========================================================
def plot_recall_focus(
    cic17_data: dict,
    ctu_data:   dict,
    filename:   str,
) -> None:
    models = list(cic17_data.keys())

    cic17_recall = [cic17_data[m].get("recall", 0) or 0 for m in models]
    ctu_recall   = [ctu_data[m].get("recall",   0) or 0 for m in models]
    cic17_f1     = [cic17_data[m].get("f1", 0) or 0 for m in models]
    ctu_f1       = [ctu_data[m].get("f1",   0) or 0 for m in models]

    x     = np.arange(len(models))
    width = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cic17_vals, ctu_vals, ylabel, title in zip(
        axes,
        [cic17_recall, cic17_f1],
        [ctu_recall,   ctu_f1],
        ["Recall (Botnet Detection Rate)", "F1-Score"],
        ["Recall: CIC-IDS2017 vs CTU-13", "F1-Score: CIC-IDS2017 vs CTU-13"],
    ):
        ax.bar(x - width / 2, cic17_vals, width,
               label="CIC-IDS2017 (Internal)", color="#4878D0", alpha=0.85)
        ax.bar(x + width / 2, ctu_vals,   width,
               label="CTU-13 (Cross, Adaptive)", color="#6ACC65", alpha=0.85)

        for i, (c17, ctu) in enumerate(zip(cic17_vals, ctu_vals)):
            ax.text(i - width / 2, c17 + 0.02, f"{c17:.3f}", ha="center", fontsize=9)
            ax.text(i + width / 2, ctu + 0.02, f"{ctu:.3f}", ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        "Botnet Detection: CIC-IDS2017 (Internal) vs CTU-13 (Cross-Dataset, Safety 2025)",
        fontsize=12,
    )
    plt.tight_layout()

    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 4. Confusion Matrix
# =========================================================
def plot_confusion_matrices(
    data:         dict,
    title_prefix: str,
    filename:     str,
) -> None:
    models = list(data.keys())
    n      = len(models)

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
                ax.text(
                    j, i,
                    f"{v:,}\n({v / total * 100:.1f}%)",
                    ha="center", va="center",
                    fontsize=10,
                    color="white" if v > cm.max() * 0.5 else "black",
                    fontweight="bold",
                )

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 5. Threshold 비교 — Adaptive threshold 값 시각화
# =========================================================
def plot_threshold_comparison(
    cic17_data: dict,
    ctu_data:   dict,
    filename:   str,
) -> None:
    models     = list(cic17_data.keys())
    cic17_thr  = [cic17_data[m].get("threshold", 0.5) or 0.5 for m in models]
    ctu_thr    = [ctu_data[m].get("threshold",   0.5) or 0.5 for m in models]

    x     = np.arange(len(models))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, cic17_thr, width,
           label="CIC-IDS2017 (val threshold)", color="#4878D0", alpha=0.85)
    ax.bar(x + width / 2, ctu_thr,   width,
           label="CTU-13 (adaptive threshold)", color="#6ACC65", alpha=0.85)

    for i, (c17, ctu) in enumerate(zip(cic17_thr, ctu_thr)):
        ax.text(i - width / 2, c17 + 0.01, f"{c17:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, ctu + 0.01, f"{ctu:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold Comparison: CIC-IDS2017 val vs CTU-13 Adaptive")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="default (0.5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 요약 테이블 콘솔 출력
# =========================================================
def print_summary_table(cic17_data: dict, ctu_data: dict) -> None:
    models = list(cic17_data.keys())

    print("\n" + "=" * 100)
    print("  Performance Summary: CIC-IDS2017 (Internal) vs CTU-13 (Cross-Dataset, Adaptive)")
    print("=" * 100)
    print(
        f"{'Model':<12} {'Dataset':<16} {'Threshold':>10} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7}"
    )
    print("-" * 100)

    for model in models:
        for label, data in [("CIC-IDS2017", cic17_data), ("CTU-13", ctu_data)]:
            d   = data[model]
            cm  = np.array(d["confusion_matrix"])
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            roc = f"{d['roc_auc']:.4f}" if d.get("roc_auc") is not None else "  None"
            thr = f"{d.get('threshold', '-'):.4f}" if isinstance(d.get("threshold"), float) else "  -"
            print(
                f"{model:<12} {label:<16} "
                f"{thr:>10} "
                f"{d['precision']:>10.4f} "
                f"{d['recall']:>8.4f} "
                f"{d['f1']:>8.4f} "
                f"{roc:>9} "
                f"{tp:>7,} {fp:>7,} {fn:>7,} {tn:>7,}"
            )
        print("-" * 100)

    print("=" * 100)
    print("  TP=봇넷 정탐 / FP=정상 오탐 / FN=봇넷 미탐(핵심) / TN=정상 정탐")
    print("  CTU-13: Adaptive threshold (Safety 2025) — target 정보 사용")


# =========================================================
# 메인
# =========================================================
def main():
    print("=== visualize.py ===")
    print(f"[LOAD] {EVAL_PATH}")

    results    = load_results()
    cic17_data = extract_models(results, "stage1_cic_test")
    ctu_data   = extract_models(results, "stage2_ctu13_cross")

    # 콘솔 요약 출력
    print_summary_table(cic17_data, ctu_data)

    # 1. CIC-IDS2017 막대 그래프
    plot_bar_comparison(
        cic17_data,
        title="Stage 1: CIC-IDS2017 Internal Test Performance",
        filename="01_cic17_bar.png",
    )

    # 2. CTU-13 교차검증 막대 그래프
    plot_bar_comparison(
        ctu_data,
        title="Stage 2: CTU-13 Cross-Dataset Performance (Adaptive, Safety 2025)",
        filename="02_ctu13_bar.png",
    )

    # 3. 히트맵 비교
    plot_heatmap_comparison(
        cic17_data, ctu_data,
        filename="03_heatmap.png",
    )

    # 4. Recall / F1 비교
    plot_recall_focus(
        cic17_data, ctu_data,
        filename="04_recall_f1.png",
    )

    # 5. Confusion Matrix — CIC-IDS2017
    plot_confusion_matrices(
        cic17_data,
        title_prefix="CIC-IDS2017 test",
        filename="05_cm_cic17.png",
    )

    # 6. Confusion Matrix — CTU-13
    plot_confusion_matrices(
        ctu_data,
        title_prefix="CTU-13 cross (Adaptive)",
        filename="06_cm_ctu13.png",
    )

    # 7. Threshold 비교
    plot_threshold_comparison(
        cic17_data, ctu_data,
        filename="07_threshold_compare.png",
    )

    print(f"\n[Done] All figures saved: {FIGURE_DIR}")
    print("  01_cic17_bar.png          — CIC-IDS2017 내부 test 성능")
    print("  02_ctu13_bar.png          — CTU-13 교차검증 성능 (Adaptive)")
    print("  03_heatmap.png            — 두 데이터셋 히트맵 비교")
    print("  04_recall_f1.png          — Recall / F1 비교 (Bot 탐지율)")
    print("  05_cm_cic17.png           — CIC-IDS2017 Confusion Matrix")
    print("  06_cm_ctu13.png           — CTU-13 Confusion Matrix")
    print("  07_threshold_compare.png  — Threshold 비교 (val vs adaptive)")


if __name__ == "__main__":
    main()