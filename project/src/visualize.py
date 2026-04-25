"""
visualize.py

목적:
- eval_results.json을 읽어 성능 비교 시각화
- 1단계(CIC 단독) / 2단계(CTU 교차검증) 비교표 및 차트 저장

저장 위치: artifacts/results/figures/
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR    = Path(__file__).resolve().parent
_PROJECT    = _SRC_DIR.parent
_ROOT       = _PROJECT.parent

RESULT_DIR  = _ROOT / "artifacts" / "results"
FIGURE_DIR  = RESULT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PATH   = RESULT_DIR / "eval_results.json"


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
}

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1-Score",
    "roc_auc":   "ROC-AUC",
}


# =========================================================
# 데이터 파싱
# =========================================================
def load_results() -> dict:
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_stage(results: dict, stage_key: str) -> dict:
    """
    반환 형태:
    {
      "RF":       {"accuracy": ..., "precision": ..., ...},
      "XGBoost":  {...},
      "CNN-LSTM": {...},
    }
    """
    stage = results[stage_key]
    return {
        "RF":       stage["rf"],
        "XGBoost":  stage["xgb"],
        "CNN-LSTM": stage["cnn_lstm"],
    }


def extract_ctu(results: dict) -> dict:
    """CTU는 scenario9 하나"""
    stage = results["stage2_ctu_cross"]["scenario9"]
    return {
        "RF":       stage["rf"],
        "XGBoost":  stage["xgb"],
        "CNN-LSTM": stage["cnn_lstm"],
    }


# =========================================================
# 1. 막대 그래프 — 단계별 지표 비교
# =========================================================
def plot_bar_comparison(
    data: dict,
    title: str,
    filename: str,
    metrics: list[str] = None,
) -> None:
    if metrics is None:
        metrics = ["precision", "recall", "f1", "roc_auc"]

    models       = list(data.keys())
    n_models     = len(models)
    n_metrics    = len(metrics)
    x            = np.arange(n_metrics)
    width        = 0.22
    offsets      = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        values = [
            data[model].get(m, 0) or 0
            for m in metrics
        ]
        bars = ax.bar(
            x + offsets[i], values,
            width=width,
            color=COLORS[model],
            label=model,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # 값 표시
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, rotation=45,
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
# 2. 히트맵 — CIC vs CTU 비교
# =========================================================
def plot_heatmap_comparison(
    cic_data: dict,
    ctu_data: dict,
    filename: str,
) -> None:
    models  = list(cic_data.keys())
    metrics = ["precision", "recall", "f1", "roc_auc"]

    cic_vals = np.array([
        [cic_data[m].get(mt, 0) or 0 for mt in metrics]
        for m in models
    ])
    ctu_vals = np.array([
        [ctu_data[m].get(mt, 0) or 0 for mt in metrics]
        for m in models
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, vals, title in zip(
        axes,
        [cic_vals, ctu_vals],
        ["Stage 1: CIC-IDS2017 Test", "Stage 2: CTU-13 Cross-Validation (scenario9)"],
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
    plt.suptitle("Model Performance: CIC vs CTU", fontsize=13, y=1.02)
    plt.tight_layout()

    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 3. Recall 중심 비교 — 봇넷 탐지율 강조
# =========================================================
def plot_recall_focus(
    cic_data: dict,
    ctu_data: dict,
    filename: str,
) -> None:
    models = list(cic_data.keys())

    cic_recall = [cic_data[m].get("recall", 0) or 0 for m in models]
    ctu_recall = [ctu_data[m].get("recall", 0) or 0 for m in models]

    cic_f1 = [cic_data[m].get("f1", 0) or 0 for m in models]
    ctu_f1 = [ctu_data[m].get("f1", 0) or 0 for m in models]

    x      = np.arange(len(models))
    width  = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Recall 비교
    ax = axes[0]
    ax.bar(x - width / 2, cic_recall, width, label="CIC",  color="#4878D0", alpha=0.85)
    ax.bar(x + width / 2, ctu_recall, width, label="CTU",  color="#EE854A", alpha=0.85)
    for i, (c, t) in enumerate(zip(cic_recall, ctu_recall)):
        ax.text(i - width / 2, c + 0.02, f"{c:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, t + 0.02, f"{t:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Recall (Botnet Detection Rate)")
    ax.set_title("Recall: CIC vs CTU")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # F1 비교
    ax = axes[1]
    ax.bar(x - width / 2, cic_f1, width, label="CIC",  color="#4878D0", alpha=0.85)
    ax.bar(x + width / 2, ctu_f1, width, label="CTU",  color="#EE854A", alpha=0.85)
    for i, (c, t) in enumerate(zip(cic_f1, ctu_f1)):
        ax.text(i - width / 2, c + 0.02, f"{c:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, t + 0.02, f"{t:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score: CIC vs CTU")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Botnet Detection: CIC vs CTU Cross-Validation", fontsize=13)
    plt.tight_layout()

    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# =========================================================
# 4. Confusion Matrix 시각화
# =========================================================
def plot_confusion_matrices(
    data: dict,
    title_prefix: str,
    filename: str,
) -> None:
    models = list(data.keys())
    n      = len(models)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for ax, model in zip(axes, models):
        cm    = np.array(data[model]["confusion_matrix"])
        total = cm.sum()

        im = ax.imshow(cm, cmap="Blues")

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
                    f"{v:,}\n({v/total*100:.1f}%)",
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
# 7. 확률 분포 시각화 — RF / XGBoost CTU
# ---------------------------------------------------------
# RF/XGBoost가 CTU 봇넷에 대해 낮은 확률값을 출력하는지 확인
# =========================================================
def plot_prob_distribution(prob_path: Path, filename: str) -> None:
    with open(prob_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Probability Distribution on CTU-13 scenario9\n(Why RF/XGBoost Fail to Detect Botnets)", fontsize=13)

    for row, (model_key, model_name) in enumerate([("rf", "RF"), ("xgb", "XGBoost")]):
        d = data[model_key]
        thr_ctu = d["threshold_ctu"]
        thr_cic = d["threshold_cic"]

        bot_probs  = np.array(d["test_botnet"])
        norm_probs = np.array(d["test_normal"])

        # ── 왼쪽: 봇넷/정상 분포 겹쳐서 표시 ────────────
        ax = axes[row][0]
        ax.hist(norm_probs, bins=50, alpha=0.6, color="#4878D0", label="Normal",  density=True)
        ax.hist(bot_probs,  bins=50, alpha=0.6, color="#EE854A", label="Botnet",  density=True)
        ax.axvline(thr_ctu, color="red",    linestyle="--", linewidth=1.5, label=f"CTU thr: {thr_ctu:.2f}")
        ax.axvline(thr_cic, color="purple", linestyle=":",  linewidth=1.5, label=f"CIC thr: {thr_cic:.2f}")
        ax.set_title(f"{model_name} — Botnet vs Normal")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # 통계 표시
        ax.text(0.98, 0.95,
                f"Botnet  mean: {bot_probs.mean():.4f}\n"
                f"Botnet  max:  {bot_probs.max():.4f}\n"
                f"Normal  mean: {norm_probs.mean():.4f}",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

        # ── 오른쪽: 봇넷만 확대해서 표시 ────────────────
        ax = axes[row][1]
        ax.hist(bot_probs, bins=50, color="#EE854A", alpha=0.85, edgecolor="white")
        ax.axvline(thr_ctu, color="red",    linestyle="--", linewidth=1.5, label=f"CTU thr: {thr_ctu:.2f}")
        ax.axvline(thr_cic, color="purple", linestyle=":",  linewidth=1.5, label=f"CIC thr: {thr_cic:.2f}")
        ax.set_title(f"{model_name} — Botnet Probability (Zoom In)")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # threshold 이상 샘플 수 표시
        above_ctu = (bot_probs >= thr_ctu).sum()
        above_cic = (bot_probs >= thr_cic).sum()
        ax.text(0.98, 0.95,
                f"Total botnet: {len(bot_probs):,}\n"
                f"≥ CTU thr ({thr_ctu:.2f}): {above_ctu:,}\n"
                f"≥ CIC thr ({thr_cic:.2f}): {above_cic:,}",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.tight_layout()
    path = FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")
def print_summary_table(cic_data: dict, ctu_data: dict) -> None:
    metrics = ["precision", "recall", "f1", "roc_auc"]
    models  = list(cic_data.keys())

    print("\n" + "="*90)
    print("  Performance Summary")
    print("="*90)
    print(f"{'Model':<12} {'Dataset':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>7}")
    print("-"*90)

    for model in models:
        for label, data in [("CIC", cic_data), ("CTU", ctu_data)]:
            d  = data[model]
            cm = np.array(d["confusion_matrix"])
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            roc = f"{d['roc_auc']:.4f}" if d.get("roc_auc") else "  None"
            print(
                f"{model:<12} {label:<8} "
                f"{d['precision']:>10.4f} "
                f"{d['recall']:>8.4f} "
                f"{d['f1']:>8.4f} "
                f"{roc:>9} "
                f"{tp:>7,} "
                f"{fp:>7,} "
                f"{fn:>7,} "
                f"{tn:>7,}"
            )
        print("-"*90)

    print("="*90)
    print("  TP=봇넷 정탐 / FP=정상 오탐(봇넷으로 잘못 분류) / FN=봇넷 미탐 / TN=정상 정탐")


# =========================================================
# 메인
# =========================================================
def main():
    print("=== visualize.py ===")
    print(f"[LOAD] {EVAL_PATH}")

    results  = load_results()
    cic_data = extract_stage(results, "stage1_cic_test")
    ctu_data = extract_ctu(results)

    # 콘솔 출력
    print_summary_table(cic_data, ctu_data)

    # 1. CIC 단독 평가 막대 그래프
    plot_bar_comparison(
        cic_data,
        title="Stage 1: CIC-IDS2017 Test Performance",
        filename="01_cic_bar.png",
    )

    # 2. CTU 교차검증 막대 그래프
    plot_bar_comparison(
        ctu_data,
        title="Stage 2: CTU-13 Cross-Validation Performance (Scenario9)",
        filename="02_ctu_bar.png",
    )

    # 3. CIC vs CTU 히트맵
    plot_heatmap_comparison(
        cic_data, ctu_data,
        filename="03_heatmap.png",
    )

    # 4. Recall / F1 중심 비교
    plot_recall_focus(
        cic_data, ctu_data,
        filename="04_recall_f1.png",
    )

    # 5. Confusion Matrix — CIC
    plot_confusion_matrices(
        cic_data,
        title_prefix="CIC test",
        filename="05_cm_cic.png",
    )

    # 6. Confusion Matrix — CTU
    plot_confusion_matrices(
        ctu_data,
        title_prefix="CTU scenario9",
        filename="06_cm_ctu.png",
    )

    # 7. 확률 분포 — RF / XGBoost CTU
    prob_path = RESULT_DIR / "prob_dist_scenario9.json"
    if prob_path.exists():
        plot_prob_distribution(prob_path, filename="07_prob_dist.png")
    else:
        print("[SKIP] prob_dist_scenario9.json 없음 → evaluate.py 먼저 실행")

    print(f"\n[Done] All figures saved: {FIGURE_DIR}")
    print("  01_cic_bar.png      — CIC test bar chart")
    print("  02_ctu_bar.png      — CTU cross-validation bar chart")
    print("  03_heatmap.png      — CIC vs CTU heatmap")
    print("  04_recall_f1.png    — Recall / F1 comparison")
    print("  05_cm_cic.png       — CIC Confusion Matrix")
    print("  06_cm_ctu.png       — CTU Confusion Matrix")
    print("  07_prob_dist.png    — RF/XGBoost 확률 분포 (왜 0이 나오는가)")


if __name__ == "__main__":
    main()