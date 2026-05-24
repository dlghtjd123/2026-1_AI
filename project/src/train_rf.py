import json
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_utils import augment_train_fold
from debug_utils import debug_fold


# =========================================================
# 인자 파싱
# =========================================================
_parser = argparse.ArgumentParser()
_parser.add_argument("--augment",  type=str, default="none",
                     choices=["none", "smote", "gan", "wgan_gp", "wcgan_gp"])
_parser.add_argument("--dataset",  type=str, default="cicids2017",
                     choices=["cicids2017", "cicids2018", "ctu13"])
_parser.add_argument("--n_folds",  type=int, default=5)
_parser.add_argument("--debug",    action="store_true",
                     help="디버그 모드: 원인 분석 로그 출력 (--augment 사용 시 권장)")
_parser.add_argument("--target_bot_ratio", type=float, default=0.05,
                     help="Benign 축소 후 목표 봇넷 비율 (기본값: 0.05=95:5, 0=전체 사용)")
AUGMENT          = _parser.parse_args().augment
DATASET          = _parser.parse_args().dataset
N_FOLDS          = _parser.parse_args().n_folds
DEBUG            = _parser.parse_args().debug
TARGET_BOT_RATIO = _parser.parse_args().target_bot_ratio


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent
_ROOT     = _PROJECT.parent

DATA_DIR   = _PROJECT / "data" / "processed" / DATASET / "flat"
DATA_ROOT  = _PROJECT / "data" / "processed"
_MODEL_SUFFIX = f"_{AUGMENT}" if AUGMENT != "none" else ""
MODEL_DIR  = _ROOT / "artifacts" / f"models_{DATASET}{_MODEL_SUFFIX}" / "rf"
RESULT_DIR = _ROOT / "artifacts" / f"results_{DATASET}{_MODEL_SUFFIX}"


# =========================================================
# 유틸
# =========================================================
def load_data(data_dir: Path):
    X = np.load(data_dir / "X_trainval.npy")
    y = np.load(data_dir / "y_trainval.npy").astype(int)
    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0, output_dict=True
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def print_fold_summary(fold_results: list[dict]) -> dict:
    keys = ["f1", "recall", "precision", "roc_auc", "accuracy"]
    summary = {}

    print(f"\n{'='*60}")
    print(f"  RF  {N_FOLDS}-Fold CV 결과 요약  [{DATASET} / {AUGMENT}]")
    print(f"  ★ 주 지표: F1, Recall  /  보조: ROC-AUC")
    print(f"{'='*60}")
    print(f"{'지표':<14} {'평균':>8} {'표준편차':>10} {'최소':>8} {'최대':>8}")
    print("-" * 52)

    for key in keys:
        vals = [r[key] for r in fold_results if r.get(key) is not None]
        if not vals:
            continue
        mean, std = float(np.mean(vals)), float(np.std(vals))
        summary[key] = {"mean": mean, "std": std,
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
        marker = " ★" if key in ("f1", "recall") else ""
        print(f"{key:<14} {mean:>8.4f} {std:>10.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}{marker}")

    print("=" * 60)
    return summary


# =========================================================
# main
# =========================================================
def main():
    print(f"[CONFIG] dataset          : {DATASET}")
    print(f"[CONFIG] augment          : {AUGMENT}")
    print(f"[CONFIG] n_folds          : {N_FOLDS}")
    print(f"[CONFIG] debug            : {DEBUG}")
    print(f"[CONFIG] target_bot_ratio : {TARGET_BOT_RATIO} "
          f"({'전체 사용' if TARGET_BOT_RATIO == 0 else f'{(1-TARGET_BOT_RATIO)*100:.0f}:{TARGET_BOT_RATIO*100:.0f}'})")
    print(f"[CONFIG] data             : {DATA_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X_all, y_all = load_data(DATA_DIR)
    print(f"[INFO] X_trainval shape : {X_all.shape}")
    print(f"[INFO] Botnet ratio     : {y_all.mean():.4f}")

    print(f"[INFO] K-fold 전 shape: {X_all.shape}  Bot={y_all.sum():,}  "
          f"(subsample은 각 fold train에만 적용)")

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results    = []
    best_fold_score = None
    best_fold_model = None
    best_fold_thr   = 0.5
    best_fold_idx   = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), 1):
        print(f"\n[Fold {fold}/{N_FOLDS}] train={len(train_idx):,}  val={len(val_idx):,}")

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        # 증강 전 라벨 저장 (debug용)
        y_train_orig = y_train.copy()

        # train fold에만 subsample 적용 — val은 원본 분포 유지
        if TARGET_BOT_RATIO > 0:
            from augment_utils import subsample_benign
            X_train, y_train = subsample_benign(X_train, y_train, TARGET_BOT_RATIO)
            y_train_orig = y_train.copy()  # subsample 후 orig 갱신

        # train fold에만 증강 적용 — val fold는 항상 원본 유지
        X_train, y_train = augment_train_fold(
            X_train, y_train, AUGMENT, DATASET, DATA_ROOT
        )
        if AUGMENT != "none":
            print(f"  [AUG] train: {len(y_train):,}  val: {len(y_val):,} (원본)")

        # ── RF class_weight: 증강 여부와 관계없이 항상 balanced_subsample ──
        # DEBUG 시 이중 보정 여부 확인 가능
        class_weight = "balanced_subsample"

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        val_prob    = model.predict_proba(X_val)[:, 1]
        y_pred      = (val_prob >= 0.5).astype(int)
        val_metrics = compute_metrics(y_val, y_pred, val_prob)
        val_metrics["selected_threshold"] = 0.5
        val_metrics["fold"]               = fold
        val_metrics["threshold"]          = 0.5
        fold_results.append(val_metrics)

        print(
            f"  thr=0.50 | "
            f"F1={val_metrics['f1']:.4f} | "
            f"Recall={val_metrics['recall']:.4f} | "
            f"Precision={val_metrics['precision']:.4f} | "
            f"ROC-AUC={val_metrics.get('roc_auc', 0):.4f}"
        )

        # ── 디버그 출력 (--debug 플래그 또는 증강 시 자동) ──
        if DEBUG or AUGMENT != "none":
            debug_fold(
                fold=fold,
                y_val=y_val,
                y_pred=y_pred,
                y_prob=val_prob,
                y_train_orig=y_train_orig,
                y_train_aug=y_train if AUGMENT != "none" else None,
                class_weight=class_weight,
                augment=AUGMENT,
            )

        # 최고 F1 fold 모델 저장
        score = (val_metrics["f1"], val_metrics["recall"])
        if best_fold_score is None or score > best_fold_score:
            best_fold_score = score
            best_fold_model = model
            best_fold_thr   = 0.5
            best_fold_idx   = fold

    # ── K-fold 요약 ──────────────────────────────────────
    summary = print_fold_summary(fold_results)
    print(f"\n[INFO] Best fold: {best_fold_idx}  (F1={best_fold_score[0]:.4f})")

    # ── 저장 ─────────────────────────────────────────────
    joblib.dump(best_fold_model, MODEL_DIR / "rf_flow.pkl")

    with open(MODEL_DIR / "rf_flow_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_fold_thr, "best_fold": best_fold_idx}, f, indent=4)

    output = {
        "dataset":        DATASET,
        "augment":        AUGMENT,
        "n_folds":        N_FOLDS,
        "primary_metric": ["f1", "recall"],
        "summary":        summary,
        "fold_results":   fold_results,
        "best_fold":      best_fold_idx,
    }
    with open(RESULT_DIR / "rf_flow_kfold_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"\n[SAVED] {MODEL_DIR}/rf_flow.pkl")
    print(f"[SAVED] {MODEL_DIR}/rf_flow_threshold.json")
    print(f"[SAVED] {RESULT_DIR}/rf_flow_kfold_results.json")


if __name__ == "__main__":
    main()