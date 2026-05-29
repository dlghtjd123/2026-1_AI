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
from threshold_utils import select_threshold, threshold_label


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
_parser.add_argument("--max_normal", type=int, default=500_000,
                     help="fold당 최대 정상 샘플 수 상한 (기본값: 500000, 0=제한 없음)")
_parser.add_argument("--max_mismatch", type=float, default=None,
                     help="train/val 봇넷 비율 최대 배수 (기본값: cicids2017=5, cicids2018/ctu13=2)")
_parser.add_argument("--threshold_mode", type=str, default="fixed",
                     choices=["fixed", "f1_opt"],
                     help="fixed=0.5, f1_opt=validation F1 기준 threshold 선택")
_args = _parser.parse_args()
AUGMENT      = _args.augment
DATASET      = _args.dataset
N_FOLDS      = _args.n_folds
DEBUG        = _args.debug
MAX_NORMAL   = _args.max_normal
MAX_MISMATCH = _args.max_mismatch if _args.max_mismatch is not None else (
    5.0 if DATASET == "cicids2017" else 2.0
)
THRESHOLD_MODE = _args.threshold_mode


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


def build_model(class_weight):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )


def prepare_train_data(X, y, fold_id=None):
    y_train_orig = y.copy()
    if MAX_NORMAL > 0:
        from augment_utils import subsample_benign
        X, y = subsample_benign(
            X, y, max_normal=MAX_NORMAL, max_mismatch=MAX_MISMATCH
        )
        y_train_orig = y.copy()

    X, y = augment_train_fold(X, y, AUGMENT, DATASET, DATA_ROOT, fold_id=fold_id)
    return X, y, y_train_orig


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
    print(f"[CONFIG] max_normal       : {MAX_NORMAL:,} (0=제한 없음)")
    print(f"[CONFIG] max_mismatch     : {MAX_MISMATCH:.0f}x  (train/val 봇넷 비율 최대 배수)")
    print(f"[CONFIG] threshold_mode   : {THRESHOLD_MODE}")
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
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), 1):
        print(f"\n[Fold {fold}/{N_FOLDS}] train={len(train_idx):,}  val={len(val_idx):,}")

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        # 증강 전 라벨 저장 (debug용)
        X_train, y_train, y_train_orig = prepare_train_data(X_train, y_train, fold_id=fold)
        if AUGMENT != "none":
            print(f"  [AUG] train: {len(y_train):,}  val: {len(y_val):,} (원본)")

        # RF class_weight:
        # - 증강 없음(none): balanced_subsample로 불균형 보정
        # - 증강 있음(smote/gan/wcgan_gp): 이미 subsample+증강으로 보정 → class_weight 제거
        class_weight = None if AUGMENT != "none" else "balanced_subsample"
        print(f"  [RF] class_weight={class_weight}")

        model = build_model(class_weight)
        model.fit(X_train, y_train)

        val_prob    = model.predict_proba(X_val)[:, 1]
        thr         = select_threshold(y_val, val_prob, THRESHOLD_MODE)
        y_pred      = (val_prob >= thr).astype(int)
        val_metrics = compute_metrics(y_val, y_pred, val_prob)
        val_metrics["selected_threshold"] = thr
        val_metrics["fold"]               = fold
        val_metrics["threshold"]          = thr
        fold_results.append(val_metrics)
        fold_thresholds.append(thr)

        print(
            f"  thr={thr:.4f} | "
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
            best_fold_thr   = thr
            best_fold_idx   = fold

    # ── K-fold 요약 ──────────────────────────────────────
    summary = print_fold_summary(fold_results)
    print(f"\n[INFO] Best fold: {best_fold_idx}  (F1={best_fold_score[0]:.4f})")
    final_threshold = (
        float(np.mean(fold_thresholds)) if THRESHOLD_MODE == "f1_opt"
        else best_fold_thr
    )
    print(f"[INFO] Final threshold: {threshold_label(final_threshold, THRESHOLD_MODE)}")

    print("\n[FINAL] trainval 전체로 최종 RF 모델 재학습")
    X_final, y_final, _ = prepare_train_data(X_all, y_all, fold_id="final")
    final_class_weight = None if AUGMENT != "none" else "balanced_subsample"
    final_model = build_model(final_class_weight)
    final_model.fit(X_final, y_final)
    print(f"[FINAL] train={len(y_final):,}  Bot 비율={y_final.mean():.4f}")

    # ── 저장 ─────────────────────────────────────────────
    joblib.dump(final_model, MODEL_DIR / "rf_flow.pkl")

    with open(MODEL_DIR / "rf_flow_threshold.json", "w", encoding="utf-8") as f:
        json.dump({
            "threshold": final_threshold,
            "threshold_mode": THRESHOLD_MODE,
            "best_fold_threshold": best_fold_thr,
            "fold_thresholds": fold_thresholds,
            "best_fold": best_fold_idx,
            "saved_model": "final_trainval_refit",
        }, f, indent=4)

    output = {
        "dataset":        DATASET,
        "augment":        AUGMENT,
        "n_folds":        N_FOLDS,
        "primary_metric": ["f1", "recall"],
        "summary":        summary,
        "fold_results":   fold_results,
        "best_fold":      best_fold_idx,
        "threshold_mode": THRESHOLD_MODE,
        "final_threshold": final_threshold,
        "fold_thresholds": fold_thresholds,
        "saved_model":    "final_trainval_refit",
    }
    with open(RESULT_DIR / "rf_flow_kfold_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"\n[SAVED] {MODEL_DIR}/rf_flow.pkl")
    print(f"[SAVED] {MODEL_DIR}/rf_flow_threshold.json")
    print(f"[SAVED] {RESULT_DIR}/rf_flow_kfold_results.json")


if __name__ == "__main__":
    main()
