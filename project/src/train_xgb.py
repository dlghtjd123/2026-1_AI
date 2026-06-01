import argparse
import json
from pathlib import Path

import joblib
import numpy as np
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
from xgboost import XGBClassifier

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_utils import augment_train_fold


_parser = argparse.ArgumentParser()
_parser.add_argument("--augment", type=str, default="none",
                     choices=["none", "smote", "gan", "wgan_gp", "wcgan_gp"])
_parser.add_argument("--dataset", type=str, default="cicids2017",
                     choices=["cicids2017", "cicids2018", "ctu13"])
_parser.add_argument("--n_folds", type=int, default=5)
_parser.add_argument("--target_bot_ratio", type=float, default=0.05,
                     help="Benign 축소 후 목표 봇넷 비율 (기본값: 0.05=95:5, 0=전체 사용)")
AUGMENT          = _parser.parse_args().augment
DATASET          = _parser.parse_args().dataset
N_FOLDS          = _parser.parse_args().n_folds
TARGET_BOT_RATIO = _parser.parse_args().target_bot_ratio


_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent
_ROOT     = _PROJECT.parent

DATA_DIR   = _PROJECT / "data" / "processed" / DATASET / "flat"
DATA_ROOT  = _PROJECT / "data" / "processed"
_MODEL_SUFFIX = f"_{AUGMENT}" if AUGMENT != "none" else ""
MODEL_DIR  = _ROOT / "artifacts" / f"models_{DATASET}{_MODEL_SUFFIX}" / "xgb"
RESULT_DIR = _ROOT / "artifacts" / f"results_{DATASET}{_MODEL_SUFFIX}"


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
    print(f"  XGBoost  {N_FOLDS}-Fold CV 결과 요약  [{DATASET} / {AUGMENT}]")
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


def main():
    print(f"[CONFIG] dataset          : {DATASET}")
    print(f"[CONFIG] augment          : {AUGMENT}")
    print(f"[CONFIG] n_folds          : {N_FOLDS}")
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

        # train fold에만 subsample 적용 — val은 원본 분포 유지
        if TARGET_BOT_RATIO > 0:
            from augment_utils import subsample_benign
            X_train, y_train = subsample_benign(X_train, y_train, TARGET_BOT_RATIO)

        X_train, y_train = augment_train_fold(
            X_train, y_train, AUGMENT, DATASET, DATA_ROOT
        )
        if AUGMENT != "none":
            print(f"  [AUG] train: {len(y_train):,}  val: {len(y_val):,} (원본)")

        pos_count        = int(np.sum(y_train == 1))
        neg_count        = int(np.sum(y_train == 0))
        scale_pos_weight = float(np.sqrt(neg_count / pos_count))

        model = XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=50,
            tree_method="hist",
            device="cpu",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_prob = model.predict_proba(X_val)[:, 1]

        # val F1 최적 threshold 탐색
        thresholds = np.arange(0.05, 0.95, 0.01)
        f1_scores  = [f1_score(y_val, (val_prob >= t).astype(int), zero_division=0)
                      for t in thresholds]
        best_thr   = float(thresholds[np.argmax(f1_scores)])

        y_pred      = (val_prob >= best_thr).astype(int)
        val_metrics = compute_metrics(y_val, y_pred, val_prob)
        val_metrics["selected_threshold"] = best_thr
        val_metrics["fold"]               = fold
        val_metrics["threshold"]          = best_thr
        fold_results.append(val_metrics)

        print(f"  thr={best_thr:.2f} | F1={val_metrics['f1']:.4f} | "
              f"Recall={val_metrics['recall']:.4f} | "
              f"Precision={val_metrics['precision']:.4f} | "
              f"ROC-AUC={val_metrics.get('roc_auc', 0):.4f}")

        score = (val_metrics["f1"], val_metrics["recall"])
        if best_fold_score is None or score > best_fold_score:
            best_fold_score = score
            best_fold_model = model
            best_fold_thr   = best_thr
            best_fold_idx   = fold

    summary = print_fold_summary(fold_results)
    print(f"\n[INFO] Best fold: {best_fold_idx}  (F1={best_fold_score[0]:.4f})")

    joblib.dump(best_fold_model, MODEL_DIR / "xgb_flow.pkl")

    all_thresholds = [r["selected_threshold"] for r in fold_results]
    mean_thr = float(np.mean(all_thresholds))
    with open(MODEL_DIR / "xgb_flow_threshold.json", "w", encoding="utf-8") as f:
<<<<<<< Updated upstream
        json.dump({"threshold": best_fold_thr, "best_fold": best_fold_idx}, f, indent=4)
=======
        json.dump({
            "threshold": mean_thr,
            "per_fold_thresholds": all_thresholds,
            "best_fold": best_fold_idx,
            "best_n_estimators": best_n_estimators,
            "saved_model": "final_trainval_refit",
        }, f, indent=4)
>>>>>>> Stashed changes

    output = {
        "dataset": DATASET, "augment": AUGMENT, "n_folds": N_FOLDS,
        "primary_metric": ["f1", "recall"],
        "summary": summary, "fold_results": fold_results, "best_fold": best_fold_idx,
    }
    with open(RESULT_DIR / "xgb_flow_kfold_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"\n[SAVED] {MODEL_DIR}/xgb_flow.pkl")
    print(f"[SAVED] {MODEL_DIR}/xgb_flow_threshold.json")
    print(f"[SAVED] {RESULT_DIR}/xgb_flow_kfold_results.json")


if __name__ == "__main__":
    main()
