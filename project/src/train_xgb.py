import json
from pathlib import Path

import joblib
import numpy as np
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


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR   = Path(__file__).resolve().parent
_PROJECT   = _SRC_DIR.parent
_ROOT      = _PROJECT.parent

MODEL_DIR  = _ROOT / "artifacts" / "models"
RESULT_DIR = _ROOT / "artifacts" / "results"

DATA_DIR = _PROJECT / "data" / "processed" / "cicids2017" / "flat"


def load_data(data_dir: Path):
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy").astype(int)
    X_val   = np.load(data_dir / "X_val.npy")
    y_val   = np.load(data_dir / "y_val.npy").astype(int)
    return X_train, y_train, X_val, y_val


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


def pick_best_threshold(y_true, y_prob):
    best = None
    best_threshold = 0.5
    best_metrics = None

    thresholds = np.arange(0.05, 0.96, 0.01)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)

        # Recall을 너무 낮게 만드는 threshold는 제외
        if metrics["recall"] < 0.8:
            continue

        candidate = (
            metrics["f1"],
            metrics["precision"],
            -threshold,
        )

        if best is None or candidate > best:
            best = candidate
            best_threshold = float(round(threshold, 4))
            best_metrics = metrics

    # recall >= 0.8을 만족하는 threshold가 없을 경우 fallback
    if best_metrics is None:
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = compute_metrics(y_true, y_pred, y_prob)

            candidate = (
                metrics["recall"],
                metrics["f1"],
                metrics["precision"],
                -threshold,
            )

            if best is None or candidate > best:
                best = candidate
                best_threshold = float(round(threshold, 4))
                best_metrics = metrics

    best_metrics["selected_threshold"] = best_threshold
    return best_threshold, best_metrics


def print_metrics(name, metrics):
    print(f"\n===== {name} =====")
    print(f"Threshold : {metrics.get('selected_threshold', 0.5):.2f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(
        f"ROC-AUC   : {metrics['roc_auc']:.4f}"
        if metrics["roc_auc"] is not None
        else "ROC-AUC   : None"
    )
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))


def main():
    print(f"[CONFIG] data : {DATA_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_data(DATA_DIR)
    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_val shape  :", X_val.shape)
    print(f"[INFO] Botnet ratio (train): {y_train.mean():.4f}")

    pos_count        = int(np.sum(y_train == 1))
    neg_count        = int(np.sum(y_train == 0))
    scale_pos_weight = np.sqrt(neg_count / pos_count)
    print(f"[INFO] scale_pos_weight: {scale_pos_weight:.4f}")

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
        device="cuda",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, val_metrics = pick_best_threshold(y_val, val_prob)
    print_metrics("XGBoost Validation", val_metrics)

    joblib.dump(model, MODEL_DIR / "xgb_flow.pkl")

    with open(MODEL_DIR / "xgb_flow_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold}, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "xgb_flow_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4, ensure_ascii=False)

    print(f"\n[SAVED] artifacts/models/xgb_flow.pkl")
    print(f"[SAVED] artifacts/models/xgb_flow_threshold.json")
    print(f"[SAVED] artifacts/results/xgb_flow_val_metrics.json")


if __name__ == "__main__":
    main()