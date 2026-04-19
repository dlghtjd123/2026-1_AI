import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_DIR = Path("data/processed/winflat")
MODEL_DIR = Path("artifacts/models")
RESULT_DIR = Path("artifacts/results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy").astype(int)

    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy").astype(int)

    return X_train, y_train, X_val, y_val


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
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

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)

        candidate = (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            -abs(threshold - 0.5),
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
    X_train, y_train, X_val, y_val = load_data()

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_val shape  :", X_val.shape)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, val_metrics = pick_best_threshold(y_val, val_prob)

    print_metrics("RF Validation (Best Threshold)", val_metrics)

    joblib.dump(model, MODEL_DIR / "rf_model.pkl")

    with open(MODEL_DIR / "rf_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold}, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "rf_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4, ensure_ascii=False)

    print("\n[SAVED] artifacts/models/rf_model.pkl")
    print("[SAVED] artifacts/models/rf_threshold.json")
    print("[SAVED] artifacts/results/rf_val_metrics.json")


if __name__ == "__main__":
    main()