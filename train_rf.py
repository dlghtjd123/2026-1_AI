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
    y_train = np.load(DATA_DIR / "y_train.npy")

    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test


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


def print_metrics(name, metrics):
    print(f"\n===== {name} =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "ROC-AUC  : None")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_val shape  :", X_val.shape)
    print("[INFO] X_test shape :", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # validation
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, val_pred, val_prob)
    print_metrics("RF Validation", val_metrics)

    # test
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_pred, test_prob)
    print_metrics("RF Test", test_metrics)

    # save model
    joblib.dump(model, MODEL_DIR / "rf_model.pkl")

    # save metrics
    with open(RESULT_DIR / "rf_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "rf_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    print("\n[SAVED] artifacts/models/rf_model.pkl")
    print("[SAVED] artifacts/results/rf_val_metrics.json")
    print("[SAVED] artifacts/results/rf_test_metrics.json")


if __name__ == "__main__":
    main()