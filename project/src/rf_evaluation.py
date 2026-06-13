from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from cicids2017_bot_config import BOT_CLASS_ID, CLASS_NAMES


def compute_bot_metrics(report: dict, cm: np.ndarray) -> dict:
    bot_name = "Bot"
    bot_row = report.get(bot_name, {})
    bot_id = BOT_CLASS_ID
    tp = int(cm[bot_id, bot_id])
    fn = int(cm[bot_id, :].sum() - tp)
    fp = int(cm[:, bot_id].sum() - tp)
    tn = int(cm.sum() - tp - fn - fp)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": float(bot_row.get("precision", 0.0)),
        "recall": float(bot_row.get("recall", recall)),
        "f1": float(bot_row.get("f1-score", 0.0)),
        "support": int(bot_row.get("support", 0)),
        "fnr": float(1.0 - recall),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_predictions(y_test: np.ndarray, pred: np.ndarray) -> dict:
    report = classification_report(
        y_test,
        pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASS_NAMES))))
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "bot": compute_bot_metrics(report, cm),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def evaluate_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int,
) -> tuple[RandomForestClassifier, dict]:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    metrics = evaluate_predictions(y_test, model.predict(X_test))
    return model, metrics
