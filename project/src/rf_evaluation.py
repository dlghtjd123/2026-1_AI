"""
Random Forest 학습과 CICIDS2017 13-class 평가 지표 계산을 담당하는 파일.

전체 macro/weighted 지표뿐 아니라 연구 핵심 대상인 Bot 클래스의
Precision, Recall, F1, FNR, FPR을 별도로 계산한다.
"""

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
    """
    classification_report와 혼동행렬에서 Bot 클래스 전용 지표를 계산한다.

    Bot 탐지 성능을 보기 위해 TP, FP, FN, TN을 직접 계산하고,
    Bot Recall의 반대값인 FNR도 함께 반환한다.
    """
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
    """
    예측 결과를 받아 전체 다중분류 지표와 Bot 클래스 지표를 계산한다.

    반환값에는 정확도, macro precision/recall/F1, weighted F1,
    Bot 전용 지표, classification report, 혼동행렬이 포함된다.
    """
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
    """
    증강된 학습 세트로 Random Forest를 학습하고 원본 테스트 세트에서 평가한다.

    테스트 세트는 증강하지 않은 원본 분포를 유지하므로, 증강 방식이 실제 Bot 탐지에
    도움이 되는지 확인할 수 있다.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    metrics = evaluate_predictions(y_test, model.predict(X_test))
    return model, metrics
