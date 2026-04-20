# =========================================================
# XGBoost 이진 분류 모델 학습 스크립트
# ---------------------------------------------------------
# 윈도우 플래튼(window-flattened) 형태로 전처리된 네트워크
# 트래픽 데이터를 입력받아 XGBoost 분류기를 학습한다.
#
# 주요 기능:
#   - 클래스 불균형 대응: neg/pos 비율로 scale_pos_weight 산출
#   - Early Stopping: val logloss 기준 50 라운드 개선 없으면 중단
#   - 최적 임계값 탐색: F1 > Recall > Precision 우선순위로 선정
#   - 학습된 모델/임계값/평가 지표를 파일로 저장
#
# 입력: data/processed/winflat/{X_train, y_train, X_val, y_val}.npy
# 출력: artifacts/models/xgb_model.pkl
#       artifacts/models/xgb_threshold.json
#       artifacts/results/xgb_val_metrics.json
# =========================================================

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
# ---------------------------------------------------------
# 전처리된 윈도우 플래튼 데이터(.npy), 학습된 모델 객체(.pkl),
# 평가 결과(.json)를 저장할 디렉터리를 지정한다.
# MODEL_DIR, RESULT_DIR은 존재하지 않으면 자동 생성한다.
# =========================================================
DATA_DIR = Path("data/processed/winflat")
MODEL_DIR = Path("artifacts/models")
RESULT_DIR = Path("artifacts/results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 데이터 로드
# ---------------------------------------------------------
# DATA_DIR에서 train/val 분할 데이터를 NumPy 배열로 불러온다.
# 레이블(y)은 정수형으로 변환하여 반환한다.
# =========================================================
def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy").astype(int)

    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy").astype(int)

    return X_train, y_train, X_val, y_val


# =========================================================
# 분류 성능 지표 계산
# ---------------------------------------------------------
# 실제 레이블(y_true), 예측 레이블(y_pred), 예측 확률(y_prob)을 받아
# Accuracy, Precision, Recall, F1, Confusion Matrix,
# Classification Report, ROC-AUC를 딕셔너리로 반환한다.
# 단일 클래스만 존재하는 경우 ROC-AUC는 None으로 처리한다.
# =========================================================
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


# =========================================================
# 최적 분류 임계값 탐색
# ---------------------------------------------------------
# 0.05 ~ 0.95 구간을 0.01 간격으로 순회하며 각 임계값에서
# 성능 지표를 계산한다.
# 선정 우선순위: F1 > Recall > Precision > 0.5와의 거리
# 최종적으로 최적 임계값과 해당 시점의 지표 딕셔너리를 반환한다.
# =========================================================
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


# =========================================================
# 성능 지표 출력
# ---------------------------------------------------------
# 지표 딕셔너리를 받아 Threshold, Accuracy, Precision,
# Recall, F1, ROC-AUC, Confusion Matrix를 콘솔에 출력한다.
# =========================================================
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


# =========================================================
# 메인 학습 루프
# ---------------------------------------------------------
# 전체 학습 파이프라인을 순서대로 실행한다.
#
# 1. 데이터 로드
# 2. 클래스 불균형 보정: neg/pos 비율로 scale_pos_weight 산출
# 3. XGBoost 모델 초기화 및 학습
#    - n_estimators=1000   : 최대 트리 1000개 (Early Stopping으로 실제 수 결정)
#    - max_depth=6         : 트리 깊이 제한으로 과적합 방지
#    - learning_rate=0.05  : 낮은 학습률로 안정적 수렴 유도
#    - subsample=0.8       : 각 트리 학습 시 80% 행 샘플링
#    - colsample_bytree=0.8: 각 트리 학습 시 80% 피처 샘플링
#    - early_stopping_rounds=50: val logloss 기준 50 라운드
#      개선 없으면 학습 조기 종료
# 4. 검증 데이터로 예측 확률 산출 후 최적 임계값 탐색
# 5. 성능 지표 출력
# 6. 모델/임계값/지표 파일 저장
# =========================================================
def main():
    X_train, y_train, X_val, y_val = load_data()

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_val shape  :", X_val.shape)

    # -------------------------------------------------------
    # 클래스 불균형 보정
    # neg/pos 샘플 수 비율을 scale_pos_weight로 산출하여
    # 소수 클래스(공격) 예측에 더 높은 가중치를 부여한다.
    # -------------------------------------------------------
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    print(f"[INFO] scale_pos_weight: {scale_pos_weight:.4f}")

    # -------------------------------------------------------
    # XGBoost 모델 초기화
    # n_jobs=-1로 모든 CPU 코어를 병렬 사용하여 학습 속도를 높인다.
    # -------------------------------------------------------
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
    )

    # -------------------------------------------------------
    # 모델 학습
    # eval_set으로 검증 데이터를 전달하여 Early Stopping에 활용한다.
    # verbose=False로 에폭별 로그 출력을 억제한다.
    # -------------------------------------------------------
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # -------------------------------------------------------
    # 검증 데이터 평가 및 최적 임계값 탐색
    # predict_proba의 양성 클래스(인덱스 1) 확률을 사용한다.
    # -------------------------------------------------------
    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold, val_metrics = pick_best_threshold(y_val, val_prob)

    print_metrics("XGBoost Validation (Best Threshold)", val_metrics)

    # -------------------------------------------------------
    # 결과 저장
    # 모델 객체, 최적 임계값, 검증 성능 지표를 각각 별도 파일로 저장한다.
    # -------------------------------------------------------
    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")

    with open(MODEL_DIR / "xgb_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold}, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "xgb_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4, ensure_ascii=False)

    print("\n[SAVED] artifacts/models/xgb_model.pkl")
    print("[SAVED] artifacts/models/xgb_threshold.json")
    print("[SAVED] artifacts/results/xgb_val_metrics.json")


if __name__ == "__main__":
    main()