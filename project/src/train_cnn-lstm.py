# =========================================================
# CNN-LSTM 이진 분류 모델 학습 스크립트
# ---------------------------------------------------------
# 시퀀스 형태로 전처리된 네트워크 트래픽 데이터를 입력받아
# CNN + LSTM 구조의 딥러닝 모델을 학습한다.
# CNN으로 로컬 패턴을 추출하고, LSTM으로 시간적 흐름을 학습한다.
#
# 주요 기능:
#   - 클래스 불균형 대응: pos_weight 기반 BCEWithLogitsLoss 사용
#   - 최적 임계값 탐색: F1 > Recall > Precision 우선순위로 선정
#   - Early Stopping: val 성능 기준 patience=5 적용
#   - Best Model 저장: val F1 기준 최고 성능 epoch의 가중치 보존
#
# 입력: data/processed/seq/{X_train, y_train, X_val, y_val}.npy
# 출력: artifacts/models/cnn_lstm_model.pt
#       artifacts/models/cnn_lstm_threshold.json
#       artifacts/results/cnn_lstm_val_metrics.json
# =========================================================

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


# =========================================================
# 경로 설정
# ---------------------------------------------------------
# 전처리된 시퀀스 데이터(.npy), 학습된 모델 가중치(.pt),
# 평가 결과(.json)를 저장할 디렉터리를 지정한다.
# MODEL_DIR, RESULT_DIR은 존재하지 않으면 자동 생성한다.
# =========================================================
DATA_DIR = Path("data/processed/seq")
MODEL_DIR = Path("artifacts/models")
RESULT_DIR = Path("artifacts/results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 랜덤 시드 고정
# ---------------------------------------------------------
# Python, NumPy, PyTorch(CPU/GPU) 시드를 동시에 고정하여
# 매 실행마다 동일한 결과를 재현할 수 있도록 한다.
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# PyTorch Dataset 정의
# ---------------------------------------------------------
# NumPy 배열(X, y)을 받아 float32 텐서로 변환한다.
# DataLoader와 함께 사용하며, 인덱스 기반으로 샘플을 반환한다.
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# CNN-LSTM 모델 정의
# ---------------------------------------------------------
# Conv1d로 시퀀스 내 로컬 피처 패턴을 추출한 뒤,
# LSTM으로 시간 방향의 흐름을 학습한다.
# 마지막 LSTM hidden state를 FC 레이어에 통과시켜
# 이진 분류용 logit 스칼라 값을 출력한다.
#
# forward 흐름:
#   입력 (batch, seq_len, feature)
#   → permute → Conv1d → ReLU → Dropout
#   → permute → LSTM → 마지막 hidden state
#   → Dropout → FC → logit (batch,)
# =========================================================
class CNNLSTMModel(nn.Module):
    def __init__(self, n_features, conv_channels=64, lstm_hidden=64, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature)
        x = x.permute(0, 2, 1)   # (batch, feature, seq_len)
        x = self.conv1(x)        # (batch, conv_channels, seq_len)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)   # (batch, seq_len, conv_channels)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]              # (batch, lstm_hidden)

        x = self.dropout(x)
        logits = self.fc(x).squeeze(1)  # (batch,)
        return logits


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
    best_score = None
    best_threshold = 0.5
    best_metrics = None

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)

        # 우선순위: F1 > Recall > Precision > 0.5와의 거리
        score = (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            -abs(threshold - 0.5),
        )

        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 4))
            best_metrics = metrics

    best_metrics["selected_threshold"] = best_threshold
    return best_threshold, best_metrics


# =========================================================
# 검증 손실 및 예측 확률 수집
# ---------------------------------------------------------
# 모델을 eval 모드로 전환한 뒤, 그래디언트 계산 없이
# 전체 로더를 순회하며 loss, 실제 레이블, 예측 확률을 수집한다.
# 평균 손실(avg_loss)과 배열 형태의 y_true, y_prob를 반환한다.
# =========================================================
def collect_probs_and_loss(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            y_true_all.extend(y_batch.cpu().numpy().tolist())
            y_prob_all.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    y_true_all = np.array(y_true_all).astype(int)
    y_prob_all = np.array(y_prob_all)

    return avg_loss, y_true_all, y_prob_all


# =========================================================
# 성능 지표 출력
# ---------------------------------------------------------
# 지표 딕셔너리를 받아 Loss, Threshold, Accuracy, Precision,
# Recall, F1, ROC-AUC, Confusion Matrix를 콘솔에 출력한다.
# loss는 선택적으로 함께 출력할 수 있다.
# =========================================================
def print_metrics(name, metrics, loss=None):
    print(f"\n===== {name} =====")
    if loss is not None:
        print(f"Loss      : {loss:.4f}")
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
# 1. 시드 고정 및 디바이스 설정
# 2. 데이터 로드 및 DataLoader 구성
# 3. CNN-LSTM 모델 초기화
# 4. 클래스 불균형 보정: neg/pos 비율로 pos_weight 산출 후
#    BCEWithLogitsLoss에 적용
# 5. 에폭 단위 학습:
#    - 배치 순회 → forward → loss → backward → gradient clipping → 가중치 갱신
#    - 매 에폭 종료 후 val loss 계산 및 최적 임계값 탐색
#    - Best model 선정 기준: F1 > Recall > Precision > ROC-AUC > -val_loss
#    - patience=5 Early Stopping 적용
# 6. Best epoch 가중치 복원 후 모델/임계값/지표 파일 저장
# =========================================================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    X_train, y_train, X_val, y_val = load_data()

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_val shape  :", X_val.shape)

    n_features = X_train.shape[2]

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = CNNLSTMModel(
        n_features=n_features,
        conv_channels=64,
        lstm_hidden=64,
        dropout=0.3,
    ).to(device)

    # -------------------------------------------------------
    # 클래스 불균형 보정
    # neg/pos 샘플 수 비율을 pos_weight로 산출하여
    # 소수 클래스(공격) 예측에 더 높은 가중치를 부여한다.
    # -------------------------------------------------------
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight_value = float(neg_count / pos_count) if pos_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    print(f"[INFO] pos_weight: {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------------------------------------
    # 학습 상태 변수 초기화
    # -------------------------------------------------------
    num_epochs = 30
    patience = 5
    best_score = None
    best_state = None
    best_threshold = 0.5
    best_epoch = 0
    best_val_loss = None
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            # gradient explosion 방지
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss, y_val_true, y_val_prob = collect_probs_and_loss(
            model, val_loader, device, criterion
        )
        current_threshold, current_val_metrics = pick_best_threshold(
            y_val_true, y_val_prob
        )

        # best model 선정 기준:
        # 1) val F1
        # 2) val Recall
        # 3) val Precision
        # 4) ROC-AUC
        # 5) val_loss가 더 작은 쪽
        current_score = (
            current_val_metrics["f1"],
            current_val_metrics["recall"],
            current_val_metrics["precision"],
            current_val_metrics["roc_auc"] if current_val_metrics["roc_auc"] is not None else -1.0,
            -val_loss,
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_thr={current_threshold:.2f} | "
            f"val_f1={current_val_metrics['f1']:.4f} | "
            f"val_recall={current_val_metrics['recall']:.4f} | "
            f"val_precision={current_val_metrics['precision']:.4f}"
        )

        # -------------------------------------------------------
        # Best Model 갱신 및 Early Stopping 판정
        # 현재 에폭 성능이 이전 최고보다 높으면 가중치를 deepcopy로
        # 보존하고 patience_counter를 초기화한다.
        # 개선이 없으면 카운터를 증가시키고 patience 초과 시 중단한다.
        # -------------------------------------------------------
        if best_score is None or current_score > best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = current_threshold
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_metrics = current_val_metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 저장 직전 정보 다시 반영
    best_val_metrics["selected_threshold"] = best_threshold

    print(f"\n[INFO] Best epoch: {best_epoch}")
    print_metrics("CNN-LSTM Validation (Best Model)", best_val_metrics, loss=best_val_loss)

    # -------------------------------------------------------
    # 결과 저장
    # 모델 가중치 및 하이퍼파라미터, 최적 임계값,
    # 검증 성능 지표를 각각 별도 파일로 저장한다.
    # -------------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "conv_channels": 64,
            "lstm_hidden": 64,
            "dropout": 0.3,
        },
        MODEL_DIR / "cnn_lstm_model.pt",
    )

    with open(MODEL_DIR / "cnn_lstm_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold}, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "cnn_lstm_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_val_metrics, f, indent=4, ensure_ascii=False)

    print("\n[SAVED] artifacts/models/cnn_lstm_model.pt")
    print("[SAVED] artifacts/models/cnn_lstm_threshold.json")
    print("[SAVED] artifacts/results/cnn_lstm_val_metrics.json")


if __name__ == "__main__":
    main()