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


DATA_DIR = Path("data/processed/seq")
MODEL_DIR = Path("artifacts/models")
RESULT_DIR = Path("artifacts/results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight_value = float(neg_count / pos_count) if pos_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    print(f"[INFO] pos_weight: {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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