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


def set_seed(seed=42):
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
        x = x.permute(0, 2, 1)          # (batch, feature, seq_len)
        x = self.conv1(x)               # (batch, conv_channels, seq_len)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)          # (batch, seq_len, conv_channels)
        _, (h_n, _) = self.lstm(x)      # h_n: (num_layers, batch, hidden)
        x = h_n[-1]                     # (batch, hidden)

        x = self.dropout(x)
        logits = self.fc(x).squeeze(1)  # (batch,)
        return logits


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


def evaluate_model(model, loader, device, criterion=None):
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

            if criterion is not None:
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * X_batch.size(0)

            y_true_all.extend(y_batch.cpu().numpy().tolist())
            y_prob_all.extend(probs.cpu().numpy().tolist())

    y_true_all = np.array(y_true_all).astype(int)
    y_prob_all = np.array(y_prob_all)
    y_pred_all = (y_prob_all >= 0.5).astype(int)

    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)

    if criterion is not None:
        metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_val shape  :", X_val.shape)
    print("[INFO] X_test shape :", X_test.shape)

    n_features = X_train.shape[2]

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

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

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    patience = 5
    best_val_f1 = -1.0
    best_state = None
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
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, device, criterion)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # best validation
    best_val_metrics = evaluate_model(model, val_loader, device, criterion)
    print_metrics("CNN-LSTM Validation", best_val_metrics)

    # final test
    test_metrics = evaluate_model(model, test_loader, device, criterion)
    print_metrics("CNN-LSTM Test", test_metrics)

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

    with open(RESULT_DIR / "cnn_lstm_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_val_metrics, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "cnn_lstm_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    print("\n[SAVED] artifacts/models/cnn_lstm_model.pt")
    print("[SAVED] artifacts/results/cnn_lstm_val_metrics.json")
    print("[SAVED] artifacts/results/cnn_lstm_test_metrics.json")


if __name__ == "__main__":
    main()