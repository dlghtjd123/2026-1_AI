import copy
import json
import random
import argparse
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
# =========================================================
_SRC_DIR   = Path(__file__).resolve().parent
_PROJECT   = _SRC_DIR.parent
_ROOT      = _PROJECT.parent

MODEL_DIR  = _ROOT / "artifacts" / "models"
RESULT_DIR = _ROOT / "artifacts" / "results"

DATA_DIR = _PROJECT / "data" / "processed" / "cicids2017" / "seq"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--augment",
    type=str,
    default="none",
    choices=["none", "smote", "gan", "wgan_gp", "wcgan_gp"],
)
AUGMENT = parser.parse_args().augment


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


class CNNGRUModel(nn.Module):
    def __init__(self, n_features, conv_channels=64, gru_hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(gru_hidden, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)

        _, h_n = self.gru(x)
        x = self.dropout(h_n[-1])
        return self.fc(x).squeeze(1)


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


def collect_probs_and_loss(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    y_true_all, y_prob_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs  = torch.sigmoid(logits)
            total_loss += criterion(logits, y_batch).item() * X_batch.size(0)
            y_true_all.extend(y_batch.cpu().numpy().tolist())
            y_prob_all.extend(probs.cpu().numpy().tolist())

    return (
        total_loss / len(loader.dataset),
        np.array(y_true_all).astype(int),
        np.array(y_prob_all),
    )


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

    print(f"[CONFIG] data   : {DATA_DIR}")
    print(f"[INFO]   device : {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_data(DATA_DIR)
    print("[INFO] X_train shape:", X_train.shape)
    print("[INFO] X_val shape  :", X_val.shape)
    print(f"[INFO] Botnet ratio (train): {y_train.mean():.4f}")

    n_features   = X_train.shape[2]
    train_loader = DataLoader(
        SequenceDataset(X_train, y_train), batch_size=128, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),     batch_size=256, shuffle=False, num_workers=0
    )

    model = CNNGRUModel(
        n_features=n_features,
        conv_channels=64,
        gru_hidden=64,
        dropout=0.3,
    ).to(device)

    pos_count        = np.sum(y_train == 1)
    neg_count        = np.sum(y_train == 0)
    pos_weight_value = np.sqrt(float(neg_count / pos_count))
    pos_weight       = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"[INFO] pos_weight: {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs       = 30
    patience         = 5
    best_score       = None
    best_state       = None
    best_threshold   = 0.5
    best_epoch       = 0
    best_val_loss    = None
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, y_val_true, y_val_prob = collect_probs_and_loss(
            model, val_loader, device, criterion
        )
        current_threshold, current_val_metrics = pick_best_threshold(y_val_true, y_val_prob)

        current_score = (
            current_val_metrics["f1"],
            current_val_metrics["recall"],
            current_val_metrics["precision"],
            current_val_metrics["roc_auc"] if current_val_metrics["roc_auc"] is not None else -1.0,
            -val_loss,
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"thr={current_threshold:.2f} | f1={current_val_metrics['f1']:.4f} | "
            f"recall={current_val_metrics['recall']:.4f}"
        )

        if best_score is None or current_score > best_score:
            best_score       = current_score
            best_state       = copy.deepcopy(model.state_dict())
            best_threshold   = current_threshold
            best_epoch       = epoch
            best_val_loss    = val_loss
            best_val_metrics = current_val_metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_val_metrics["selected_threshold"] = best_threshold
    print(f"\n[INFO] Best epoch: {best_epoch}")
    print_metrics("CNN-GRU Validation", best_val_metrics, loss=best_val_loss)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features":    n_features,
            "window_size":   X_train.shape[1],
            "conv_channels": 64,
            "gru_hidden":    64,
            "dropout":       0.3,
        },
        MODEL_DIR / "cnn_gru_flow.pt",
    )

    with open(MODEL_DIR / "cnn_gru_flow_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold}, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "cnn_gru_flow_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_val_metrics, f, indent=4, ensure_ascii=False)

    print(f"\n[SAVED] artifacts/models/cnn_gru_flow.pt")
    print(f"[SAVED] artifacts/models/cnn_gru_flow_threshold.json")
    print(f"[SAVED] artifacts/results/cnn_gru_flow_val_metrics.json")


if __name__ == "__main__":
    main()