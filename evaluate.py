import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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


WINFLAT_DIR = Path("data/processed/winflat")
SEQ_DIR = Path("data/processed/seq")
MODEL_DIR = Path("artifacts/models")
RESULT_DIR = Path("artifacts/results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)


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
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]

        x = self.dropout(x)
        logits = self.fc(x).squeeze(1)
        return logits


def load_threshold(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["threshold"]


def compute_metrics(y_true, y_pred, y_prob, threshold):
    metrics = {
        "threshold": float(threshold),
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
    print(f"Threshold : {metrics['threshold']:.2f}")
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


def evaluate_rf():
    model = joblib.load(MODEL_DIR / "rf_model.pkl")
    threshold = load_threshold(MODEL_DIR / "rf_threshold.json")

    X_test = np.load(WINFLAT_DIR / "X_test.npy")
    y_test = np.load(WINFLAT_DIR / "y_test.npy").astype(int)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return compute_metrics(y_test, y_pred, y_prob, threshold)


def evaluate_xgb():
    model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    threshold = load_threshold(MODEL_DIR / "xgb_threshold.json")

    X_test = np.load(WINFLAT_DIR / "X_test.npy")
    y_test = np.load(WINFLAT_DIR / "y_test.npy").astype(int)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return compute_metrics(y_test, y_pred, y_prob, threshold)


def evaluate_cnn_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = load_threshold(MODEL_DIR / "cnn_lstm_threshold.json")

    checkpoint = torch.load(MODEL_DIR / "cnn_lstm_model.pt", map_location=device)

    model = CNNLSTMModel(
        n_features=checkpoint["n_features"],
        conv_channels=checkpoint["conv_channels"],
        lstm_hidden=checkpoint["lstm_hidden"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test = np.load(SEQ_DIR / "X_test.npy")
    y_test = np.load(SEQ_DIR / "y_test.npy").astype(int)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        y_prob = torch.sigmoid(logits).cpu().numpy()

    y_pred = (y_prob >= threshold).astype(int)

    return compute_metrics(y_test, y_pred, y_prob, threshold)


def main():
    rf_metrics = evaluate_rf()
    xgb_metrics = evaluate_xgb()
    cnn_metrics = evaluate_cnn_lstm()

    print_metrics("RF Test", rf_metrics)
    print_metrics("XGBoost Test", xgb_metrics)
    print_metrics("CNN-LSTM Test", cnn_metrics)

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "RandomForest",
                "Threshold": rf_metrics["threshold"],
                "Accuracy": rf_metrics["accuracy"],
                "Precision": rf_metrics["precision"],
                "Recall": rf_metrics["recall"],
                "F1": rf_metrics["f1"],
                "ROC-AUC": rf_metrics["roc_auc"],
            },
            {
                "Model": "XGBoost",
                "Threshold": xgb_metrics["threshold"],
                "Accuracy": xgb_metrics["accuracy"],
                "Precision": xgb_metrics["precision"],
                "Recall": xgb_metrics["recall"],
                "F1": xgb_metrics["f1"],
                "ROC-AUC": xgb_metrics["roc_auc"],
            },
            {
                "Model": "CNN-LSTM",
                "Threshold": cnn_metrics["threshold"],
                "Accuracy": cnn_metrics["accuracy"],
                "Precision": cnn_metrics["precision"],
                "Recall": cnn_metrics["recall"],
                "F1": cnn_metrics["f1"],
                "ROC-AUC": cnn_metrics["roc_auc"],
            },
        ]
    )

    print("\n===== Model Comparison =====")
    print(comparison_df)

    comparison_df.to_csv(RESULT_DIR / "model_comparison.csv", index=False)

    with open(RESULT_DIR / "rf_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rf_metrics, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "xgb_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(xgb_metrics, f, indent=4, ensure_ascii=False)

    with open(RESULT_DIR / "cnn_lstm_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cnn_metrics, f, indent=4, ensure_ascii=False)

    print("\n[SAVED] artifacts/results/model_comparison.csv")
    print("[SAVED] artifacts/results/rf_test_metrics.json")
    print("[SAVED] artifacts/results/xgb_test_metrics.json")
    print("[SAVED] artifacts/results/cnn_lstm_test_metrics.json")


if __name__ == "__main__":
    main()