"""
evaluate.py

1단계: CIC-IDS2017 내부 test 평가
2단계: CTU-13 시나리오 9 교차검증 (Adaptive threshold — Safety 2025 방식)

사용법:
  python evaluate.py                    # Baseline
  python evaluate.py --augment smote
  python evaluate.py --augment gan
  python evaluate.py --augment wgan_gp
  python evaluate.py --augment wcgan_gp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# --augment 인자 파싱
# =========================================================
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--augment",
    type=str,
    default="none",
    choices=["none", "smote", "gan", "wgan_gp", "wcgan_gp"],
    help="사용할 증강 방식 (default: none)",
)
AUGMENT = _parser.parse_args().augment


# =========================================================
# 경로 설정 — AUGMENT 값에 따라 자동 변경
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent
_ROOT     = _PROJECT.parent

_SUFFIX   = f"_{AUGMENT}" if AUGMENT != "none" else ""

MODEL_DIR  = _ROOT / "artifacts" / f"models{_SUFFIX}"
RESULT_DIR = _ROOT / "artifacts" / f"results{_SUFFIX}"
DATA_ROOT  = _PROJECT / "data" / "processed"

# CIC-IDS2017 test 데이터는 증강과 무관하게 원본 사용
CIC_FLAT = DATA_ROOT / "cicids2017" / "flat"
CIC_SEQ  = DATA_ROOT / "cicids2017" / "seq"

# CTU-13은 항상 동일
CTU_FLAT = DATA_ROOT / "ctu13" / "flat"
CTU_SEQ  = DATA_ROOT / "ctu13" / "seq"


# =========================================================
# 모델 정의
# =========================================================
class CNNLSTMModel(nn.Module):
    def __init__(self, n_features, conv_channels=64, lstm_hidden=64, dropout=0.3):
        super().__init__()
        self.conv1   = nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm    = nn.LSTM(conv_channels, lstm_hidden, num_layers=1, batch_first=True)
        self.fc      = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(x).permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1])).squeeze(1)


class GRUModel(nn.Module):
    def __init__(self, n_features, gru_hidden=64, dropout=0.3):
        super().__init__()
        self.gru     = nn.GRU(n_features, gru_hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(gru_hidden, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1])).squeeze(1)


class CNNGRUModel(nn.Module):
    def __init__(self, n_features, conv_channels=64, gru_hidden=64, dropout=0.3):
        super().__init__()
        self.conv1   = nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gru     = nn.GRU(conv_channels, gru_hidden, num_layers=1, batch_first=True)
        self.fc      = nn.Linear(gru_hidden, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(x).permute(0, 2, 1)
        _, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1])).squeeze(1)


# =========================================================
# 지표 계산
# =========================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
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


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx  = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


# =========================================================
# 모델 로드 & 예측
# =========================================================
def load_sklearn_model(name: str):
    model     = joblib.load(MODEL_DIR / f"{name}.pkl")
    threshold = json.loads((MODEL_DIR / f"{name}_threshold.json").read_text())["threshold"]
    return model, float(threshold)


def load_torch_model(name: str):
    ckpt_path = MODEL_DIR / f"{name}.pt"
    threshold = json.loads((MODEL_DIR / f"{name}_threshold.json").read_text())["threshold"]
    return ckpt_path, float(threshold)


def predict_sklearn_probs(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def load_sequence_model(ckpt_path: Path, model_type: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if model_type == "cnn_lstm":
        model = CNNLSTMModel(
            n_features    = ckpt["n_features"],
            conv_channels = ckpt.get("conv_channels", 64),
            lstm_hidden   = ckpt.get("lstm_hidden", 64),
            dropout       = ckpt.get("dropout", 0.3),
        )
    elif model_type == "gru":
        model = GRUModel(
            n_features = ckpt["n_features"],
            gru_hidden = ckpt.get("gru_hidden", 64),
            dropout    = ckpt.get("dropout", 0.3),
        )
    elif model_type == "cnn_gru":
        model = CNNGRUModel(
            n_features    = ckpt["n_features"],
            conv_channels = ckpt.get("conv_channels", 64),
            gru_hidden    = ckpt.get("gru_hidden", 64),
            dropout       = ckpt.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def predict_sequence_probs(
    ckpt_path:  Path,
    model_type: str,
    X:          np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_sequence_model(ckpt_path, model_type, device)
    probs  = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_b = torch.tensor(
                X[start : start + batch_size], dtype=torch.float32
            ).to(device)
            probs.append(torch.sigmoid(model(X_b)).cpu().numpy())
    return np.concatenate(probs)


# =========================================================
# 비교표 출력
# =========================================================
def row_from_metrics(model_name: str, m: dict) -> dict:
    return {
        "model":     model_name,
        "threshold": m.get("threshold"),
        "accuracy":  m["accuracy"],
        "precision": m["precision"],
        "recall":    m["recall"],
        "f1":        m["f1"],
        "roc_auc":   m["roc_auc"],
    }


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(f"{'Model':<22} {'Threshold':>10} {'Accuracy':>9} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 78)
    for r in rows:
        roc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "   None"
        thr = f"{r['threshold']:.4f}" if r.get("threshold") is not None else "     -"
        print(f"{r['model']:<22} {thr:>10} {r['accuracy']:>9.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {roc:>9}")
    print("=" * 78)


# =========================================================
# 1단계: CIC-IDS2017 내부 test 평가
# =========================================================
def stage1_cic_test() -> dict:
    print("\n[1단계] CIC-IDS2017 내부 test 평가")

    X_flat = np.load(CIC_FLAT / "X_test.npy")
    y_flat = np.load(CIC_FLAT / "y_test.npy").astype(int)
    X_seq  = np.load(CIC_SEQ  / "X_test.npy")
    y_seq  = np.load(CIC_SEQ  / "y_test.npy").astype(int)

    print(f"  flat: {X_flat.shape}  Bot 비율: {y_flat.mean():.4f}")
    print(f"  seq:  {X_seq.shape}   Bot 비율: {y_seq.mean():.4f}")

    rf_model,     rf_thr      = load_sklearn_model("rf_flow")
    xgb_model,    xgb_thr     = load_sklearn_model("xgb_flow")
    cnn_ckpt,     cnn_thr     = load_torch_model("cnn_lstm_flow")
    gru_ckpt,     gru_thr     = load_torch_model("gru_flow")
    cnn_gru_ckpt, cnn_gru_thr = load_torch_model("cnn_gru_flow")

    def eval_sk(model, thr, X, y):
        prob = predict_sklearn_probs(model, X)
        pred = (prob >= thr).astype(int)
        m    = compute_metrics(y, pred, prob)
        m["threshold"] = thr
        return m

    def eval_seq(ckpt, mtype, thr, X, y):
        prob = predict_sequence_probs(ckpt, mtype, X)
        pred = (prob >= thr).astype(int)
        m    = compute_metrics(y, pred, prob)
        m["threshold"] = thr
        return m

    results = {
        "rf":       eval_sk(rf_model,  rf_thr,  X_flat, y_flat),
        "xgb":      eval_sk(xgb_model, xgb_thr, X_flat, y_flat),
        "cnn_lstm": eval_seq(cnn_ckpt,     "cnn_lstm", cnn_thr,     X_seq, y_seq),
        "gru":      eval_seq(gru_ckpt,     "gru",      gru_thr,     X_seq, y_seq),
        "cnn_gru":  eval_seq(cnn_gru_ckpt, "cnn_gru",  cnn_gru_thr, X_seq, y_seq),
    }

    print_table(f"1단계: CIC-IDS2017 내부 test [{AUGMENT}]", [
        row_from_metrics("RF",       results["rf"]),
        row_from_metrics("XGBoost",  results["xgb"]),
        row_from_metrics("CNN-LSTM", results["cnn_lstm"]),
        row_from_metrics("GRU",      results["gru"]),
        row_from_metrics("CNN-GRU",  results["cnn_gru"]),
    ])

    return results


# =========================================================
# 2단계: CTU-13 시나리오 9 교차검증 (Adaptive threshold)
# =========================================================
def stage2_ctu13_cross() -> dict:
    print("\n[2단계] CTU-13 시나리오 9 교차검증")
    print("  방식: Safety 2025 (Scaler 정렬 + Adaptive threshold)")
    print("  ※ target 정보 사용 — 논문 명시 필요")

    X_flat = np.load(CTU_FLAT / "X_test.npy")
    y_flat = np.load(CTU_FLAT / "y_test.npy").astype(int)
    X_seq  = np.load(CTU_SEQ  / "X_test.npy")
    y_seq  = np.load(CTU_SEQ  / "y_test.npy").astype(int)

    print(f"\n  flat: {X_flat.shape}  Bot 비율: {y_flat.mean():.4f}")
    print(f"  seq:  {X_seq.shape}   Bot 비율: {y_seq.mean():.4f}")

    rf_model,     _ = load_sklearn_model("rf_flow")
    xgb_model,    _ = load_sklearn_model("xgb_flow")
    cnn_ckpt,     _ = load_torch_model("cnn_lstm_flow")
    gru_ckpt,     _ = load_torch_model("gru_flow")
    cnn_gru_ckpt, _ = load_torch_model("cnn_gru_flow")

    rf_prob      = predict_sklearn_probs(rf_model,  X_flat)
    xgb_prob     = predict_sklearn_probs(xgb_model, X_flat)
    cnn_prob     = predict_sequence_probs(cnn_ckpt,     "cnn_lstm", X_seq)
    gru_prob     = predict_sequence_probs(gru_ckpt,     "gru",      X_seq)
    cnn_gru_prob = predict_sequence_probs(cnn_gru_ckpt, "cnn_gru",  X_seq)

    def eval_adaptive(y_true, y_prob):
        thr  = find_best_threshold(y_true, y_prob)
        pred = (y_prob >= thr).astype(int)
        m    = compute_metrics(y_true, pred, y_prob)
        m["threshold"] = thr
        return m

    results = {
        "rf":       eval_adaptive(y_flat, rf_prob),
        "xgb":      eval_adaptive(y_flat, xgb_prob),
        "cnn_lstm": eval_adaptive(y_seq,  cnn_prob),
        "gru":      eval_adaptive(y_seq,  gru_prob),
        "cnn_gru":  eval_adaptive(y_seq,  cnn_gru_prob),
    }

    print_table(f"2단계: CTU-13 교차검증 (Adaptive) [{AUGMENT}]", [
        row_from_metrics("RF",       results["rf"]),
        row_from_metrics("XGBoost",  results["xgb"]),
        row_from_metrics("CNN-LSTM", results["cnn_lstm"]),
        row_from_metrics("GRU",      results["gru"]),
        row_from_metrics("CNN-GRU",  results["cnn_gru"]),
    ])

    return results


# =========================================================
# Recall 비교 출력
# =========================================================
def print_recall_comparison(stage1: dict, stage2: dict) -> None:
    print(f"\n[Recall 비교] CIC2017 내부 → CTU-13 교차검증  [{AUGMENT}]")
    print(f"{'Model':<12} {'CIC2017 test':>13} {'CTU13 Adaptive':>15}")
    print("-" * 42)
    for display, key in [
        ("RF", "rf"), ("XGBoost", "xgb"),
        ("CNN-LSTM", "cnn_lstm"), ("GRU", "gru"), ("CNN-GRU", "cnn_gru"),
    ]:
        cic = stage1.get(key, {}).get("recall", 0.0)
        ctu = stage2.get(key, {}).get("recall", 0.0)
        print(f"{display:<12} {cic:>13.4f} {ctu:>15.4f}")


# =========================================================
# 결과 저장
# =========================================================
def save_results(stage1: dict, stage2: dict) -> None:
    out = {
        "augment":            AUGMENT,
        "stage1_cic_test":    stage1,
        "stage2_ctu13_cross": stage2,
    }
    out_path = RESULT_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f"\n[SAVED] {out_path}")


# =========================================================
# 메인
# =========================================================
def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  evaluate.py — CIC2017 학습 / CTU-13 교차검증  [augment={AUGMENT}]")
    print("=" * 78)
    print(f"  MODEL_DIR  = {MODEL_DIR}")
    print(f"  RESULT_DIR = {RESULT_DIR}")
    print("=" * 78)

    stage1 = stage1_cic_test()
    stage2 = stage2_ctu13_cross()

    print_recall_comparison(stage1, stage2)
    save_results(stage1, stage2)

    print(f"\n[완료] {RESULT_DIR}/eval_results.json 저장됨")


if __name__ == "__main__":
    main()