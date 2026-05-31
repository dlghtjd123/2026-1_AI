"""
evaluate.py

K-fold 학습 후 최종 holdout test set 평가

사용법:
  python evaluate.py --dataset cicids2017
  python evaluate.py --dataset cicids2018
  python evaluate.py --dataset ctu13
  python evaluate.py --dataset cicids2017 --augment smote
  python evaluate.py --dataset cicids2017 --augment smote --augment_multiplier 5
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
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# 인자 파싱
# =========================================================
_parser = argparse.ArgumentParser()
_parser.add_argument("--dataset", type=str, default="cicids2017",
                     choices=["cicids2017", "cicids2018", "ctu13"])
_parser.add_argument("--augment", type=str, default="none",
                     choices=["none", "smote", "gan", "wgan_gp", "wcgan_gp"])
_parser.add_argument("--bot_ratio_factor", type=float, default=10.0,
                     help="학습 시 사용한 봇넷 비율 배수 (정보 표시용)")
_parser.add_argument("--augment_multiplier", type=float, default=2.0,
                     help="학습 시 사용한 증강 후 Bot 수 목표 배수 (기본값: 2)")
_args = _parser.parse_args()
DATASET          = _args.dataset
AUGMENT          = _args.augment
BOT_RATIO_FACTOR = _args.bot_ratio_factor
AUGMENT_MULTIPLIER = _args.augment_multiplier


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent
_ROOT     = _PROJECT.parent

_MUL_SUFFIX = "" if AUGMENT == "none" or AUGMENT_MULTIPLIER == 2.0 else f"_mul{AUGMENT_MULTIPLIER:g}"
_MODEL_SUFFIX = f"_{AUGMENT}{_MUL_SUFFIX}" if AUGMENT != "none" else ""

MODEL_DIR  = _ROOT / "artifacts" / f"models_{DATASET}{_MODEL_SUFFIX}"
RESULT_DIR = _ROOT / "artifacts" / f"results_{DATASET}{_MODEL_SUFFIX}"
DATA_ROOT  = _PROJECT / "data" / "processed"

DATASET_DIRS = {
    "cicids2017": (DATA_ROOT / "cicids2017" / "flat", DATA_ROOT / "cicids2017" / "seq"),
    "cicids2018": (DATA_ROOT / "cicids2018" / "flat", DATA_ROOT / "cicids2018" / "seq"),
    "ctu13":      (DATA_ROOT / "ctu13"      / "flat", DATA_ROOT / "ctu13"      / "seq"),
}

DATASET_DISPLAY = {
    "cicids2017": "CIC-IDS2017",
    "cicids2018": "CSE-CIC-IDS2018",
    "ctu13":      "CTU-13 Scenario 9",
}

MODEL_SUBDIRS = {
    "rf_flow":       "rf",
    "xgb_flow":      "xgb",
    "cnn_lstm_flow": "cnn_lstm",
    "gru_flow":      "gru",
    "cnn_gru_flow":  "cnn_gru",
}

MODEL_DISPLAY = {
    "rf": "RF", "xgb": "XGBoost",
    "cnn_lstm": "CNN-LSTM", "gru": "GRU", "cnn_gru": "CNN-GRU",
}


# =========================================================
# 모델 정의 — 2-class softmax (argmax 방식)
# =========================================================
class CNNLSTMModel(nn.Module):
    def __init__(self, n_features, conv_channels=64, lstm_hidden=64, dropout=0.3, n_classes=2):
        super().__init__()
        self.conv1   = nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm    = nn.LSTM(conv_channels, lstm_hidden, num_layers=1, batch_first=True)
        self.fc      = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(x).permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))   # (batch, n_classes)


class GRUModel(nn.Module):
    def __init__(self, n_features, gru_hidden=64, dropout=0.3, n_classes=2):
        super().__init__()
        self.gru     = nn.GRU(n_features, gru_hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(gru_hidden, n_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1]))   # (batch, n_classes)


class CNNGRUModel(nn.Module):
    def __init__(self, n_features, conv_channels=64, gru_hidden=64, dropout=0.3, n_classes=2):
        super().__init__()
        self.conv1   = nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gru     = nn.GRU(conv_channels, gru_hidden, num_layers=1, batch_first=True)
        self.fc      = nn.Linear(gru_hidden, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(x).permute(0, 2, 1)
        _, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1]))   # (batch, n_classes)


# =========================================================
# 지표 계산
# =========================================================
def compute_metrics(y_true, y_pred, y_prob) -> dict:
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


# =========================================================
# 모델 로드
# =========================================================
def _model_path(name: str, ext: str) -> Path:
    return MODEL_DIR / MODEL_SUBDIRS.get(name, "") / f"{name}.{ext}"

def _threshold_path(name: str) -> Path:
    return MODEL_DIR / MODEL_SUBDIRS.get(name, "") / f"{name}_threshold.json"

def load_sklearn_model(name: str):
    model = joblib.load(_model_path(name, "pkl"))
    thr_data = json.loads(_threshold_path(name).read_text())
    thr = thr_data["threshold"]
    thr = float(thr) if thr != "argmax" else 0.5
    return model, thr

def load_torch_checkpoint(name: str):
    """
    threshold가 "argmax" 또는 float 모두 처리.
    Returns: (ckpt_path, threshold_float_or_str)
    """
    ckpt_path = _model_path(name, "pt")
    thr_data  = json.loads(_threshold_path(name).read_text())
    thr       = thr_data["threshold"]   # "argmax" 또는 float
    return ckpt_path, thr

def predict_sklearn_probs(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]

def load_sequence_model(ckpt_path: Path, model_type: str, device: torch.device):
    ckpt   = torch.load(ckpt_path, map_location=device)
    n_feat = ckpt["n_features"]

    # 항상 fc.weight shape에서 직접 감지 (저장된 n_classes 키 무시)
    # 이유: 코드 변경 전후 호환 — 저장된 n_classes가 실제 가중치와 다를 수 있음
    fc_w      = ckpt.get("model_state_dict", {}).get("fc.weight")
    n_classes = int(fc_w.shape[0]) if fc_w is not None else ckpt.get("n_classes", 2)

    if model_type == "cnn_lstm":
        model = CNNLSTMModel(n_feat,
                             conv_channels=ckpt.get("conv_channels", 64),
                             lstm_hidden=ckpt.get("lstm_hidden", 64),
                             dropout=ckpt.get("dropout", 0.3),
                             n_classes=n_classes)
    elif model_type == "gru":
        model = GRUModel(n_feat,
                         gru_hidden=ckpt.get("gru_hidden", 64),
                         dropout=ckpt.get("dropout", 0.3),
                         n_classes=n_classes)
    elif model_type == "cnn_gru":
        model = CNNGRUModel(n_feat,
                            conv_channels=ckpt.get("conv_channels", 64),
                            gru_hidden=ckpt.get("gru_hidden", 64),
                            dropout=ckpt.get("dropout", 0.3),
                            n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), n_classes


def predict_sequence_probs(ckpt_path: Path, model_type: str, X: np.ndarray,
                           batch_size: int = 512) -> np.ndarray:
    """
    n_classes=2 → softmax[:, 1] (봇넷 확률)
    n_classes=1 → sigmoid (구버전 호환)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_classes = load_sequence_model(ckpt_path, model_type, device)

    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_b    = torch.tensor(X[start:start+batch_size], dtype=torch.float32).to(device)
            logits = model(X_b)
            if n_classes == 2:
                p = torch.softmax(logits, dim=1)[:, 1]   # 봇넷 클래스 확률
            else:
                p = torch.sigmoid(logits).squeeze(1)      # 구버전 sigmoid
            probs.append(p.cpu().numpy())

    return np.concatenate(probs)


# =========================================================
# 출력 테이블
# =========================================================
def print_result_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'='*88}")
    print(f"  {title}")
    print(f"  ★ 주 지표: F1, Recall  /  보조: ROC-AUC, Precision")
    print(f"{'='*88}")
    print(f"{'Model':<12} {'F1':>8} {'Recall':>8} {'Precision':>10} "
          f"{'ROC-AUC':>9} {'Accuracy':>10} {'Decision':>10}")
    print("-" * 88)
    for r in rows:
        f1  = f"{r['f1']:.4f}"        if r.get("f1")        is not None else "-"
        rec = f"{r['recall']:.4f}"    if r.get("recall")    is not None else "-"
        pre = f"{r['precision']:.4f}" if r.get("precision") is not None else "-"
        roc = f"{r['roc_auc']:.4f}"   if r.get("roc_auc")   is not None else "-"
        acc = f"{r['accuracy']:.4f}"  if r.get("accuracy")  is not None else "-"
        thr = r.get("decision", "argmax")
        print(f"{r['model']:<12} {f1:>8} {rec:>8} {pre:>10} "
              f"{roc:>9} {acc:>10} {str(thr):>10}")
    print("=" * 88)

def row_from_metrics(name: str, m: dict) -> dict:
    return {
        "model":     name,
        "f1":        m.get("f1"),
        "recall":    m.get("recall"),
        "precision": m.get("precision"),
        "roc_auc":   m.get("roc_auc"),
        "accuracy":  m.get("accuracy"),
        "decision":  m.get("decision", "argmax"),
    }


# =========================================================
# 평가
# =========================================================
def run_evaluation(flat_dir: Path, seq_dir: Path) -> dict:
    display_name = DATASET_DISPLAY[DATASET]
    print(f"\n[{display_name}] holdout test set 평가")

    X_flat = np.load(flat_dir / "X_test.npy")
    y_flat = np.load(flat_dir / "y_test.npy").astype(int)
    X_seq  = np.load(seq_dir  / "X_test.npy")
    y_seq  = np.load(seq_dir  / "y_test.npy").astype(int)

    print(f"  flat: {X_flat.shape}  Bot 비율: {y_flat.mean():.4f}")
    print(f"  seq:  {X_seq.shape}   Bot 비율: {y_seq.mean():.4f}")

    rf_model,  rf_thr           = load_sklearn_model("rf_flow")
    xgb_model, xgb_thr          = load_sklearn_model("xgb_flow")
    cnn_ckpt,  cnn_thr          = load_torch_checkpoint("cnn_lstm_flow")
    gru_ckpt,  gru_thr          = load_torch_checkpoint("gru_flow")
    cnn_gru_ckpt, cnn_gru_thr   = load_torch_checkpoint("cnn_gru_flow")

    def eval_sk(model, thr, X, y, name):
        prob  = predict_sklearn_probs(model, X)
        thr_f = float(thr) if thr != "argmax" else 0.5
        pred  = (prob >= thr_f).astype(int)
        m     = compute_metrics(y, pred, prob)
        m["decision"] = "argmax" if thr == "argmax" or thr_f == 0.5 else f"{thr_f:.2f}"
        return m

    def eval_seq(ckpt, mtype, thr, X, y):
        prob  = predict_sequence_probs(ckpt, mtype, X)
        thr_f = float(thr) if thr != "argmax" else 0.5
        pred  = (prob >= thr_f).astype(int)
        m     = compute_metrics(y, pred, prob)
        m["decision"] = "argmax" if thr == "argmax" else f"{thr_f:.2f}"
        return m

    results = {
        "rf":       eval_sk(rf_model,  rf_thr,  X_flat, y_flat, "rf"),
        "xgb":      eval_sk(xgb_model, xgb_thr, X_flat, y_flat, "xgb"),
        "cnn_lstm": eval_seq(cnn_ckpt,     "cnn_lstm", cnn_thr,     X_seq, y_seq),
        "gru":      eval_seq(gru_ckpt,     "gru",      gru_thr,     X_seq, y_seq),
        "cnn_gru":  eval_seq(cnn_gru_ckpt, "cnn_gru",  cnn_gru_thr, X_seq, y_seq),
    }

    print_result_table(
        f"{display_name} — Holdout Test [{AUGMENT}]",
        [row_from_metrics(MODEL_DISPLAY[k], results[k]) for k in results]
    )
    return results


# =========================================================
# 결과 저장
# =========================================================
def save_results(results: dict) -> None:
    out = {
        "dataset":          DATASET,
        "augment":          AUGMENT,
        "augment_multiplier": AUGMENT_MULTIPLIER,
        "primary_metric":   ["f1", "recall"],
        "secondary_metric": "roc_auc",
        "note":             "K-fold로 설정을 선택한 뒤 trainval 전체로 재학습한 최종 모델의 holdout test 평가",
        "test_results":     results,
    }
    out_path = RESULT_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f"\n[SAVED] {out_path}")


# =========================================================
# main
# =========================================================
def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  evaluate.py  [dataset={DATASET} / augment={AUGMENT}]")
    print("=" * 72)
    print(f"  MODEL_DIR        = {MODEL_DIR}")
    print(f"  RESULT_DIR       = {RESULT_DIR}")
    print(f"  학습 bot 비율    = 원본 × {BOT_RATIO_FACTOR:.0f}배 "
          f"({BOT_RATIO_FACTOR:.0f}x 배수)")
    print(f"  증강 목표        = Bot × {AUGMENT_MULTIPLIER:g}")
    print(f"  test set         = 원본 유지 (subsample 없음)")
    print(f"  ★ 주 지표 = F1-score, Recall  /  보조: ROC-AUC")
    print("=" * 72)

    flat_dir, seq_dir = DATASET_DIRS[DATASET]
    results = run_evaluation(flat_dir, seq_dir)
    save_results(results)
    print(f"\n[완료] {RESULT_DIR}/eval_results.json")


if __name__ == "__main__":
    main()
