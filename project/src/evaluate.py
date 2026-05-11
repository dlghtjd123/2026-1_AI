"""
evaluate.py

1단계: CIC-IDS2017 내부 test 평가
2단계: CSE-CIC-IDS2018 교차검증
3단계: CTU-13 시나리오 9 교차검증

주 평가 지표: ROC-AUC (threshold-independent)
  → 클래스 불균형 + 교차 데이터셋 환경에서 표준 지표
  → threshold 선택과 무관하게 모델 구별 능력 측정
  (Transformer-IDS, JCS 2025; IoT Botnet arXiv:2104.02231)

보조 지표: Accuracy / Precision / Recall / F1
  → target dataset에서 Youden's J로 threshold를 보정한 뒤 계산
  → 최적 threshold에서의 실제 탐지 성능
  → Safety 2025 방식 (target dataset 통계 사용, 논문 명시 필요)

사용법:
  python evaluate.py
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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
)
AUGMENT = _parser.parse_args().augment


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent
_ROOT     = _PROJECT.parent

_SUFFIX   = f"_{AUGMENT}" if AUGMENT != "none" else ""

MODEL_DIR  = _ROOT / "artifacts" / f"models{_SUFFIX}"
RESULT_DIR = _ROOT / "artifacts" / f"results{_SUFFIX}"
DATA_ROOT  = _PROJECT / "data" / "processed"

CIC_FLAT   = DATA_ROOT / "cicids2017" / "flat"
CIC_SEQ    = DATA_ROOT / "cicids2017" / "seq"
CIC18_FLAT = DATA_ROOT / "cicids2018" / "flat"
CIC18_SEQ  = DATA_ROOT / "cicids2018" / "seq"
CTU_FLAT   = DATA_ROOT / "ctu13" / "flat"
CTU_SEQ    = DATA_ROOT / "ctu13" / "seq"

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


def find_youdens_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden's J = TPR - FPR 최대화"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


# =========================================================
# 모델 로드
# =========================================================
def _model_path(name: str, ext: str) -> Path:
    return MODEL_DIR / MODEL_SUBDIRS.get(name, "") / f"{name}.{ext}"


def _threshold_path(name: str) -> Path:
    return MODEL_DIR / MODEL_SUBDIRS.get(name, "") / f"{name}_threshold.json"


def load_sklearn_model(name: str):
    model     = joblib.load(_model_path(name, "pkl"))
    threshold = json.loads(_threshold_path(name).read_text())["threshold"]
    return model, float(threshold)


def load_torch_model(name: str):
    ckpt_path = _model_path(name, "pt")
    threshold = json.loads(_threshold_path(name).read_text())["threshold"]
    return ckpt_path, float(threshold)


def predict_sklearn_probs(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def load_sequence_model(ckpt_path, model_type, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if model_type == "cnn_lstm":
        model = CNNLSTMModel(
            n_features=ckpt["n_features"],
            conv_channels=ckpt.get("conv_channels", 64),
            lstm_hidden=ckpt.get("lstm_hidden", 64),
            dropout=ckpt.get("dropout", 0.3),
        )
    elif model_type == "gru":
        model = GRUModel(
            n_features=ckpt["n_features"],
            gru_hidden=ckpt.get("gru_hidden", 64),
            dropout=ckpt.get("dropout", 0.3),
        )
    elif model_type == "cnn_gru":
        model = CNNGRUModel(
            n_features=ckpt["n_features"],
            conv_channels=ckpt.get("conv_channels", 64),
            gru_hidden=ckpt.get("gru_hidden", 64),
            dropout=ckpt.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def predict_sequence_probs(ckpt_path, model_type, X, batch_size=512) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_sequence_model(ckpt_path, model_type, device)
    probs  = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_b = torch.tensor(
                X[start:start+batch_size], dtype=torch.float32
            ).to(device)
            probs.append(torch.sigmoid(model(X_b)).cpu().numpy())
    return np.concatenate(probs)


def get_all_probs(X_flat, X_seq) -> dict:
    rf_model,     _ = load_sklearn_model("rf_flow")
    xgb_model,    _ = load_sklearn_model("xgb_flow")
    cnn_ckpt,     _ = load_torch_model("cnn_lstm_flow")
    gru_ckpt,     _ = load_torch_model("gru_flow")
    cnn_gru_ckpt, _ = load_torch_model("cnn_gru_flow")
    return {
        "rf":       predict_sklearn_probs(rf_model,  X_flat),
        "xgb":      predict_sklearn_probs(xgb_model, X_flat),
        "cnn_lstm": predict_sequence_probs(cnn_ckpt,     "cnn_lstm", X_seq),
        "gru":      predict_sequence_probs(gru_ckpt,     "gru",      X_seq),
        "cnn_gru":  predict_sequence_probs(cnn_gru_ckpt, "cnn_gru",  X_seq),
    }


# =========================================================
# 출력 테이블 — ROC-AUC 첫 번째 (주 지표)
# =========================================================
def print_roc_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'='*92}")
    print(f"  {title}")
    print("  ★ 주 지표: ROC-AUC / 보조 지표: Accuracy, Precision, Recall, F1")
    print(f"{'='*92}")

    print(
        f"{'Model':<22} {'ROC-AUC':>9} {'Threshold':>10} "
        f"{'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    )
    print("-" * 92)

    for r in rows:
        roc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "-"
        thr = f"{r['threshold']:.4f}" if r.get("threshold") is not None else "-"
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "-"
        pre = f"{r['precision']:.4f}" if r.get("precision") is not None else "-"
        rec = f"{r['recall']:.4f}" if r.get("recall") is not None else "-"
        f1 = f"{r['f1']:.4f}" if r.get("f1") is not None else "-"

        print(
            f"{r['model']:<22} {roc:>9} {thr:>10} "
            f"{acc:>10} {pre:>10} {rec:>8} {f1:>8}"
        )

    print("=" * 92)

def row_from_metrics(name: str, m: dict, thr: float = None) -> dict:
    return {
        "model": name,
        "roc_auc": m.get("roc_auc"),
        "threshold": thr if thr is not None else m.get("threshold"),
        "accuracy": m.get("accuracy"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1"),
    }


# =========================================================
# 1단계: CIC-IDS2017 내부 test
# =========================================================
def stage1_cic_test() -> dict:
    print("\n[1단계] CIC-IDS2017 내부 test 평가")

    X_flat = np.load(CIC_FLAT / "X_test.npy")
    y_flat = np.load(CIC_FLAT / "y_test.npy").astype(int)
    X_seq  = np.load(CIC_SEQ  / "X_test.npy")
    y_seq  = np.load(CIC_SEQ  / "y_test.npy").astype(int)
    print(f"  flat: {X_flat.shape}  Bot 비율: {y_flat.mean():.4f}")

    rf_model,     rf_thr      = load_sklearn_model("rf_flow")
    xgb_model,    xgb_thr     = load_sklearn_model("xgb_flow")
    cnn_ckpt,     cnn_thr     = load_torch_model("cnn_lstm_flow")
    gru_ckpt,     gru_thr     = load_torch_model("gru_flow")
    cnn_gru_ckpt, cnn_gru_thr = load_torch_model("cnn_gru_flow")

    def eval_sk(model, thr, X, y):
        prob = predict_sklearn_probs(model, X)
        m    = compute_metrics(y, (prob >= thr).astype(int), prob)
        m["threshold"] = thr
        return m

    def eval_seq(ckpt, mtype, thr, X, y):
        prob = predict_sequence_probs(ckpt, mtype, X)
        m    = compute_metrics(y, (prob >= thr).astype(int), prob)
        m["threshold"] = thr
        return m

    results = {
        "rf":       eval_sk(rf_model,  rf_thr,  X_flat, y_flat),
        "xgb":      eval_sk(xgb_model, xgb_thr, X_flat, y_flat),
        "cnn_lstm": eval_seq(cnn_ckpt,     "cnn_lstm", cnn_thr,     X_seq, y_seq),
        "gru":      eval_seq(gru_ckpt,     "gru",      gru_thr,     X_seq, y_seq),
        "cnn_gru":  eval_seq(cnn_gru_ckpt, "cnn_gru",  cnn_gru_thr, X_seq, y_seq),
    }

    print_roc_table(f"1단계: CIC-IDS2017 내부 test [{AUGMENT}]", [
        row_from_metrics("RF",       results["rf"]),
        row_from_metrics("XGBoost",  results["xgb"]),
        row_from_metrics("CNN-LSTM", results["cnn_lstm"]),
        row_from_metrics("GRU",      results["gru"]),
        row_from_metrics("CNN-GRU",  results["cnn_gru"]),
    ])
    return results


# =========================================================
# 교차검증 공통 — ROC-AUC + Youden's J F1
# =========================================================
def evaluate_cross(probs: dict, labels: dict) -> dict:
    results = {}
    for key, prob in probs.items():
        y = labels[key]

        # ROC-AUC (주 지표)
        try:
            roc_auc = float(roc_auc_score(y, prob))
        except ValueError:
            roc_auc = None

        # Youden's J threshold → F1 (보조 지표)
        thr_j = find_youdens_j_threshold(y, prob)
        pred_j = (prob >= thr_j).astype(int)
        m_j    = compute_metrics(y, pred_j, prob)
        m_j["threshold"] = thr_j
        m_j["roc_auc"]   = roc_auc

        results[key] = m_j
    return results


# =========================================================
# 2단계: CSE-CIC-IDS2018 교차검증
# ---------------------------------------------------------
# D'Hooge et al. (2020) 동일 설정
# 봇넷 유형: CIC2017(Neris IRC) → CIC2018(Ares+Zeus HTTP)
# 주 지표: ROC-AUC / 보조: F1@Youden's J
# =========================================================
def stage2_cic2018_cross() -> dict:
    print("\n[2단계] CSE-CIC-IDS2018 교차검증")
    print("  D'Hooge et al. (2020) 동일 설정")
    print("  봇넷: CIC2017(Neris IRC) → CIC2018(Ares+Zeus HTTP)")

    if not (CIC18_FLAT / "X_test.npy").exists():
        print("  [SKIP] CIC2018 데이터 없음 → preprocess_cicids2018.py 먼저 실행")
        return {}

    X_flat = np.load(CIC18_FLAT / "X_test.npy")
    y_flat = np.load(CIC18_FLAT / "y_test.npy").astype(int)
    X_seq  = np.load(CIC18_SEQ  / "X_test.npy")
    y_seq  = np.load(CIC18_SEQ  / "y_test.npy").astype(int)
    print(f"  flat: {X_flat.shape}  Bot 비율: {y_flat.mean():.4f}")

    probs  = get_all_probs(X_flat, X_seq)
    labels = {
        "rf": y_flat, "xgb": y_flat,
        "cnn_lstm": y_seq, "gru": y_seq, "cnn_gru": y_seq,
    }
    results = evaluate_cross(probs, labels)

    print_roc_table(
        f"2단계: CIC-IDS2018 교차검증 (Target-adapted Youden's J) [{AUGMENT}]",
        [row_from_metrics(MODEL_DISPLAY[k], results[k]) for k in results]
    )
    return results


# =========================================================
# 3단계: CTU-13 시나리오 9 교차검증
# ---------------------------------------------------------
# Safety 2025 방식
# 봇넷: CIC2017(Neris IRC 1bot) → CTU-13(Neris IRC 10bots)
# 주 지표: ROC-AUC / 보조: F1@Youden's J
# =========================================================
def stage3_ctu13_cross() -> dict:
    print("\n[3단계] CTU-13 시나리오 9 교차검증")
    print("  Safety 2025 방식 (MinMaxScaler + Secondary MinMaxScaler)")
    print("  봇넷: CIC2017(Neris IRC 1bot) → CTU-13(Neris IRC 10bots)")

    X_flat = np.load(CTU_FLAT / "X_test.npy")
    y_flat = np.load(CTU_FLAT / "y_test.npy").astype(int)
    X_seq  = np.load(CTU_SEQ  / "X_test.npy")
    y_seq  = np.load(CTU_SEQ  / "y_test.npy").astype(int)
    print(f"  flat: {X_flat.shape}  Bot 비율: {y_flat.mean():.4f}")

    probs  = get_all_probs(X_flat, X_seq)
    labels = {
        "rf": y_flat, "xgb": y_flat,
        "cnn_lstm": y_seq, "gru": y_seq, "cnn_gru": y_seq,
    }
    results = evaluate_cross(probs, labels)

    print_roc_table(
        f"3단계: CTU-13 교차검증 (Target-adapted Youden's J) [{AUGMENT}]",
        [row_from_metrics(MODEL_DISPLAY[k], results[k]) for k in results]
    )
    return results


# =========================================================
# ROC-AUC 요약 — 핵심 비교표
# =========================================================
def print_roc_auc_summary(stage1: dict, stage2: dict, stage3: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  ★ ROC-AUC 요약 (주 지표)  [{AUGMENT}]")
    print(f"  ROC-AUC: 0.5=랜덤 / 0.7=양호 / 0.9=우수 / 1.0=완벽")
    print(f"{'='*65}")
    print(f"{'Model':<12} {'CIC2017':>9} {'CIC2018':>9} {'CTU-13':>9}")
    print("-" * 42)

    for key, display in MODEL_DISPLAY.items():
        c = stage1.get(key, {}).get("roc_auc")
        c18 = stage2.get(key, {}).get("roc_auc")
        ctu = stage3.get(key, {}).get("roc_auc")

        c_str = f"{c:.4f}" if isinstance(c, float) else "-"
        c18_str = f"{c18:.4f}" if isinstance(c18, float) else "-"
        ctu_str = f"{ctu:.4f}" if isinstance(ctu, float) else "-"

        print(f"{display:<12} {c_str:>9} {c18_str:>9} {ctu_str:>9}")

    print("=" * 65)
    print("  CIC2017: 내부 평가")
    print("  CIC2018, CTU-13: target-adapted cross-dataset evaluation")


# =========================================================
# 결과 저장
# =========================================================
def save_results(stage1: dict, stage2: dict, stage3: dict) -> None:
    out = {
        "augment":             AUGMENT,
        "primary_metric":      "ROC-AUC (threshold-independent)",
        "secondary_metric":    "F1 @ Youden's J threshold (Safety 2025)",
        "scaler":              "MinMaxScaler + Secondary MinMaxScaler",
        "references": {
            "roc_auc":   "Transformer-IDS, JCS 2025; arXiv:2104.02231",
            "threshold": "de Nascimento & Hou, Safety 2025",
            "scaler":    "D'Hooge et al., JISA 2020",
        },
        "stage1_cic_test":      stage1,
        "stage2_cic2018_cross": stage2,
        "stage3_ctu13_cross":   stage3,
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

    print("=" * 72)
    print(f"  evaluate.py  [augment={AUGMENT}]")
    print("=" * 72)
    print(f"  MODEL_DIR      = {MODEL_DIR}")
    print(f"  RESULT_DIR     = {RESULT_DIR}")
    print(f"  ★ 주 지표     = ROC-AUC (threshold-independent)")
    print(f"  보조 지표     = F1 @ Youden's J (Safety 2025)")
    print(f"  Scaler        = MinMaxScaler + Secondary MinMaxScaler")
    print("=" * 72)

    stage1 = stage1_cic_test()
    stage2 = stage2_cic2018_cross()
    stage3 = stage3_ctu13_cross()

    print_roc_auc_summary(stage1, stage2, stage3)
    save_results(stage1, stage2, stage3)
    print(f"\n[완료] {RESULT_DIR}/eval_results.json")


if __name__ == "__main__":
    main()