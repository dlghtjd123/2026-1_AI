"""
evaluate.py

1단계: CIC test 평가       — full 모델 (RF, XGBoost, CNN-LSTM) → cicids/winflat|seq test
2단계: CTU 교차검증        — common 모델                        → ctu13/scenario1, scenario9

[CTU 교차검증 방식]
① CTU 데이터를 CTU 자체 StandardScaler로 정규화
   - CIC scaler로 CTU를 변환하면 분포 불일치로 성능 급락
   - CTU 데이터 자체를 mean=0, std=1로 맞추면 모델이 학습한 스케일과 유사해짐
   - 논문 서술: "CTU 데이터를 독립적으로 정규화한 뒤 평가"

② CTU에서 threshold 재탐색
   - CIC val에서 최적화된 threshold는 CTU 분포에서 맞지 않을 수 있음
   - CTU 데이터 기준으로 threshold를 다시 찾아 최대 성능을 측정
   - 논문 서술: "CTU 데이터 기준 최적 threshold 적용"
"""

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
from sklearn.preprocessing import StandardScaler


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR   = Path(__file__).resolve().parent
_PROJECT   = _SRC_DIR.parent
_ROOT      = _PROJECT.parent

MODEL_DIR  = _ROOT / "artifacts" / "models"
RESULT_DIR = _ROOT / "artifacts" / "results"
DATA_ROOT  = _PROJECT / "data" / "processed"

CIC_WINFLAT        = DATA_ROOT / "cicids" / "winflat"
CIC_SEQ            = DATA_ROOT / "cicids" / "seq"
CIC_WINFLAT_COMMON = DATA_ROOT / "cicids" / "winflat_common"
CIC_SEQ_COMMON     = DATA_ROOT / "cicids" / "seq_common"

CTU_ROOT     = DATA_ROOT / "ctu13"
CTU_SCENARIOS = ["scenario1", "scenario9"]


# =========================================================
# CNN-LSTM 모델 정의 (train_cnn_lstm.py와 동일)
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
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm    = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        return self.fc(x).squeeze(1)


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
# threshold 재탐색
# =========================================================
def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    """F1 → Recall → Precision 순으로 최적 threshold를 탐색한다."""
    best_score     = None
    best_threshold = 0.5
    best_metrics   = None

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred   = (y_prob >= threshold).astype(int)
        metrics  = compute_metrics(y_true, y_pred, y_prob)
        score    = (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            -abs(threshold - 0.5),
        )
        if best_score is None or score > best_score:
            best_score     = score
            best_threshold = float(round(threshold, 4))
            best_metrics   = metrics

    best_metrics["selected_threshold"] = best_threshold
    return best_threshold, best_metrics


# =========================================================
# CTU 데이터 정규화 (CTU 자체 scaler)
# ---------------------------------------------------------
# CIC scaler를 CTU에 적용하면 분포 불일치로 성능 급락.
# CTU 데이터를 자체적으로 StandardScaler로 정규화하면
# 모델이 학습한 스케일(mean≈0, std≈1)과 유사해진다.
# =========================================================
def normalize_ctu(X: np.ndarray) -> np.ndarray:
    """CTU 데이터를 자체 StandardScaler로 정규화한다."""
    is_3d = X.ndim == 3
    if is_3d:
        n, w, f = X.shape
        X_2d = X.reshape(-1, f)
    else:
        X_2d = X

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)

    return X_scaled.reshape(n, w, f).astype(np.float32) if is_3d else X_scaled.astype(np.float32)


# =========================================================
# RF / XGBoost 평가
# =========================================================
def evaluate_sklearn(
    model,
    threshold: float,
    X: np.ndarray,
    y: np.ndarray,
    retune_threshold: bool = False,
) -> dict:
    y_prob = model.predict_proba(X)[:, 1]

    if retune_threshold:
        threshold, metrics = pick_best_threshold(y, y_prob)
    else:
        y_pred  = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y, y_pred, y_prob)
        metrics["selected_threshold"] = threshold

    return metrics


# =========================================================
# CNN-LSTM 평가
# =========================================================
def evaluate_cnn_lstm(
    ckpt_path: Path,
    threshold: float,
    X: np.ndarray,
    y: np.ndarray,
    scaler_path: Path | None = None,
    normalize_self: bool = False,
    retune_threshold: bool = False,
    batch_size: int = 512,
) -> dict:
    """
    Parameters
    ----------
    scaler_path     : CIC scaler 경로. None이면 적용 안 함 (CIC test는 이미 스케일됨).
    normalize_self  : True면 데이터 자체 StandardScaler 적용 (CTU 교차검증용).
    retune_threshold: True면 해당 데이터 기준으로 threshold 재탐색.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)

    # CIC scaler 적용 (CIC test에는 사용 안 함 — 이미 스케일됨)
    if scaler_path is not None and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        n, w, f = X.shape
        X = scaler.transform(X.reshape(-1, f)).reshape(n, w, f)

    # CTU 자체 scaler 적용
    if normalize_self:
        X = normalize_ctu(X)

    model = CNNLSTMModel(
        n_features=ckpt["n_features"],
        conv_channels=ckpt.get("conv_channels", 64),
        lstm_hidden=ckpt.get("lstm_hidden", 64),
        dropout=ckpt.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 배치 단위 추론 (OOM 방지)
    y_prob_list = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_batch = torch.tensor(
                X[start : start + batch_size], dtype=torch.float32
            ).to(device)
            probs = torch.sigmoid(model(X_batch)).cpu().numpy()
            y_prob_list.append(probs)

    y_prob = np.concatenate(y_prob_list)

    if retune_threshold:
        threshold, metrics = pick_best_threshold(y, y_prob)
    else:
        y_pred  = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y, y_pred, y_prob)
        metrics["selected_threshold"] = threshold

    return metrics


# =========================================================
# 비교표 출력
# =========================================================
def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 70)
    for r in rows:
        roc = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "   None"
        print(
            f"{r['model']:<20} "
            f"{r['accuracy']:>9.4f} "
            f"{r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{roc:>9}"
        )
    print("=" * 70)


def row_from_metrics(model_name: str, m: dict) -> dict:
    return {
        "model":     model_name,
        "accuracy":  m["accuracy"],
        "precision": m["precision"],
        "recall":    m["recall"],
        "f1":        m["f1"],
        "roc_auc":   m["roc_auc"],
    }


# =========================================================
# 모델 & threshold 로드 헬퍼
# =========================================================
def load_sklearn_model(name: str):
    model     = joblib.load(MODEL_DIR / f"{name}.pkl")
    threshold = json.loads((MODEL_DIR / f"{name}_threshold.json").read_text())["threshold"]
    return model, threshold


def load_cnn_lstm(name: str):
    ckpt_path = MODEL_DIR / f"{name}.pt"
    threshold = json.loads((MODEL_DIR / f"{name}_threshold.json").read_text())["threshold"]
    return ckpt_path, threshold


# =========================================================
# 1단계: CIC test 평가 (full 모델)
# =========================================================
def stage1_cic_test() -> dict:
    print("\n[1단계] CIC test 평가 — full 모델 (77 features)")

    X_flat = np.load(CIC_WINFLAT / "X_test.npy")
    y_flat = np.load(CIC_WINFLAT / "y_test.npy").astype(int)
    X_seq  = np.load(CIC_SEQ / "X_test.npy")
    y_seq  = np.load(CIC_SEQ / "y_test.npy").astype(int)

    rf_model,  rf_thr  = load_sklearn_model("rf_full")
    xgb_model, xgb_thr = load_sklearn_model("xgb_full")
    cnn_ckpt,  cnn_thr = load_cnn_lstm("cnn_lstm_full")

    results = {
        "rf":  evaluate_sklearn(rf_model,  rf_thr,  X_flat, y_flat),
        "xgb": evaluate_sklearn(xgb_model, xgb_thr, X_flat, y_flat),
        "cnn_lstm": evaluate_cnn_lstm(
            cnn_ckpt, cnn_thr, X_seq, y_seq,
            scaler_path=None,      # X_test.npy는 전처리 시 이미 스케일됨
            normalize_self=False,
            retune_threshold=False,
        ),
    }

    print_table("1단계: CIC test 평가", [
        row_from_metrics("RF (full)",       results["rf"]),
        row_from_metrics("XGBoost (full)",  results["xgb"]),
        row_from_metrics("CNN-LSTM (full)", results["cnn_lstm"]),
    ])
    return results


# =========================================================
# 2단계: CTU 교차검증 (common 모델)
# ---------------------------------------------------------
# normalize_self=True  : CTU 데이터 자체 scaler로 정규화
# retune_threshold=True: CTU 데이터 기준 threshold 재탐색
# =========================================================
def stage2_ctu_cross() -> dict:
    print("\n[2단계] CTU 교차검증 — common 모델 (8 features)")
    print("  [적용] CTU 자체 정규화 + threshold 재탐색")

    rf_model,  rf_thr  = load_sklearn_model("rf_common")
    xgb_model, xgb_thr = load_sklearn_model("xgb_common")
    cnn_ckpt,  cnn_thr = load_cnn_lstm("cnn_lstm_common")

    all_results = {}

    for scenario in CTU_SCENARIOS:
        print(f"\n  [{scenario}]")

        X_flat = np.load(CTU_ROOT / scenario / "winflat" / "X.npy")
        y_flat = np.load(CTU_ROOT / scenario / "winflat" / "y.npy").astype(int)
        X_seq  = np.load(CTU_ROOT / scenario / "seq"     / "X.npy")
        y_seq  = np.load(CTU_ROOT / scenario / "seq"     / "y.npy").astype(int)

        # RF / XGBoost: 트리 모델은 scaler 불필요, threshold만 재탐색
        X_flat_norm = normalize_ctu(X_flat)

        results = {
            "rf":  evaluate_sklearn(
                rf_model,  rf_thr,  X_flat_norm, y_flat,
                retune_threshold=True,
            ),
            "xgb": evaluate_sklearn(
                xgb_model, xgb_thr, X_flat_norm, y_flat,
                retune_threshold=True,
            ),
            "cnn_lstm": evaluate_cnn_lstm(
                cnn_ckpt, cnn_thr, X_seq, y_seq,
                scaler_path=None,
                normalize_self=True,   # CTU 자체 scaler 적용
                retune_threshold=True, # CTU 기준 threshold 재탐색
            ),
        }

        print_table(f"2단계: CTU 교차검증 — {scenario}", [
            row_from_metrics("RF (common)",       results["rf"]),
            row_from_metrics("XGBoost (common)",  results["xgb"]),
            row_from_metrics("CNN-LSTM (common)", results["cnn_lstm"]),
        ])

        all_results[scenario] = results

    return all_results


# =========================================================
# 결과 저장
# =========================================================
def save_results(stage1: dict, stage2: dict) -> None:
    out = {
        "stage1_cic_test":  stage1,
        "stage2_ctu_cross": stage2,
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

    print("=" * 70)
    print("  evaluate.py — 성능 비교 평가")
    print("=" * 70)

    stage1_results = stage1_cic_test()
    stage2_results = stage2_ctu_cross()

    save_results(stage1_results, stage2_results)

    print("\n[완료] artifacts/results/eval_results.json 저장됨")
    print("\n[다음 단계]")
    print("  증강 기법 비교 (SMOTE / GAN / WGAN / WGAN-GP)")
    print("  → 증강 후 best 모델 재학습 → evaluate.py 재실행")


if __name__ == "__main__":
    main()