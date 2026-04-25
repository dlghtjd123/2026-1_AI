"""
evaluate.py

1단계: CIC test 평가   — full 모델 (RF, XGBoost, CNN-LSTM) → cicids/winflat|seq test
2단계: CTU 교차검증    — full 모델                          → ctu13/scenario9

[CTU 교차검증 방식]
- scenario9만 사용 (scenario1은 봇넷 그룹 1개로 split 불균형 문제)
- preprocess_ctu13.py가 CICFlowMeter로 77개 feature 생성 + train/val/test 분리 + scaler 적용 완료
- val set으로 threshold 탐색, test set으로 최종 평가
- full 모델(77 features)을 그대로 사용
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


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR   = Path(__file__).resolve().parent
_PROJECT   = _SRC_DIR.parent
_ROOT      = _PROJECT.parent

MODEL_DIR  = _ROOT / "artifacts" / "models"
RESULT_DIR = _ROOT / "artifacts" / "results"
DATA_ROOT  = _PROJECT / "data" / "processed"

CIC_WINFLAT = DATA_ROOT / "cicids" / "winflat"
CIC_SEQ     = DATA_ROOT / "cicids" / "seq"
CTU_ROOT    = DATA_ROOT / "ctu13"

CTU_SCENARIOS = ["scenario9"]


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
# threshold 탐색
# =========================================================
def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    best_score     = None
    best_threshold = 0.5
    best_metrics   = None

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred  = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        score   = (
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
# RF / XGBoost 평가
# =========================================================
def evaluate_sklearn(model, threshold: float, X: np.ndarray, y: np.ndarray) -> dict:
    y_prob  = model.predict_proba(X)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y, y_pred, y_prob)
    metrics["selected_threshold"] = threshold
    return metrics


# =========================================================
# CNN-LSTM 평가
# ---------------------------------------------------------
# X_eval=None이면 X_tune으로 평가 (CIC test용)
# X_eval이 있으면 X_tune으로 threshold 탐색, X_eval로 평가 (CTU용)
# =========================================================
def evaluate_cnn_lstm(
    ckpt_path: Path,
    threshold: float,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    scaler_path: Path | None = None,
    batch_size: int = 512,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)

    if X_eval is None:
        X_eval, y_eval = X_tune, y_tune
        retune = False
    else:
        retune = True

    if scaler_path is not None and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        n, w, f = X_tune.shape
        X_tune = scaler.transform(X_tune.reshape(-1, f)).reshape(n, w, f)
        n, w, f = X_eval.shape
        X_eval = scaler.transform(X_eval.reshape(-1, f)).reshape(n, w, f)

    model = CNNLSTMModel(
        n_features=ckpt["n_features"],
        conv_channels=ckpt.get("conv_channels", 64),
        lstm_hidden=ckpt.get("lstm_hidden", 64),
        dropout=ckpt.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def _infer(X: np.ndarray) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                X_batch = torch.tensor(
                    X[start : start + batch_size], dtype=torch.float32
                ).to(device)
                probs.append(torch.sigmoid(model(X_batch)).cpu().numpy())
        return np.concatenate(probs)

    if retune:
        prob_tune    = _infer(X_tune)
        threshold, _ = pick_best_threshold(y_tune, prob_tune)

    prob_eval = _infer(X_eval)
    y_pred    = (prob_eval >= threshold).astype(int)
    metrics   = compute_metrics(y_eval, y_pred, prob_eval)
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
        "model":    model_name,
        "accuracy": m["accuracy"],
        "precision":m["precision"],
        "recall":   m["recall"],
        "f1":       m["f1"],
        "roc_auc":  m["roc_auc"],
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
            cnn_ckpt, cnn_thr,
            X_tune=X_seq, y_tune=y_seq,
            scaler_path=None,  # 이미 스케일됨
        ),
    }

    print_table("1단계: CIC test 평가", [
        row_from_metrics("RF (full)",       results["rf"]),
        row_from_metrics("XGBoost (full)",  results["xgb"]),
        row_from_metrics("CNN-LSTM (full)", results["cnn_lstm"]),
    ])
    return results


# =========================================================
# 2단계: CTU 교차검증 — scenario9 (full 모델, 77 features)
# ---------------------------------------------------------
# [threshold 방식]
# CTU val set으로 threshold sweep → test set으로 평가
# RF/XGBoost가 0이 나오는 이유를 확인하기 위해
# 확률 분포도 함께 저장한다.
# =========================================================
def stage2_ctu_cross() -> dict:
    print("\n[2단계] CTU 교차검증 — scenario9, full 모델 (77 features)")
    print("  [적용] CTU val set으로 threshold sweep, test set으로 평가")

    rf_model,  rf_thr  = load_sklearn_model("rf_full")
    xgb_model, xgb_thr = load_sklearn_model("xgb_full")
    cnn_ckpt,  cnn_thr = load_cnn_lstm("cnn_lstm_full")

    all_results = {}

    for scenario in CTU_SCENARIOS:
        print(f"\n  [{scenario}]")

        X_flat_val  = np.load(CTU_ROOT / scenario / "winflat" / "X_val.npy")
        y_flat_val  = np.load(CTU_ROOT / scenario / "winflat" / "y_val.npy").astype(int)
        X_flat_test = np.load(CTU_ROOT / scenario / "winflat" / "X_test.npy")
        y_flat_test = np.load(CTU_ROOT / scenario / "winflat" / "y_test.npy").astype(int)

        X_seq_val  = np.load(CTU_ROOT / scenario / "seq" / "X_val.npy")
        y_seq_val  = np.load(CTU_ROOT / scenario / "seq" / "y_val.npy").astype(int)
        X_seq_test = np.load(CTU_ROOT / scenario / "seq" / "X_test.npy")
        y_seq_test = np.load(CTU_ROOT / scenario / "seq" / "y_test.npy").astype(int)

        # ── 확률값 계산 ──────────────────────────────────
        rf_prob_val   = rf_model.predict_proba(X_flat_val)[:, 1]
        rf_prob_test  = rf_model.predict_proba(X_flat_test)[:, 1]

        xgb_prob_val  = xgb_model.predict_proba(X_flat_val)[:, 1]
        xgb_prob_test = xgb_model.predict_proba(X_flat_test)[:, 1]

        # ── 확률 분포 출력 (디버깅용) ────────────────────
        print(f"\n  [확률 분포 — val 봇넷 샘플]")
        rf_bot_probs  = rf_prob_val[y_flat_val == 1]
        xgb_bot_probs = xgb_prob_val[y_flat_val == 1]
        print(f"    RF  봇넷 확률: mean={rf_bot_probs.mean():.4f}  max={rf_bot_probs.max():.4f}  min={rf_bot_probs.min():.4f}")
        print(f"    XGB 봇넷 확률: mean={xgb_bot_probs.mean():.4f}  max={xgb_bot_probs.max():.4f}  min={xgb_bot_probs.min():.4f}")

        # ── CTU val 기준 threshold sweep ─────────────────
        rf_thr_ctu,  _ = pick_best_threshold(y_flat_val, rf_prob_val)
        xgb_thr_ctu, _ = pick_best_threshold(y_flat_val, xgb_prob_val)
        print(f"\n  [CTU threshold sweep 결과]")
        print(f"    RF  최적 threshold: {rf_thr_ctu:.2f}  (CIC: {rf_thr:.2f})")
        print(f"    XGB 최적 threshold: {xgb_thr_ctu:.2f}  (CIC: {xgb_thr:.2f})")

        # ── test 평가 ────────────────────────────────────
        rf_pred_test  = (rf_prob_test  >= rf_thr_ctu).astype(int)
        xgb_pred_test = (xgb_prob_test >= xgb_thr_ctu).astype(int)

        rf_metrics  = compute_metrics(y_flat_test, rf_pred_test,  rf_prob_test)
        xgb_metrics = compute_metrics(y_flat_test, xgb_pred_test, xgb_prob_test)
        rf_metrics["selected_threshold"]  = rf_thr_ctu
        xgb_metrics["selected_threshold"] = xgb_thr_ctu

        # CNN-LSTM: CTU val로 threshold 탐색
        cnn_metrics = evaluate_cnn_lstm(
            cnn_ckpt, cnn_thr,
            X_tune=X_seq_val,  y_tune=y_seq_val,
            X_eval=X_seq_test, y_eval=y_seq_test,
            scaler_path=None,
        )

        # ── 확률 분포 저장 (visualize.py에서 사용) ───────
        prob_dist = {
            "rf": {
                "val_botnet":  rf_prob_val[y_flat_val == 1].tolist(),
                "val_normal":  rf_prob_val[y_flat_val == 0].tolist(),
                "test_botnet": rf_prob_test[y_flat_test == 1].tolist(),
                "test_normal": rf_prob_test[y_flat_test == 0].tolist(),
                "threshold_ctu": float(rf_thr_ctu),
                "threshold_cic": float(rf_thr),
            },
            "xgb": {
                "val_botnet":  xgb_prob_val[y_flat_val == 1].tolist(),
                "val_normal":  xgb_prob_val[y_flat_val == 0].tolist(),
                "test_botnet": xgb_prob_test[y_flat_test == 1].tolist(),
                "test_normal": xgb_prob_test[y_flat_test == 0].tolist(),
                "threshold_ctu": float(xgb_thr_ctu),
                "threshold_cic": float(xgb_thr),
            },
        }

        prob_path = RESULT_DIR / f"prob_dist_{scenario}.json"
        with open(prob_path, "w") as f:
            json.dump(prob_dist, f)
        print(f"\n  [SAVED] 확률 분포: {prob_path}")

        results = {
            "rf":       rf_metrics,
            "xgb":      xgb_metrics,
            "cnn_lstm": cnn_metrics,
        }

        print_table(f"2단계: CTU 교차검증 — {scenario}", [
            row_from_metrics("RF (full)",       results["rf"]),
            row_from_metrics("XGBoost (full)",  results["xgb"]),
            row_from_metrics("CNN-LSTM (full)", results["cnn_lstm"]),
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