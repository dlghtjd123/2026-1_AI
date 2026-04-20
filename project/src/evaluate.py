from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# 1. 설정
# =========================================================

@dataclass
class EvalConfig:
    ctu_scenario_dirs: list[Path]
    model_dir: Path = Path("artifacts/models")
    result_dir: Path = Path("artifacts/results/ctu13_external")
    default_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "rf": 0.21,
            "xgb": 0.10,
            "cnn_lstm": 0.50,
        }
    )

    # 중요:
    # CTU npy가 이미 CIC scaler로 transform된 상태라면 False 유지
    apply_winflat_scaler: bool = False
    apply_seq_scaler: bool = False

    winflat_scaler_path: Path = Path("artifacts/models/scaler_winflat.pkl")
    seq_scaler_path: Path = Path("artifacts/models/scaler_seq_w15.pkl")

    rf_model_path: Path = Path("artifacts/models/rf_model.pkl")
    xgb_model_path: Path = Path("artifacts/models/xgb_model.pkl")
    cnn_lstm_model_path: Path = Path("artifacts/models/cnn_lstm_best.pt")

    cnn_batch_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 중요:
    # 네 CNN-LSTM의 실제 출력 형태에 맞게 반드시 설정
    # - "logit_binary"     : shape (N,) 또는 (N,1), raw logits
    # - "prob_binary"      : shape (N,) 또는 (N,1), 이미 sigmoid 확률
    # - "logit_multiclass" : shape (N,2), raw logits
    # - "prob_multiclass"  : shape (N,2), 이미 softmax 확률
    cnn_output_mode: str = "logit_binary"

    # 중요:
    # True일 때 torch.load(..., weights_only=False) 경로를 허용
    # 반드시 "신뢰 가능한 로컬 모델 파일"만 사용해야 함
    trusted_model_files: bool = True


CONFIG = EvalConfig(
    ctu_scenario_dirs=[
        Path("data/processed/ctu13/scenario1"),
        Path("data/processed/ctu13/scenario9"),
    ]
)

CONFIG.result_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 데이터 구조
# =========================================================

@dataclass
class ScenarioBundle:
    name: str

    X_winflat: np.ndarray
    y_winflat: np.ndarray
    X_seq: np.ndarray
    y_seq: np.ndarray

    winflat_x_path: Path
    winflat_y_path: Path
    seq_x_path: Path
    seq_y_path: Path

    winflat_sample_ids: np.ndarray | None = None
    seq_sample_ids: np.ndarray | None = None
    winflat_sample_ids_path: Path | None = None
    seq_sample_ids_path: Path | None = None


# =========================================================
# 3. JSON 직렬화 유틸
# =========================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict[str, Any], save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


# =========================================================
# 4. 파일 로드 유틸
# =========================================================

def find_existing_file(candidates: list[Path]) -> Path:
    """
    후보 파일 중 존재하는 첫 번째 파일을 반환.
    첫 번째 우선순위 파일이 아닌 경우 경고 로그 출력.
    """
    if not candidates:
        raise ValueError("find_existing_file()에 빈 후보 리스트가 전달되었습니다.")

    for idx, path in enumerate(candidates):
        if path.exists():
            if idx > 0:
                print(f"[WARN] 기본 경로 대신 대체 파일 사용: {path}")
            return path

    raise FileNotFoundError(
        "다음 후보 경로들 중 어떤 파일도 찾지 못했습니다:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_numpy_pair(
    base_dir: Path,
    subdir: str,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    """
    winflat / seq 폴더에서 X, y 로드.
    파일명은 여러 형태를 허용.
    """
    target_dir = base_dir / subdir

    x_path = find_existing_file([
        target_dir / "X.npy",
        target_dir / "X_test.npy",
        target_dir / "X_ctu.npy",
    ])
    y_path = find_existing_file([
        target_dir / "y.npy",
        target_dir / "y_test.npy",
        target_dir / "y_ctu.npy",
    ])

    X = np.load(x_path)
    y = np.load(y_path).astype(int)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"[{base_dir.name}/{subdir}] X와 y 샘플 수 불일치: "
            f"X={X.shape[0]}, y={y.shape[0]}"
        )

    return X, y, x_path, y_path


def load_optional_sample_ids(base_dir: Path, subdir: str) -> tuple[np.ndarray | None, Path | None]:
    """
    샘플 정합성 검증용 선택 파일 로드.
    있으면 사용하고, 없으면 None 반환.
    """
    target_dir = base_dir / subdir
    candidates = [
        target_dir / "sample_ids.npy",
        target_dir / "sample_id.npy",
        target_dir / "indices.npy",
        target_dir / "index.npy",
        target_dir / "original_indices.npy",
        target_dir / "original_idx.npy",
    ]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None, None

    if len(existing) > 1:
        print(f"[WARN] {target_dir} 에 sample id 후보 파일이 여러 개 있습니다. 첫 파일 사용: {existing[0]}")

    ids_path = existing[0]
    ids = np.load(ids_path, allow_pickle=True)
    return ids, ids_path


def validate_optional_sample_ids(
    scenario_name: str,
    ids_winflat: np.ndarray | None,
    ids_seq: np.ndarray | None,
    y_winflat: np.ndarray,
    y_seq: np.ndarray,
) -> None:
    """
    winflat / seq의 샘플 순서가 truly aligned 되어 있는지 선택적으로 검증.
    sample_ids 파일이 둘 다 있을 때만 강하게 비교.
    """
    if ids_winflat is None and ids_seq is None:
        print(
            f"[WARN] {scenario_name}: sample_ids 파일이 없어 winflat/seq의 원본 샘플 정렬 일치 여부를 "
            "코드로 확인할 수 없습니다. 두 파이프라인이 동일한 순서로 생성되었다는 전제가 필요합니다."
        )
        return

    if (ids_winflat is None) != (ids_seq is None):
        print(
            f"[WARN] {scenario_name}: winflat/seq 중 한쪽에만 sample_ids 파일이 있습니다. "
            "정렬 일치 여부를 완전하게 검증할 수 없습니다."
        )
        return

    if ids_winflat.shape[0] != ids_seq.shape[0]:
        raise ValueError(
            f"[{scenario_name}] sample_ids 길이 불일치: "
            f"winflat={ids_winflat.shape[0]}, seq={ids_seq.shape[0]}"
        )

    if not np.array_equal(ids_winflat, ids_seq):
        raise ValueError(
            f"[{scenario_name}] sample_ids가 불일치합니다. "
            "winflat/seq가 동일한 샘플 순서를 공유하지 않을 수 있습니다."
        )

    # sample_ids가 동일하다면 사실상 같은 샘플 정렬을 가정하므로 y도 일치해야 정상
    if not np.array_equal(y_winflat, y_seq):
        raise ValueError(
            f"[{scenario_name}] sample_ids는 일치하지만 y_winflat과 y_seq가 다릅니다."
        )


def load_ctu_scenarios(config: EvalConfig) -> list[ScenarioBundle]:
    """
    CTU 시나리오들을 로드.
    주의:
    - winflat과 seq는 서로 다른 표현이므로 sample_ids가 없을 때 y 동일성은 강제하지 않음
    - 하지만 샘플 수 자체가 다르면 동일 외부검증 세트 비교가 어려우므로 에러 처리
    """
    scenarios: list[ScenarioBundle] = []

    for scenario_dir in config.ctu_scenario_dirs:
        if not scenario_dir.exists():
            raise FileNotFoundError(f"시나리오 폴더를 찾을 수 없습니다: {scenario_dir}")

        Xw, yw, xw_path, yw_path = load_numpy_pair(scenario_dir, "winflat")
        Xs, ys, xs_path, ys_path = load_numpy_pair(scenario_dir, "seq")

        if Xw.shape[0] != Xs.shape[0]:
            raise ValueError(
                f"[{scenario_dir.name}] winflat/seq 샘플 수 불일치: "
                f"winflat={Xw.shape[0]}, seq={Xs.shape[0]}"
            )

        ids_w, ids_w_path = load_optional_sample_ids(scenario_dir, "winflat")
        ids_s, ids_s_path = load_optional_sample_ids(scenario_dir, "seq")
        validate_optional_sample_ids(scenario_dir.name, ids_w, ids_s, yw, ys)

        bundle = ScenarioBundle(
            name=scenario_dir.name,
            X_winflat=Xw,
            y_winflat=yw,
            X_seq=Xs,
            y_seq=ys,
            winflat_x_path=xw_path,
            winflat_y_path=yw_path,
            seq_x_path=xs_path,
            seq_y_path=ys_path,
            winflat_sample_ids=ids_w,
            seq_sample_ids=ids_s,
            winflat_sample_ids_path=ids_w_path,
            seq_sample_ids_path=ids_s_path,
        )
        scenarios.append(bundle)

        print(f"[LOAD] {bundle.name}")
        print(f"       winflat: X={bundle.X_winflat.shape}, y={bundle.y_winflat.shape}")
        print(f"       seq    : X={bundle.X_seq.shape}, y={bundle.y_seq.shape}")

    return scenarios


def concat_winflat(scenarios: list[ScenarioBundle]) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([s.X_winflat for s in scenarios], axis=0)
    y = np.concatenate([s.y_winflat for s in scenarios], axis=0)
    return X, y


def concat_seq(scenarios: list[ScenarioBundle]) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([s.X_seq for s in scenarios], axis=0)
    y = np.concatenate([s.y_seq for s in scenarios], axis=0)
    return X, y


def scenario_source_metadata(scenarios: list[ScenarioBundle]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for s in scenarios:
        meta[s.name] = {
            "winflat_x_path": str(s.winflat_x_path),
            "winflat_y_path": str(s.winflat_y_path),
            "seq_x_path": str(s.seq_x_path),
            "seq_y_path": str(s.seq_y_path),
            "winflat_sample_ids_path": str(s.winflat_sample_ids_path) if s.winflat_sample_ids_path else None,
            "seq_sample_ids_path": str(s.seq_sample_ids_path) if s.seq_sample_ids_path else None,
            "winflat_num_samples": int(s.X_winflat.shape[0]),
            "seq_num_samples": int(s.X_seq.shape[0]),
        }
    return meta


# =========================================================
# 5. 공통 유틸
# =========================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """
    공통 평가 지표 계산.
    classification_report는 문자열 + dict 둘 다 저장.
    ROC-AUC는 y_true가 단일 클래스면 계산 불가할 수 있어 예외 처리.
    """
    report_text = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def print_metrics(title: str, threshold: float, metrics: dict[str, Any]) -> None:
    roc_auc_text = (
        round(metrics["roc_auc"], 4)
        if metrics["roc_auc"] is not None
        else "N/A"
    )

    print(f"\n===== {title} =====")
    print(f"Threshold : {threshold:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {roc_auc_text}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(metrics["classification_report_text"])


def maybe_load_scaler(scaler_path: Path, use_scaler: bool) -> object | None:
    if not use_scaler:
        return None

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler 적용이 켜져 있는데 파일이 없습니다: {scaler_path}"
        )

    print(f"[LOAD] scaler: {scaler_path}")
    return joblib.load(scaler_path)


def apply_seq_scaler(X_seq: np.ndarray, scaler: object | None) -> np.ndarray:
    """
    seq 데이터 (N, seq_len, feature_dim)에 scaler 적용.
    scaler는 feature_dim 기준으로 학습된 것으로 가정.
    """
    if scaler is None:
        return X_seq

    n_samples, seq_len, n_features = X_seq.shape
    X_2d = X_seq.reshape(-1, n_features)
    X_scaled = scaler.transform(X_2d)
    return X_scaled.reshape(n_samples, seq_len, n_features)


def load_model_artifact(
    model_path: Path,
    default_threshold: float,
) -> tuple[object, float]:
    """
    sklearn artifact 로드.
    지원 형태:
    1) model 객체만 저장된 경우
    2) {"model": model, "threshold": x} 형태
    """
    artifact = joblib.load(model_path)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        threshold = artifact.get("threshold", default_threshold)
    else:
        model = artifact
        threshold = default_threshold

    return model, float(threshold)


def safe_torch_load(model_path: Path, device: str, trusted_model_files: bool) -> Any:
    """
    PyTorch 버전 차이를 고려한 안전 로드.
    trusted_model_files=True일 때만 weights_only=False 경로를 허용.
    이 경우 반드시 신뢰 가능한 로컬 모델 파일만 사용해야 함.
    """
    if trusted_model_files:
        try:
            return torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(model_path, map_location=device)

    # 신뢰되지 않은 파일은 가급적 weights_only=True만 허용
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError as e:
        raise RuntimeError(
            "현재 PyTorch 버전에서는 안전한 weights_only 로드를 지원하지 않습니다. "
            "신뢰되지 않은 모델 파일은 로드하지 않는 것이 안전합니다."
        ) from e


def to_1d_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    batch size=1일 때도 안전하게 1차원 numpy 배열로 변환.
    """
    arr = tensor.detach().cpu().numpy()
    arr = np.asarray(arr)
    arr = np.atleast_1d(arr)
    return arr.reshape(-1)


def extract_positive_probability(
    outputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    output_mode: str,
) -> np.ndarray:
    """
    CNN 출력에서 positive class probability 추출.
    자동 추정 대신 output_mode를 명시적으로 받아 안전하게 처리.
    """
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

    if output_mode == "logit_binary":
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(-1)
        elif outputs.ndim > 2:
            raise ValueError(
                f"logit_binary는 출력 shape이 (N,) 또는 (N,1)이어야 합니다. 현재: {tuple(outputs.shape)}"
            )
        probs = torch.sigmoid(outputs)
        return to_1d_numpy(probs)

    if output_mode == "prob_binary":
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(-1)
        elif outputs.ndim > 2:
            raise ValueError(
                f"prob_binary는 출력 shape이 (N,) 또는 (N,1)이어야 합니다. 현재: {tuple(outputs.shape)}"
            )
        return to_1d_numpy(outputs)

    if output_mode == "logit_multiclass":
        if outputs.ndim != 2 or outputs.shape[1] != 2:
            raise ValueError(
                f"logit_multiclass는 출력 shape이 (N, 2)여야 합니다. 현재: {tuple(outputs.shape)}"
            )
        probs = torch.softmax(outputs, dim=1)[:, 1]
        return to_1d_numpy(probs)

    if output_mode == "prob_multiclass":
        if outputs.ndim != 2 or outputs.shape[1] != 2:
            raise ValueError(
                f"prob_multiclass는 출력 shape이 (N, 2)여야 합니다. 현재: {tuple(outputs.shape)}"
            )
        probs = outputs[:, 1]
        return to_1d_numpy(probs)

    raise ValueError(
        f"지원하지 않는 cnn_output_mode입니다: {output_mode}. "
        "허용값: logit_binary, prob_binary, logit_multiclass, prob_multiclass"
    )


def build_summary_row(
    model_name: str,
    scenario_name: str,
    threshold: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    cm = metrics["confusion_matrix"]
    return {
        "model": model_name,
        "scenario": scenario_name,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": np.nan if metrics["roc_auc"] is None else metrics["roc_auc"],
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
    }


# =========================================================
# 6. 모델 평가 함수
# =========================================================

def evaluate_sklearn_model(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    model_name: str,
    scaler: object | None = None,
) -> dict[str, Any]:
    X_eval = scaler.transform(X) if scaler is not None else X

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model_name} 모델에 predict_proba가 없습니다. "
            "threshold 기반 평가를 위해 predict_proba 지원 모델이어야 합니다."
        )

    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return compute_metrics(y, y_pred, y_prob)


def load_cnn_lstm_model(
    model_path: Path,
    device: str,
    input_dim: int,
    default_threshold: float,
    trusted_model_files: bool,
) -> tuple[nn.Module, float]:
    """
    CNN-LSTM 모델 로드.
    지원 형태:
    1) torch.save(model, path)
    2) state_dict (OrderedDict 포함)
    3) {"state_dict": ..., "threshold": ..., "model_kwargs": ...}
    """
    checkpoint = safe_torch_load(model_path, device, trusted_model_files=trusted_model_files)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model, float(default_threshold)

    try:
        from project.src.train_cnn_lstm import CNNLSTMModel
    except ImportError as e:
        raise ImportError(
            "CNNLSTMModel 클래스를 import하지 못했습니다. "
            "학습에 사용한 동일한 모델 클래스를 import 가능하도록 경로를 맞춰야 합니다."
        ) from e

    model_kwargs: dict[str, Any] = {}
    threshold = default_threshold

    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        if "state_dict" not in checkpoint:
            raise ValueError(
                f"checkpoint가 dict지만 'state_dict' 키가 없습니다. "
                f"존재하는 키: {list(checkpoint.keys())}"
            )
        state_dict = checkpoint["state_dict"]
        model_kwargs = checkpoint.get("model_kwargs", {})
        threshold = float(checkpoint.get("threshold", default_threshold))
    else:
        raise TypeError(
            f"지원하지 않는 checkpoint 타입입니다: {type(checkpoint)}"
        )

    model_kwargs.setdefault("input_dim", input_dim)
    model_kwargs.setdefault("num_classes", 1)

    model = CNNLSTMModel(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, float(threshold)


def evaluate_torch_binary_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    device: str,
    batch_size: int,
    output_mode: str,
    scaler: object | None = None,
) -> dict[str, Any]:
    X_eval = apply_seq_scaler(X, scaler)

    X_tensor = torch.as_tensor(X_eval, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    prob_list: list[np.ndarray] = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            batch_prob = extract_positive_probability(outputs, output_mode=output_mode)
            prob_list.append(batch_prob)

    y_prob = np.concatenate(prob_list, axis=0)
    y_pred = (y_prob >= threshold).astype(int)

    return compute_metrics(y, y_pred, y_prob)


def run_rf_evaluation(
    config: EvalConfig,
    scenarios: list[ScenarioBundle],
    winflat_scaler: object | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], float] | None:
    if not config.rf_model_path.exists():
        print(f"[SKIP] RF 모델 파일 없음: {config.rf_model_path}")
        return None

    rf_model, rf_threshold = load_model_artifact(
        config.rf_model_path,
        config.default_thresholds["rf"],
    )

    scenario_rows: list[dict[str, Any]] = []
    scenario_metrics_json: dict[str, Any] = {}

    for s in scenarios:
        metrics = evaluate_sklearn_model(
            model=rf_model,
            X=s.X_winflat,
            y=s.y_winflat,
            threshold=rf_threshold,
            model_name="RF",
            scaler=winflat_scaler,
        )
        scenario_rows.append(build_summary_row("rf_ctu", s.name, rf_threshold, metrics))
        scenario_metrics_json[s.name] = metrics
        print(
            f"[SCENARIO] RF | {s.name} | "
            f"Acc={metrics['accuracy']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}"
        )

    X_all, y_all = concat_winflat(scenarios)
    merged_metrics = evaluate_sklearn_model(
        model=rf_model,
        X=X_all,
        y=y_all,
        threshold=rf_threshold,
        model_name="RF",
        scaler=winflat_scaler,
    )

    print_metrics("RF CTU External Validation", rf_threshold, merged_metrics)

    save_json(
        {
            "model": "rf_ctu",
            "threshold": rf_threshold,
            "source_files": scenario_source_metadata(scenarios),
            "merged": merged_metrics,
            "per_scenario": scenario_metrics_json,
        },
        config.result_dir / "rf_ctu_metrics.json",
    )

    return merged_metrics, scenario_rows, rf_threshold


def run_xgb_evaluation(
    config: EvalConfig,
    scenarios: list[ScenarioBundle],
    winflat_scaler: object | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], float] | None:
    if not config.xgb_model_path.exists():
        print(f"[SKIP] XGBoost 모델 파일 없음: {config.xgb_model_path}")
        return None

    xgb_model, xgb_threshold = load_model_artifact(
        config.xgb_model_path,
        config.default_thresholds["xgb"],
    )

    scenario_rows: list[dict[str, Any]] = []
    scenario_metrics_json: dict[str, Any] = {}

    for s in scenarios:
        metrics = evaluate_sklearn_model(
            model=xgb_model,
            X=s.X_winflat,
            y=s.y_winflat,
            threshold=xgb_threshold,
            model_name="XGBoost",
            scaler=winflat_scaler,
        )
        scenario_rows.append(build_summary_row("xgb_ctu", s.name, xgb_threshold, metrics))
        scenario_metrics_json[s.name] = metrics
        print(
            f"[SCENARIO] XGB | {s.name} | "
            f"Acc={metrics['accuracy']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}"
        )

    X_all, y_all = concat_winflat(scenarios)
    merged_metrics = evaluate_sklearn_model(
        model=xgb_model,
        X=X_all,
        y=y_all,
        threshold=xgb_threshold,
        model_name="XGBoost",
        scaler=winflat_scaler,
    )

    print_metrics("XGBoost CTU External Validation", xgb_threshold, merged_metrics)

    save_json(
        {
            "model": "xgb_ctu",
            "threshold": xgb_threshold,
            "source_files": scenario_source_metadata(scenarios),
            "merged": merged_metrics,
            "per_scenario": scenario_metrics_json,
        },
        config.result_dir / "xgb_ctu_metrics.json",
    )

    return merged_metrics, scenario_rows, xgb_threshold


def run_cnn_lstm_evaluation(
    config: EvalConfig,
    scenarios: list[ScenarioBundle],
    seq_scaler: object | None,
    input_dim: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], float] | None:
    if not config.cnn_lstm_model_path.exists():
        print(f"[SKIP] CNN-LSTM 모델 파일 없음: {config.cnn_lstm_model_path}")
        return None

    cnn_model, cnn_threshold = load_cnn_lstm_model(
        model_path=config.cnn_lstm_model_path,
        device=config.device,
        input_dim=input_dim,
        default_threshold=config.default_thresholds["cnn_lstm"],
        trusted_model_files=config.trusted_model_files,
    )

    scenario_rows: list[dict[str, Any]] = []
    scenario_metrics_json: dict[str, Any] = {}

    for s in scenarios:
        metrics = evaluate_torch_binary_model(
            model=cnn_model,
            X=s.X_seq,
            y=s.y_seq,
            threshold=cnn_threshold,
            device=config.device,
            batch_size=config.cnn_batch_size,
            output_mode=config.cnn_output_mode,
            scaler=seq_scaler,
        )
        scenario_rows.append(build_summary_row("cnn_lstm_ctu", s.name, cnn_threshold, metrics))
        scenario_metrics_json[s.name] = metrics
        print(
            f"[SCENARIO] CNN-LSTM | {s.name} | "
            f"Acc={metrics['accuracy']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}"
        )

    X_all, y_all = concat_seq(scenarios)
    merged_metrics = evaluate_torch_binary_model(
        model=cnn_model,
        X=X_all,
        y=y_all,
        threshold=cnn_threshold,
        device=config.device,
        batch_size=config.cnn_batch_size,
        output_mode=config.cnn_output_mode,
        scaler=seq_scaler,
    )

    print_metrics("CNN-LSTM CTU External Validation", cnn_threshold, merged_metrics)

    save_json(
        {
            "model": "cnn_lstm_ctu",
            "threshold": cnn_threshold,
            "output_mode": config.cnn_output_mode,
            "source_files": scenario_source_metadata(scenarios),
            "merged": merged_metrics,
            "per_scenario": scenario_metrics_json,
        },
        config.result_dir / "cnn_lstm_ctu_metrics.json",
    )

    return merged_metrics, scenario_rows, cnn_threshold


# =========================================================
# 7. 메인 실행
# =========================================================

def main() -> None:
    print("=== CTU-13 External Validation Start ===")
    print(f"[DEVICE] {CONFIG.device}")

    scenarios = load_ctu_scenarios(CONFIG)

    X_ctu_winflat, y_ctu_winflat = concat_winflat(scenarios)
    X_ctu_seq, y_ctu_seq = concat_seq(scenarios)

    print("\n[SUMMARY]")
    print(f"CTU winflat X: {X_ctu_winflat.shape}")
    print(f"CTU winflat y: {y_ctu_winflat.shape}")
    print(f"CTU seq X    : {X_ctu_seq.shape}")
    print(f"CTU seq y    : {y_ctu_seq.shape}")

    if X_ctu_winflat.shape[0] != X_ctu_seq.shape[0]:
        raise ValueError(
            f"winflat({X_ctu_winflat.shape[0]})과 seq({X_ctu_seq.shape[0]}) 샘플 수가 다릅니다."
        )

    if X_ctu_seq.ndim != 3:
        raise ValueError(f"seq 입력 shape이 3차원이 아닙니다: {X_ctu_seq.shape}")

    input_dim = X_ctu_seq.shape[2]
    print(f"[INFO] inferred input_dim from seq data: {input_dim}")

    winflat_scaler = maybe_load_scaler(
        CONFIG.winflat_scaler_path,
        CONFIG.apply_winflat_scaler,
    )
    seq_scaler = maybe_load_scaler(
        CONFIG.seq_scaler_path,
        CONFIG.apply_seq_scaler,
    )

    overall_rows: list[dict[str, Any]] = []
    scenario_rows_all: list[dict[str, Any]] = []

    rf_result = run_rf_evaluation(CONFIG, scenarios, winflat_scaler)
    if rf_result is not None:
        merged_metrics, scenario_rows, rf_threshold = rf_result
        overall_rows.append(build_summary_row("rf_ctu", "ALL", rf_threshold, merged_metrics))
        scenario_rows_all.extend(scenario_rows)

    xgb_result = run_xgb_evaluation(CONFIG, scenarios, winflat_scaler)
    if xgb_result is not None:
        merged_metrics, scenario_rows, xgb_threshold = xgb_result
        overall_rows.append(build_summary_row("xgb_ctu", "ALL", xgb_threshold, merged_metrics))
        scenario_rows_all.extend(scenario_rows)

    cnn_result = run_cnn_lstm_evaluation(CONFIG, scenarios, seq_scaler, input_dim=input_dim)
    if cnn_result is not None:
        merged_metrics, scenario_rows, cnn_threshold = cnn_result
        overall_rows.append(build_summary_row("cnn_lstm_ctu", "ALL", cnn_threshold, merged_metrics))
        scenario_rows_all.extend(scenario_rows)

    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        overall_path = CONFIG.result_dir / "ctu_external_summary.csv"
        overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
        print(f"\n[SAVE] 전체 결과 요약 저장 완료: {overall_path}")
        print(overall_df)
    else:
        print("\n[WARN] 저장할 전체 평가 결과가 없습니다.")

    if scenario_rows_all:
        scenario_df = pd.DataFrame(scenario_rows_all)
        scenario_path = CONFIG.result_dir / "ctu_external_per_scenario_summary.csv"
        scenario_df.to_csv(scenario_path, index=False, encoding="utf-8-sig")
        print(f"\n[SAVE] 시나리오별 결과 요약 저장 완료: {scenario_path}")
        print(scenario_df)
    else:
        print("\n[WARN] 저장할 시나리오별 평가 결과가 없습니다.")

    print("\n=== CTU-13 External Validation End ===")


if __name__ == "__main__":
    main()