"""
CICIDS2017 MachineLearningCSV를 13-class Bot 증강 실험에 맞게 전처리하는 파일.

여러 CSV 파일을 읽고, 원본 Label을 정리하며, 학습에 사용할 숫자형 feature를
선택하고 결측값/무한대 값을 처리한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cicids2017_bot_config import (
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_COLUMNS,
    LOG_TRANSFORM_HINTS,
    RANDOM_STATE,
    RAW_DIR,
)


def normalize_label(label: object) -> str | None:
    """
    원본 CICIDS2017 Label 값을 본 실험에서 사용하는 13개 클래스 이름으로 변환한다.

    Web Attack 세부 라벨은 모두 하나의 "Web Attack" 클래스로 통합하고,
    정의되지 않은 라벨은 None을 반환하여 이후 단계에서 제외한다.
    """
    text = str(label).strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = " ".join(text.split())

    mapping = {
        "benign": "Benign",
        "ddos": "DDoS",
        "portscan": "PortScan",
        "port scan": "PortScan",
        "bot": "Bot",
        "infiltration": "Infiltration",
        "ftp-patator": "FTP-Patator",
        "ssh-patator": "SSH-Patator",
        "dos goldeneye": "DoS GoldenEye",
        "dos hulk": "DoS Hulk",
        "dos slowhttptest": "DoS Slowhttptest",
        "dos slowloris": "DoS Slowloris",
        "heartbleed": "Heartbleed",
    }
    if text.startswith("web attack"):
        return "Web Attack"
    return mapping.get(text)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    CSV 파일을 여러 인코딩으로 읽는다.

    CICIDS2017 CSV는 파일에 따라 인코딩 문제가 발생할 수 있으므로
    utf-8, cp1252, latin-1 순서로 시도한다.
    """
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def load_cicids2017() -> pd.DataFrame:
    """
    raw 데이터 폴더의 CICIDS2017 CSV 파일을 모두 로드하고 하나의 DataFrame으로 합친다.

    Label 컬럼이 없는 파일은 건너뛰고, 13개 클래스에 매핑되는 행만 남긴다.
    실행 중 전체 데이터 크기와 클래스 분포를 출력하여 데이터 로드 상태를 확인한다.
    """
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CICIDS2017 CSV files not found: {RAW_DIR}")

    frames = []
    for path in files:
        print(f"[LOAD] {path.name}")
        frame = read_csv_flexible(path)
        frame.columns = frame.columns.str.strip()
        if "Label" not in frame.columns:
            print(f"[SKIP] Label column missing: {path.name}")
            continue
        frame["class_name"] = frame["Label"].map(normalize_label)
        frame = frame.dropna(subset=["class_name"])
        if len(frame):
            frames.append(frame)

    if not frames:
        raise ValueError("No CICIDS2017 rows matched the 13-class label map.")

    df = pd.concat(frames, ignore_index=True)
    print(f"[LOAD] merged shape={df.shape}")
    print("[LABEL] class distribution:")
    print(df["class_name"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int))
    return df.reset_index(drop=True)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    학습에 사용할 숫자형 feature 컬럼 목록을 추론한다.

    Flow ID, IP, Timestamp, Label, class_name, 중복 컬럼인 Fwd Header Length.1은
    제외하고, 숫자로 변환 가능한 컬럼만 feature로 사용한다.
    """
    candidates = [col for col in df.columns if col not in ID_COLUMNS and col != "class_name"]
    numeric_cols = []
    for col in candidates:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_cols.append(col)
    if not numeric_cols:
        raise ValueError("No numeric feature columns found.")
    return numeric_cols


def clean_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    preprocess: str,
) -> pd.DataFrame:
    """
    feature 값을 숫자로 변환하고 결측값/무한대 값을 정리한다.

    preprocess가 "paper"이면 log 변환 없이 MinMax scaling 전 단계까지만 처리한다.
    preprocess가 "log1p"이면 flow 통계량 계열 feature에 log1p 변환을 적용한다.
    마지막으로 class_name을 정수 class_id로 변환한다.
    """
    work = df.copy()
    for col in feature_cols:
        values = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        lower = col.lower()
        if preprocess == "log1p" and any(token in lower for token in LOG_TRANSFORM_HINTS):
            values = np.log1p(np.maximum(values.fillna(0).to_numpy(), 0))
        work[col] = values
    work[feature_cols] = work[feature_cols].fillna(0)
    work["class_id"] = work["class_name"].map(CLASS_TO_ID).astype(np.int32)
    return work


def stratified_debug_sample(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """
    빠른 디버깅을 위해 클래스 비율을 유지하면서 최대 max_rows개만 샘플링한다.

    전체 데이터 실행 전에 코드 흐름만 확인하고 싶을 때 사용하며,
    max_rows가 None이면 원본 DataFrame을 그대로 반환한다.
    """
    if max_rows is None or len(df) <= max_rows:
        return df

    samples = []
    ratios = df["class_id"].value_counts(normalize=True).to_dict()
    for class_id, ratio in ratios.items():
        subset = df[df["class_id"] == class_id]
        n_rows = max(1, int(max_rows * ratio))
        n_rows = min(n_rows, len(subset))
        samples.append(subset.sample(n=n_rows, random_state=RANDOM_STATE))

    sampled = pd.concat(samples, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=RANDOM_STATE)
    return sampled.reset_index(drop=True)
