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
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def load_cicids2017() -> pd.DataFrame:
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
