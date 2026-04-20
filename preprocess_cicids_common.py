# preprocess_cicids_common.py
# 목적:
# - CIC / CTU 공통 전처리 함수 모음
# - 공통 feature 스키마 정의
# - raw DataFrame -> 공통 row DataFrame 변환

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =========================================================
# 1. 공통 feature 정의
# =========================================================

COMMON_FEATURES = [
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Average Packet Size",
    "Flow IAT Mean",
    "Flow IAT Max",
]

COMMON_BASE_COLS = [
    "Timestamp",
    "Source IP",
    "Destination IP",
    "Label_raw",
    "Label_binary",
]

COMMON_ALL_COLS = COMMON_BASE_COLS + COMMON_FEATURES


# =========================================================
# 2. 기본 유틸
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0, np.nan)
    out = numerator / den
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def parse_protocol(value) -> int:
    if pd.isna(value):
        return 0

    s = str(value).strip().lower()

    if s in {"tcp", "6"}:
        return 6
    if s in {"udp", "17"}:
        return 17
    if s in {"icmp", "1"}:
        return 1

    try:
        return int(float(s))
    except Exception:
        return 0


def parse_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if ts.notna().sum() == 0:
        ts = pd.to_datetime(series, errors="coerce")
    return ts


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> Optional[str]:
    col_map = {c.strip().lower(): c for c in df.columns}

    for cand in candidates:
        key = cand.strip().lower()
        if key in col_map:
            return col_map[key]

    if required:
        raise KeyError(
            f"필수 컬럼을 찾지 못했습니다. 후보: {candidates}\n"
            f"현재 컬럼: {list(df.columns)}"
        )
    return None


# =========================================================
# 3. 라벨 이진화
# =========================================================

def cic_label_to_binary(label_value: str) -> int:
    if pd.isna(label_value):
        return 0

    s = str(label_value).strip().lower()

    if s in {"bot", "botnet"}:
        return 1

    # CIC는 bot/botnet만 1, 나머지는 0
    return 0


def ctu_label_to_binary(label_value: str, treat_to_botnet_as_attack: bool = False) -> int:
    if pd.isna(label_value):
        return 0

    s = str(label_value).strip().lower()

    if "to-botnet" in s:
        return 1 if treat_to_botnet_as_attack else 0

    if any(k in s for k in ["from-botnet", "botnet", "malicious"]):
        return 1

    return 0


# =========================================================
# 4. CIC -> 공통 스키마
# =========================================================

def standardize_cic_to_common(df: pd.DataFrame) -> pd.DataFrame:
    """
    CIC-IDS2017 TrafficLabeling raw -> 공통 row 스키마
    """
    df = normalize_column_names(df).copy()

    required = [
        "Timestamp",
        "Source IP",
        "Destination IP",
        "Label",
        "Protocol",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Average Packet Size",
        "Flow IAT Mean",
        "Flow IAT Max",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CIC 필수 컬럼 누락: {missing}")

    df["Timestamp"] = parse_timestamp(df["Timestamp"])
    df["Source IP"] = df["Source IP"].astype(str).str.strip()
    df["Destination IP"] = df["Destination IP"].astype(str).str.strip()

    df["Label_raw"] = df["Label"].astype(str)
    df["Label_binary"] = df["Label_raw"].apply(cic_label_to_binary)

    df["Protocol"] = df["Protocol"].apply(parse_protocol)
    df["Flow Duration"] = to_numeric(df["Flow Duration"])
    df["Total Fwd Packets"] = to_numeric(df["Total Fwd Packets"])
    df["Total Bwd Packets"] = to_numeric(df["Total Backward Packets"])
    df["Flow Bytes/s"] = to_numeric(df["Flow Bytes/s"])
    df["Flow Packets/s"] = to_numeric(df["Flow Packets/s"])
    df["Average Packet Size"] = to_numeric(df["Average Packet Size"])
    df["Flow IAT Mean"] = to_numeric(df["Flow IAT Mean"])
    df["Flow IAT Max"] = to_numeric(df["Flow IAT Max"])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Timestamp", "Source IP", "Destination IP"]).copy()
    df[COMMON_FEATURES] = df[COMMON_FEATURES].fillna(0.0)

    return df[COMMON_ALL_COLS].copy()


# =========================================================
# 5. CTU -> 공통 스키마
# =========================================================

def standardize_ctu_to_common(
    df: pd.DataFrame,
    treat_to_botnet_as_attack: bool = False,
) -> pd.DataFrame:
    """
    CTU-13 raw -> 공통 row 스키마
    """
    df = normalize_column_names(df).copy()

    timestamp_col = find_column(df, ["StartTime", "Timestamp", "starttime", "stime"])
    duration_col = find_column(df, ["Dur", "Duration", "dur"])
    proto_col = find_column(df, ["Proto", "Protocol", "proto"])
    src_ip_col = find_column(df, ["SrcAddr", "Source IP", "Src IP", "srcaddr", "src_ip"])
    dst_ip_col = find_column(df, ["DstAddr", "Destination IP", "Dst IP", "dstaddr", "dst_ip"])
    label_col = find_column(df, ["Label", "label"])

    total_pkts_col = find_column(df, ["TotPkts", "Total Packets", "totpkts", "pkts"], required=False)
    total_bytes_col = find_column(df, ["TotBytes", "Total Bytes", "totbytes", "bytes"], required=False)
    src_pkts_col = find_column(df, ["SrcPkts", "srcpkts", "spkts"], required=False)
    dst_pkts_col = find_column(df, ["DstPkts", "dstpkts", "dpkts"], required=False)

    if total_pkts_col is None:
        raise KeyError("CTU 총 패킷 수 컬럼을 찾지 못했습니다.")
    if total_bytes_col is None:
        raise KeyError("CTU 총 바이트 수 컬럼을 찾지 못했습니다.")

    df["Timestamp"] = parse_timestamp(df[timestamp_col])
    df["Source IP"] = df[src_ip_col].astype(str).str.strip()
    df["Destination IP"] = df[dst_ip_col].astype(str).str.strip()

    df["Label_raw"] = df[label_col].astype(str)
    df["Label_binary"] = df["Label_raw"].apply(
        lambda x: ctu_label_to_binary(x, treat_to_botnet_as_attack=treat_to_botnet_as_attack)
    )

    df["Protocol"] = df[proto_col].apply(parse_protocol)
    df["Flow Duration"] = to_numeric(df[duration_col])

    df["TotPkts"] = to_numeric(df[total_pkts_col])
    df["TotBytes"] = to_numeric(df[total_bytes_col])

    if src_pkts_col is not None and dst_pkts_col is not None:
        df["Total Fwd Packets"] = to_numeric(df[src_pkts_col])
        df["Total Bwd Packets"] = to_numeric(df[dst_pkts_col])
    else:
        df["Total Fwd Packets"] = df["TotPkts"]
        df["Total Bwd Packets"] = 0.0

    df["Flow Bytes/s"] = safe_divide(df["TotBytes"], df["Flow Duration"])
    df["Flow Packets/s"] = safe_divide(df["TotPkts"], df["Flow Duration"])
    df["Average Packet Size"] = safe_divide(df["TotBytes"], df["TotPkts"])

    denom = (df["TotPkts"] - 1).clip(lower=1)
    df["Flow IAT Mean"] = safe_divide(df["Flow Duration"], denom)
    df["Flow IAT Max"] = df["Flow Duration"].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Timestamp", "Source IP", "Destination IP"]).copy()
    df[COMMON_FEATURES] = df[COMMON_FEATURES].fillna(0.0)

    return df[COMMON_ALL_COLS].copy()


# =========================================================
# 6. 검증 / 저장
# =========================================================

def validate_common_schema(df: pd.DataFrame) -> None:
    missing = [c for c in COMMON_ALL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"공통 스키마 컬럼 누락: {missing}")


def save_common_parquet(df: pd.DataFrame, save_path: Path) -> None:
    validate_common_schema(df)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)
    print(f"[SAVE] {save_path}")