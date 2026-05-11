"""
preprocess_cicids2017.py

CIC-IDS2017 전처리 — flow 기반 (논문들과 동일)

전처리 방식: flow 1개 = 샘플 1개
  - RF / XGBoost : flat/  (n_flows, n_features)
  - CNN-LSTM/GRU : seq/   (n_flows, n_features, 1)

저장 구조:
  data/processed/cicids2017/
    flat/  X_train/val/test.npy  y_train/val/test.npy
    seq/   X_train/val/test.npy  y_train/val/test.npy
           scaler_flow.pkl       ← CIC2018, CTU13 전처리에서 재사용
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(_SRC_DIR)

RAW_DIR   = os.path.join(BASE_DIR, "data", "raw", "cic-ids2017")
SAVE_DIR  = os.path.join(BASE_DIR, "data", "processed", "cicids2017")
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 내부 네트워크 대역 정의
# =========================================================
INTERNAL_IP_PREFIXES = (
    "192.168.",
    "172.16.",
)


# =========================================================
# ML 피처셋 (77개)
# preprocess_cicids2018.py, preprocess_ctu13.py 와 완전히 동일
# =========================================================
ML_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Protocol",
]


# =========================================================
# 로그 변환 대상 피처
# preprocess_cicids2018.py, preprocess_ctu13.py 와 완전히 동일
# =========================================================
LOG_TRANSFORM_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


# =========================================================
# 필수 컬럼 목록
# =========================================================
REQUIRED_BASE_COLS = [
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Timestamp",
    "Label",
    "Flow Duration",
    "Protocol",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
]


# =========================================================
# 라벨 매핑
# Botnet(C&C 통신) = 1, 나머지 = 0
# =========================================================
BOTNET_LABELS = {"bot", "botnet"}

KNOWN_NON_BOT_LABELS = {
    "benign",
    "dos hulk",
    "portscan",
    "ddos",
    "dos goldeneye",
    "ftp-patator",
    "ssh-patator",
    "dos slowloris",
    "dos slowhttptest",
    "web attack \u2013 brute force",
    "web attack \u2013 xss",
    "web attack \u2013 sql injection",
    "infiltration",
    "heartbleed",
}

_label_counter: Counter = Counter()


# =========================================================
# 1. CSV 로드
# =========================================================
def load_all_csv(raw_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {raw_dir}")

    print(f"[LOAD] CSV 파일 수: {len(csv_files)}")

    df_list = []
    for file_path in csv_files:
        print(f"[LOAD] {os.path.basename(file_path)}")
        temp_df: Optional[pd.DataFrame] = None
        last_error: Optional[Exception] = None

        for encoding in ["utf-8", "cp1252", "latin-1"]:
            try:
                temp_df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                print(f"       encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                last_error = e
                break

        if temp_df is None:
            raise ValueError(f"파일 로드 실패: {os.path.basename(file_path)}") from last_error

        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    print(f"[LOAD] 병합 후 shape: {df.shape}")
    return df


# =========================================================
# 2. 컬럼명 정규화 / 검증
# =========================================================
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_BASE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")


# =========================================================
# 3. 기본 정제
# =========================================================
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    before_rows = len(df)
    print("\n[CLEAN] 정제 시작")

    # Label / Source IP 결측 제거
    df = df.dropna(subset=["Label", "Source IP"])
    df = df[
        (df["Label"].astype(str).str.strip() != "")
        & (df["Source IP"].astype(str).str.strip() != "")
    ]

    # 문자열 컬럼 strip
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # ML 피처 수치형 변환
    for col in [c for c in ML_FEATURES if c in df.columns]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamp 파싱
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce", dayfirst=True, format="mixed"
    )
    df = df.dropna(subset=["Timestamp"])

    # inf → NaN → 0
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Port 정제
    for port_col in ["Source Port", "Destination Port"]:
        if port_col in df.columns:
            df[port_col] = pd.to_numeric(
                df[port_col], errors="coerce"
            ).fillna(-1).astype(np.int32)

    print(f"[CLEAN] 제거된 행 수: {before_rows - len(df):,}")
    print(f"[CLEAN] 정제 후 shape: {df.shape}")
    return df.reset_index(drop=True)


# =========================================================
# 4. 이진 라벨 생성
# =========================================================
def _map_binary_label(label: str) -> int:
    label = str(label).strip().lower()
    if label in BOTNET_LABELS:
        return 1
    if label in KNOWN_NON_BOT_LABELS:
        return 0
    _label_counter[label] += 1
    return 0


def create_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    _label_counter.clear()
    df["Label_binary"] = df["Label"].map(_map_binary_label)

    if _label_counter:
        print("[LABEL] Unknown 라벨 (0으로 처리):")
        for lbl, cnt in _label_counter.most_common():
            print(f"  '{lbl}': {cnt:,}건")

    print("\n[LABEL] Label_binary 분포:")
    print(df["Label_binary"].value_counts())
    return df


# =========================================================
# 5. Protocol 정규화
# =========================================================
def normalize_protocol_column(df: pd.DataFrame) -> pd.DataFrame:
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")
    df["Protocol"] = df["Protocol"].fillna(-1).astype(np.int32)
    return df


# =========================================================
# 6. 시간 정렬
# =========================================================
def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print("[ORDER] Timestamp 기준 정렬 완료")
    return df


# =========================================================
# 7. 로그 변환 — CIC2018, CTU13 과 동일 조건
# =========================================================
def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    targets = [c for c in LOG_TRANSFORM_FEATURES if c in df.columns]
    print(f"\n[LOG] 변환 피처: {len(targets)}개")
    for col in targets:
        df[col] = np.log1p(np.maximum(
            pd.to_numeric(df[col], errors="coerce").fillna(0).values, 0
        ))
    return df


# =========================================================
# 8. 내부 IP 판별
# =========================================================
def is_internal_ip(ip: str) -> bool:
    return str(ip).startswith(INTERNAL_IP_PREFIXES)


# =========================================================
# 9. host 그룹 단위 train / val / test split
# (Source IP, 날짜) 단위로 그룹화 → 데이터 누수 방지
# 반환 데이터는 내부 IP flow만 포함
# =========================================================
def split_host_groups(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # 내부 IP만 필터링
    internal_mask = df["Source IP"].apply(is_internal_ip)
    df_internal   = df[internal_mask].copy()
    print(f"[SPLIT] 외부 IP flow {(~internal_mask).sum():,}개 제외")

    df_internal["_date"]      = df_internal["Timestamp"].dt.date
    df_internal["_group_key"] = (
        df_internal["Source IP"].astype(str) + "_"
        + df_internal["_date"].astype(str)
    )

    group_info = (
        df_internal.groupby("_group_key")["Label_binary"]
        .sum().gt(0).astype(int).reset_index()
    )
    group_keys    = group_info["_group_key"].tolist()
    group_has_bot = group_info["Label_binary"].tolist()

    print(f"[SPLIT] 전체 그룹: {len(group_keys):,}  "
          f"(Botnet 포함: {sum(group_has_bot):,})")

    keys_trainval, keys_test, _, _ = train_test_split(
        group_keys, group_has_bot,
        test_size=test_ratio, random_state=random_state,
        stratify=group_has_bot,
    )
    label_map       = dict(zip(group_keys, group_has_bot))
    labels_trainval = [label_map[k] for k in keys_trainval]
    adjusted_val    = val_ratio / (1.0 - test_ratio)
    keys_train, keys_val, _, _ = train_test_split(
        keys_trainval, labels_trainval,
        test_size=adjusted_val, random_state=random_state,
        stratify=labels_trainval,
    )

    drop_cols = ["_date", "_group_key"]
    df_train = df_internal[df_internal["_group_key"].isin(set(keys_train))].drop(columns=drop_cols).reset_index(drop=True)
    df_val   = df_internal[df_internal["_group_key"].isin(set(keys_val))].drop(columns=drop_cols).reset_index(drop=True)
    df_test  = df_internal[df_internal["_group_key"].isin(set(keys_test))].drop(columns=drop_cols).reset_index(drop=True)

    print(f"[SPLIT] train: {len(df_train):,} / val: {len(df_val):,} / test: {len(df_test):,}")
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  {name} Botnet 비율: {subset['Label_binary'].mean():.4f}")

    return df_train, df_val, df_test


# =========================================================
# 10. flow 단위 데이터 생성
# flow 1개 = 샘플 1개 (논문들과 동일)
#
# split_host_groups() 이후 호출되므로
# 이미 내부 IP만 있음 → 내부 IP 재필터링 불필요
#
# 출력:
#   X : (n_flows, n_features)      ← RF/XGB용
#   y : (n_flows,)
#   CNN-LSTM/GRU용 reshape는 main()에서 수행
# =========================================================
def create_flow_data(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:

    valid_cols   = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARN] 없는 컬럼 (스킵): {missing_cols}")

    X = df[valid_cols].values.astype(np.float32)
    y = df["Label_binary"].values.astype(np.int32)

    print(f"[FLOW] X shape: {X.shape}")
    print(f"[FLOW] Botnet 비율: {y.mean():.4f} ({y.sum():,}/{len(y):,})")

    return X, y


# =========================================================
# 11. 저장 유틸리티
# =========================================================
def save_outputs(df: pd.DataFrame, save_dir: str) -> None:
    save_cols = (
        ["Source IP", "Destination IP", "Destination Port",
         "Timestamp", "Label", "Label_binary"]
        + [col for col in ML_FEATURES if col in df.columns]
    )
    save_cols = [c for c in save_cols if c in df.columns]
    out_path  = os.path.join(save_dir, "cicids2017_traffic.parquet")
    df[save_cols].to_parquet(out_path, index=False)
    print(f"[SAVE] 전처리 데이터: {out_path}")


def save_numpy(
    data: np.ndarray,
    label: np.ndarray,
    save_dir: str,
    split_name: str,
) -> None:
    np.save(os.path.join(save_dir, f"X_{split_name}.npy"), data)
    np.save(os.path.join(save_dir, f"y_{split_name}.npy"), label)
    print(f"[SAVE] {split_name}: X={data.shape}  y={label.shape}")


# =========================================================
# 12. 분포 시각화 (선택)
# =========================================================
def analyze_flow_distribution(df: pd.DataFrame, save_dir: str) -> None:
    internal_mask   = df["Source IP"].apply(is_internal_ip)
    internal_counts = df[internal_mask].groupby("Source IP").size()

    print("\n[DIST] 내부 IP 당 flow 수:")
    print(internal_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(internal_counts, bins=50, kde=True, ax=axes[0],
                 log_scale=True, color="skyblue")
    axes[0].set_title("Flow Counts per Internal IP (Log Scale)")
    sns.boxplot(x=internal_counts, ax=axes[1], color="lightgreen")
    axes[1].set_xscale("log")
    axes[1].set_title("Boxplot of Flow Counts per Internal IP")
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "flow_distribution.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[DIST] 저장: {plot_path}")


def preview_data(df: pd.DataFrame) -> None:
    cols = ["Source IP", "Timestamp", "Label", "Label_binary",
            "Flow Duration", "Protocol"]
    cols = [c for c in cols if c in df.columns]
    print("\n[PREVIEW] 상위 5행")
    print(df[cols].head())
    print(f"[PREVIEW] 전체 shape: {df.shape}")


# =========================================================
# main
# =========================================================
def main():
    print("=" * 60)
    print("  CIC-IDS2017 Preprocessing  (Flow-Based)")
    print("=" * 60)
    print(f"  RAW_DIR  = {RAW_DIR}")
    print(f"  SAVE_DIR = {SAVE_DIR}")
    print(f"  ML_FEATURES = {len(ML_FEATURES)}개")
    print("=" * 60)

    FLAT_DIR = os.path.join(SAVE_DIR, "flat")   # RF / XGBoost
    SEQ_DIR  = os.path.join(SAVE_DIR, "seq")    # CNN-LSTM / GRU

    for d in [FLAT_DIR, SEQ_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 공통 전처리 ───────────────────────────────────────
    df = load_all_csv(RAW_DIR)
    df = normalize_column_names(df)
    validate_columns(df)
    df = basic_cleaning(df)
    df = create_binary_label(df)
    df = normalize_protocol_column(df)
    df = sort_by_time(df)
    df = apply_log_transform(df)

    preview_data(df)
    save_outputs(df, SAVE_DIR)
    analyze_flow_distribution(df, SAVE_DIR)

    # ── train / val / test 분리 ───────────────────────────
    df_train, df_val, df_test = split_host_groups(df)

    # ── flow 단위 데이터 생성 ─────────────────────────────
    X_tr, y_tr = create_flow_data(df_train, ML_FEATURES)
    X_va, y_va = create_flow_data(df_val,   ML_FEATURES)
    X_te, y_te = create_flow_data(df_test,  ML_FEATURES)

    n_feat = len(ML_FEATURES)

    # ── Scaler: train 기준 fit, val/test는 transform ──────
    # MinMaxScaler → [0, 1] 고정
    # D'Hooge et al. (2020): MinMaxScaler가 cross-dataset 성능에 중요
    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_va   = scaler.transform(X_va).astype(np.float32)
    X_te   = scaler.transform(X_te).astype(np.float32)

    # scaler 저장 → preprocess_cicids2018.py, preprocess_ctu13.py 에서 재사용
    scaler_path = os.path.join(SEQ_DIR, "scaler_flow.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\n[SCALER] 저장: {scaler_path}")

    # ── RF / XGBoost 저장: (n_flows, n_features) ─────────
    save_numpy(X_tr, y_tr, FLAT_DIR, "train")
    save_numpy(X_va, y_va, FLAT_DIR, "val")
    save_numpy(X_te, y_te, FLAT_DIR, "test")

    # ── CNN-LSTM / GRU 저장: (n_flows, n_features, 1) ────
    save_numpy(X_tr.reshape(-1, n_feat, 1), y_tr, SEQ_DIR, "train")
    save_numpy(X_va.reshape(-1, n_feat, 1), y_va, SEQ_DIR, "val")
    save_numpy(X_te.reshape(-1, n_feat, 1), y_te, SEQ_DIR, "test")

    print(f"\n[DONE] 저장 완료")
    print(f"  flat/ X shape: {X_tr.shape}               ← RF/XGB")
    print(f"  seq/  X shape: {X_tr.reshape(-1, n_feat, 1).shape}  ← CNN-LSTM/GRU")
    print(f"\n[NEXT]")
    print(f"  python preprocess_cicids2018.py")
    print(f"  python preprocess_ctu13.py")


if __name__ == "__main__":
    main()