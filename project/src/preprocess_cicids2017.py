"""
preprocess_cicids2017.py

CIC-IDS2017 전처리 — flow 기반
Chi-square 피처 선택: 77개 → 상위 20개 (Sayegh et al. 2024 동일)

저장 구조:
  data/processed/cicids2017/
    flat/  X_trainval.npy  y_trainval.npy
           X_test.npy      y_test.npy
    seq/   X_trainval.npy  y_trainval.npy
           X_test.npy      y_test.npy
           scaler_flow.pkl
    selected_features.json  ← chi-square 선택 피처 (CIC2018, CTU13에서 공유)
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
from sklearn.feature_selection import SelectKBest, chi2
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
# Chi-square 피처 선택 설정
# =========================================================
N_FEATURES = 32   # 상위 32개 피처

INTERNAL_IP_PREFIXES = ("192.168.", "172.16.",)

ML_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Protocol",
]

LOG_TRANSFORM_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

REQUIRED_BASE_COLS = [
    "Source IP", "Destination IP", "Source Port", "Destination Port",
    "Timestamp", "Label", "Flow Duration", "Protocol",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
]

BOTNET_LABELS      = {"bot", "botnet"}
KNOWN_NON_BOT_LABELS = {
    "benign", "dos hulk", "portscan", "ddos", "dos goldeneye",
    "ftp-patator", "ssh-patator", "dos slowloris", "dos slowhttptest",
    "web attack \u2013 brute force", "web attack \u2013 xss",
    "web attack \u2013 sql injection", "infiltration", "heartbleed",
}
_label_counter: Counter = Counter()


# =========================================================
# 전처리 함수들
# =========================================================
def load_all_csv(raw_dir):
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {raw_dir}")
    print(f"[LOAD] CSV 파일 수: {len(csv_files)}")
    df_list = []
    for file_path in csv_files:
        print(f"[LOAD] {os.path.basename(file_path)}")
        temp_df = None
        for encoding in ["utf-8", "cp1252", "latin-1"]:
            try:
                temp_df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if temp_df is None:
            raise ValueError(f"파일 로드 실패: {os.path.basename(file_path)}")
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    print(f"[LOAD] 병합 후 shape: {df.shape}")
    return df

def normalize_column_names(df):
    df.columns = df.columns.str.strip()
    return df

def validate_columns(df):
    missing = [col for col in REQUIRED_BASE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

def basic_cleaning(df):
    before_rows = len(df)
    df = df.dropna(subset=["Label", "Source IP"])
    df = df[(df["Label"].astype(str).str.strip() != "") & (df["Source IP"].astype(str).str.strip() != "")]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    for col in [c for c in ML_FEATURES if c in df.columns]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True, format="mixed")
    df = df.dropna(subset=["Timestamp"])
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    for port_col in ["Source Port", "Destination Port"]:
        if port_col in df.columns:
            df[port_col] = pd.to_numeric(df[port_col], errors="coerce").fillna(-1).astype(np.int32)
    print(f"[CLEAN] 제거된 행 수: {before_rows - len(df):,}  정제 후: {df.shape}")
    return df.reset_index(drop=True)

def _map_binary_label(label):
    label = str(label).strip().lower()
    if label in BOTNET_LABELS:      return 1
    if label in KNOWN_NON_BOT_LABELS: return 0
    _label_counter[label] += 1
    return 0

def create_binary_label(df):
    _label_counter.clear()
    df["Label_binary"] = df["Label"].map(_map_binary_label)
    print("\n[LABEL] Label_binary 분포:")
    print(df["Label_binary"].value_counts())
    return df

def normalize_protocol_column(df):
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(-1).astype(np.int32)
    return df

def sort_by_time(df):
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print("[ORDER] Timestamp 기준 정렬 완료")
    return df

def apply_log_transform(df):
    targets = [c for c in LOG_TRANSFORM_FEATURES if c in df.columns]
    for col in targets:
        df[col] = np.log1p(np.maximum(pd.to_numeric(df[col], errors="coerce").fillna(0).values, 0))
    print(f"[LOG] 변환 피처: {len(targets)}개")
    return df

def is_internal_ip(ip):
    return str(ip).startswith(INTERNAL_IP_PREFIXES)

def split_random(df, test_ratio=0.2, random_state=42):
    """
    Flow 단위 랜덤 Stratified split (Zhao et al. 2024 방식)
    봇넷 비율 유지 (stratify=Label_binary)
    """
    idx      = np.arange(len(df))
    y        = df["Label_binary"].values
    tv_idx, te_idx = train_test_split(
        idx, test_size=test_ratio, random_state=random_state, stratify=y
    )
    df_trainval = df.iloc[tv_idx].reset_index(drop=True)
    df_test     = df.iloc[te_idx].reset_index(drop=True)
    print(f"[SPLIT] 랜덤 Stratified split  trainval: {len(df_trainval):,} / test: {len(df_test):,}")
    for name, s in [("trainval", df_trainval), ("test", df_test)]:
        print(f"  {name} Botnet 비율: {s['Label_binary'].mean():.4f}")
    return df_trainval, df_test

def create_flow_data(df, feature_cols):
    valid_cols = [c for c in feature_cols if c in df.columns]
    X = df[valid_cols].values.astype(np.float32)
    y = df["Label_binary"].values.astype(np.int32)
    print(f"[FLOW] X={X.shape}  Botnet={y.mean():.4f}")
    return X, y

def save_numpy(data, label, save_dir, split_name):
    np.save(os.path.join(save_dir, f"X_{split_name}.npy"), data)
    np.save(os.path.join(save_dir, f"y_{split_name}.npy"), label)
    print(f"[SAVE] {split_name}: X={data.shape}  y={label.shape}")


# =========================================================
# Chi-square 피처 선택
# =========================================================
def select_features_chi2(X_tv, y_tv, n_features=N_FEATURES, save_dir=SAVE_DIR):
    """
    MinMaxScaler 후 chi-square로 상위 n_features개 선택.
    selected_features.json 저장 → CIC2018/CTU13에서 동일 피처 사용.
    """
    print(f"\n[CHI2] chi-square 피처 선택: {len(ML_FEATURES)}개 → {n_features}개")

    selector = SelectKBest(chi2, k=n_features)
    selector.fit(X_tv, y_tv)

    selected_indices = selector.get_support(indices=True).tolist()
    selected_names   = [ML_FEATURES[i] for i in selected_indices]

    scores = selector.scores_
    print(f"[CHI2] 선택된 {n_features}개 피처 (점수 높은 순):")
    top = sorted(zip(scores, ML_FEATURES), reverse=True)[:n_features]
    for rank, (score, feat) in enumerate(top, 1):
        print(f"  {rank:2d}. {feat:<42s}  chi2={score:.2f}")

    sel_path = os.path.join(save_dir, "selected_features.json")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(
            {"method": "chi2", "n_features": n_features,
             "indices": selected_indices, "features": selected_names},
            f, indent=4, ensure_ascii=False,
        )
    print(f"[CHI2] 저장: {sel_path}  (CIC2018/CTU13에서 공유)")
    return selected_indices, selected_names


# =========================================================
# main
# =========================================================
def main():
    print("=" * 65)
    print("  CIC-IDS2017 Preprocessing  (Chi-square 피처 선택 포함)")
    print("=" * 65)
    print(f"  RAW_DIR     = {RAW_DIR}")
    print(f"  SAVE_DIR    = {SAVE_DIR}")
    print(f"  ML_FEATURES = {len(ML_FEATURES)}개 → chi2 선택 후 {N_FEATURES}개")
    print("=" * 65)

    FLAT_DIR = os.path.join(SAVE_DIR, "flat")
    SEQ_DIR  = os.path.join(SAVE_DIR, "seq")
    for d in [FLAT_DIR, SEQ_DIR]:
        os.makedirs(d, exist_ok=True)

    df = load_all_csv(RAW_DIR)
    df = normalize_column_names(df)
    validate_columns(df)
    df = basic_cleaning(df)
    df = create_binary_label(df)
    df = normalize_protocol_column(df)
    df = sort_by_time(df)
    df = apply_log_transform(df)

    df_trainval, df_test = split_random(df)

    X_tv, y_tv = create_flow_data(df_trainval, ML_FEATURES)
    X_te, y_te = create_flow_data(df_test,     ML_FEATURES)

    # MinMaxScaler: trainval fit → [0,1] (chi2 요건)
    scaler = MinMaxScaler()
    X_tv   = scaler.fit_transform(X_tv).astype(np.float32)
    X_te   = scaler.transform(X_te).astype(np.float32)

    scaler_path = os.path.join(SEQ_DIR, "scaler_flow.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\n[SCALER] 저장: {scaler_path}")

    # Chi-square 피처 선택 (trainval 기준)
    selected_indices, selected_names = select_features_chi2(X_tv, y_tv)

    X_tv   = X_tv[:, selected_indices].astype(np.float32)
    X_te   = X_te[:, selected_indices].astype(np.float32)
    n_feat = N_FEATURES

    print(f"\n[CHI2] 선택 후: trainval={X_tv.shape}  test={X_te.shape}")

    # 저장
    save_numpy(X_tv,                    y_tv, FLAT_DIR, "trainval")
    save_numpy(X_te,                    y_te, FLAT_DIR, "test")
    save_numpy(X_tv.reshape(-1,n_feat,1), y_tv, SEQ_DIR,  "trainval")
    save_numpy(X_te.reshape(-1,n_feat,1), y_te, SEQ_DIR,  "test")

    print(f"\n[DONE]")
    print(f"  flat/ X_trainval : {X_tv.shape}")
    print(f"  seq/  X_trainval : {X_tv.reshape(-1,n_feat,1).shape}")
    print(f"  selected_features.json : {N_FEATURES}개 피처")
    print(f"\n[NEXT]  python preprocess_cicids2018.py  /  python preprocess_ctu13.py")


if __name__ == "__main__":
    main()