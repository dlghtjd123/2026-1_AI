"""
preprocess_ctu13.py

CTU-13 시나리오 9 전처리 — K-fold 대응
Chi-square 피처 선택: CIC-IDS2017에서 선택된 20개 피처 공유 적용

저장 구조:
  data/processed/ctu13/
    flat/  X_trainval.npy  y_trainval.npy  X_test.npy  y_test.npy
    seq/   X_trainval.npy  y_trainval.npy  X_test.npy  y_test.npy

Scaler:
  ① CIC2017 MinMaxScaler (transform only)
  ② Secondary MinMaxScaler (trainval fit)
  ③ selected_features.json에서 20개 피처 인덱스 로드 후 적용
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
BASE_DIR  = _SRC_DIR.parent

RAW_DIR   = BASE_DIR / "data" / "raw" / "ctu-13"
SAVE_ROOT = BASE_DIR / "data" / "processed" / "ctu13"
RAW_CSV   = RAW_DIR / "scenario9_raw.csv"
CIC17_DIR = BASE_DIR / "data" / "processed" / "cicids2017"

SCALER_PATH   = CIC17_DIR / "seq" / "scaler_flow.pkl"
SEL_FEAT_PATH = CIC17_DIR / "selected_features.json"   # chi2 선택 피처

BOTNET_IPS: set[str] = {
    "147.32.84.165", "147.32.84.191", "147.32.84.192",
    "147.32.84.193", "147.32.84.204", "147.32.84.205",
}

COLUMN_MAP: dict[str, str] = {
    "src_ip": "Source IP", "timestamp": "Timestamp", "protocol": "Protocol",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets", "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max", "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean", "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max", "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean", "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s", "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean", "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max", "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total", "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std", "fwd_iat_max": "Fwd IAT Max", "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total", "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std", "bwd_iat_max": "Bwd IAT Max", "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags", "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags", "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length", "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s", "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length", "pkt_len_max": "Max Packet Length",
    "pkt_len_mean": "Packet Length Mean", "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count", "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count", "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count", "urg_flag_cnt": "URG Flag Count",
    "cwr_flag_count": "CWE Flag Count", "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio", "pkt_size_avg": "Average Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size", "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk", "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk", "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets", "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets", "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd", "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "Active Mean", "active_std": "Active Std",
    "active_max": "Active Max", "active_min": "Active Min",
    "idle_mean": "Idle Mean", "idle_std": "Idle Std",
    "idle_max": "Idle Max", "idle_min": "Idle Min",
}

ML_FEATURES: list[str] = [
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

LOG_TRANSFORM_FEATURES: list[str] = [
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


# =========================================================
# 전처리 함수들
# =========================================================
def load_raw(path):
    print(f"[LOAD] {path}")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"[LOAD] shape: {df.shape}")
    return df

def apply_column_map(df):
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)
    print(f"[MAP] 컬럼명 변환: {len(rename)}개")
    return df

def assign_labels(df):
    if "Source IP" not in df.columns:
        raise ValueError("'Source IP' 컬럼 없음")
    df["Label_binary"] = df["Source IP"].apply(
        lambda ip: 1 if str(ip).strip() in BOTNET_IPS else 0
    ).astype(np.int32)
    print(f"[LABEL] Botnet(1): {df['Label_binary'].sum():,}  Normal(0): {(df['Label_binary']==0).sum():,}")
    print(f"[LABEL] Botnet 비율: {df['Label_binary'].mean():.4f}")
    return df

def clean_features(df):
    available = [c for c in ML_FEATURES if c in df.columns]
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[available] = df[available].fillna(0)
    if "Protocol" in df.columns:
        df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(-1).astype(np.int32)
    return df

def apply_log_transform(df):
    targets = [c for c in LOG_TRANSFORM_FEATURES if c in df.columns]
    for col in targets:
        df[col] = np.log1p(np.maximum(df[col].values, 0))
    print(f"[LOG] 변환 피처: {len(targets)}개")
    return df

def split_random(df: pd.DataFrame, test_ratio: float = 0.2,
                 random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flow 단위 랜덤 Stratified split (Zhao et al. 2024 방식)
    봇넷 비율 유지 (stratify=Label_binary)
    """
    idx = np.arange(len(df))
    y   = df["Label_binary"].values
    try:
        tv_idx, te_idx = train_test_split(
            idx, test_size=test_ratio, random_state=random_state, stratify=y
        )
    except ValueError:
        print("[SPLIT] stratify 불가 → 일반 랜덤 split")
        tv_idx, te_idx = train_test_split(
            idx, test_size=test_ratio, random_state=random_state
        )
    df_tv = df.iloc[tv_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    print(f"[SPLIT] 랜덤 Stratified split  trainval: {len(df_tv):,} / test: {len(df_te):,}")
    for name, s in [("trainval", df_tv), ("test", df_te)]:
        print(f"  {name} Botnet 비율: {s['Label_binary'].mean():.4f}")
    return df_tv, df_te

def apply_scaler(X_trainval, X_test):
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"CIC2017 scaler 없음: {SCALER_PATH}")
    cic_scaler   = joblib.load(SCALER_PATH)
    X_tv_scaled  = cic_scaler.transform(X_trainval).astype(np.float32)
    X_te_scaled  = cic_scaler.transform(X_test).astype(np.float32)
    aligner      = MinMaxScaler()
    X_tv_aligned = aligner.fit_transform(X_tv_scaled).astype(np.float32)
    X_te_aligned = aligner.transform(X_te_scaled).astype(np.float32)
    joblib.dump(aligner, SAVE_ROOT / "aligner.pkl")
    print(f"[SCALER] CIC2017 MinMaxScaler + Secondary MinMaxScaler 완료")
    return X_tv_aligned, X_te_aligned


# =========================================================
# Chi-square 피처 선택 적용 (CIC2017 기준)
# =========================================================
def apply_feature_selection(X_tv, X_te):
    if not SEL_FEAT_PATH.exists():
        raise FileNotFoundError(
            f"selected_features.json 없음: {SEL_FEAT_PATH}\n"
            "먼저 preprocess_cicids2017.py 를 실행하세요."
        )
    with open(SEL_FEAT_PATH, "r") as f:
        sel = json.load(f)
    selected_indices = sel["indices"]
    selected_names   = sel["features"]
    n_features       = sel["n_features"]
    print(f"\n[CHI2] CIC2017 기준 {n_features}개 피처 적용:")
    for i, name in enumerate(selected_names, 1):
        print(f"  {i:2d}. {name}")
    X_tv_sel = X_tv[:, selected_indices].astype(np.float32)
    X_te_sel = X_te[:, selected_indices].astype(np.float32)
    print(f"[CHI2] 선택 후: trainval={X_tv_sel.shape}  test={X_te_sel.shape}")
    return X_tv_sel, X_te_sel, n_features


def save_outputs(X_trainval, y_trainval, X_test, y_test, n_feat):
    flat_dir = SAVE_ROOT / "flat"
    seq_dir  = SAVE_ROOT / "seq"
    flat_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    np.save(flat_dir / "X_trainval.npy", X_trainval)
    np.save(flat_dir / "y_trainval.npy", y_trainval)
    np.save(flat_dir / "X_test.npy",     X_test)
    np.save(flat_dir / "y_test.npy",     y_test)
    np.save(seq_dir  / "X_trainval.npy", X_trainval.reshape(-1, n_feat, 1))
    np.save(seq_dir  / "y_trainval.npy", y_trainval)
    np.save(seq_dir  / "X_test.npy",     X_test.reshape(-1, n_feat, 1))
    np.save(seq_dir  / "y_test.npy",     y_test)

    meta = {
        "dataset": "CTU-13 Scenario 9", "botnet_type": "Neris (IRC-based, 10 bots)",
        "botnet_ips": sorted(BOTNET_IPS),
        "feature_selection": {"method": "chi2", "n_features": n_feat, "source": "cicids2017"},
        "split": {"trainval": int(len(y_trainval)), "test": int(len(y_test)),
                  "trainval_bot_ratio": float(y_trainval.mean()), "test_bot_ratio": float(y_test.mean())},
    }
    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=4, ensure_ascii=False)

    print(f"\n[SAVE] {SAVE_ROOT}")
    print(f"  flat/ X_trainval: {X_trainval.shape}")
    print(f"  seq/  X_trainval: {X_trainval.reshape(-1, n_feat, 1).shape}")


# =========================================================
# main
# =========================================================
def main():
    print("=" * 65)
    print("  CTU-13 Scenario 9 전처리  (Chi-square 피처 선택 공유)")
    print("=" * 65)
    print(f"  RAW_CSV    = {RAW_CSV}")
    print(f"  BOTNET_IPS = {sorted(BOTNET_IPS)}")
    print("=" * 65)

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"CSV 없음: {RAW_CSV}")
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    df = load_raw(RAW_CSV)
    df = apply_column_map(df)
    df = assign_labels(df)
    df = clean_features(df)
    df = apply_log_transform(df)

    df_trainval, df_test = split_random(df)

    available = [c for c in ML_FEATURES if c in df.columns]
    X_tv = df_trainval[available].values.astype(np.float32)
    y_tv = df_trainval["Label_binary"].values.astype(np.int32)
    X_te = df_test[available].values.astype(np.float32)
    y_te = df_test["Label_binary"].values.astype(np.int32)

    if len(available) < len(ML_FEATURES):
        missing_count = len(ML_FEATURES) - len(available)
        print(f"\n[WARN] 누락 피처 {missing_count}개 → 0으로 패딩")
        X_tv = np.hstack([X_tv, np.zeros((len(X_tv), missing_count), dtype=np.float32)])
        X_te = np.hstack([X_te, np.zeros((len(X_te), missing_count), dtype=np.float32)])

    print(f"\n[FLOW] X_trainval: {X_tv.shape}  X_test: {X_te.shape}")

    # ① CIC2017 Scaler + ② Secondary Scaler
    X_tv, X_te = apply_scaler(X_tv, X_te)

    # ③ Chi-square 피처 선택 (CIC2017 기준 인덱스 적용)
    X_tv, X_te, n_feat = apply_feature_selection(X_tv, X_te)

    save_outputs(X_tv, y_tv, X_te, y_te, n_feat)
    print("\n[DONE]  다음 단계: train_rf.py --dataset ctu13")


if __name__ == "__main__":
    main()