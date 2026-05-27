"""
preprocess_cicids2018.py

CSE-CIC-IDS2018 Friday-02-03-2018 — K-fold 내부 검증 대응
Chi-square 피처 선택: CIC-IDS2017에서 선택된 20개 피처 공유 적용

저장 구조:
  data/processed/cicids2018/
    flat/  X_trainval.npy  y_trainval.npy  X_test.npy  y_test.npy
    seq/   X_trainval.npy  y_trainval.npy  X_test.npy  y_test.npy

Scaler:
  ① CIC2017 MinMaxScaler (transform only)
  ② Secondary MinMaxScaler (trainval fit, test transform)
  ③ selected_features.json 에서 20개 피처 인덱스 로드 후 적용
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

RAW_FILE  = BASE_DIR / "data" / "raw" / "cic-ids2018" / "Friday-02-03-2018.csv"
SAVE_ROOT = BASE_DIR / "data" / "processed" / "cicids2018"
CIC17_DIR = BASE_DIR / "data" / "processed" / "cicids2017"

SCALER_PATH   = CIC17_DIR / "seq" / "scaler_flow.pkl"
SEL_FEAT_PATH = CIC17_DIR / "selected_features.json"   # chi2 선택 피처


# =========================================================
# ML 피처셋 (77개 원본 — CIC2017과 동일)
# =========================================================
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

COLUMN_MAP: dict[str, str] = {
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}


# =========================================================
# 전처리 함수들 (기존과 동일)
# =========================================================
def load_raw(file_path):
    print(f"[LOAD] {file_path}")
    chunks = []
    keep_cols = list(dict.fromkeys(["Label"] + ML_FEATURES))
    for chunk in pd.read_csv(file_path, chunksize=200_000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        chunk = fix_duplicate_columns(chunk)
        rename = {k: v for k, v in COLUMN_MAP.items() if k in chunk.columns}
        chunk.rename(columns=rename, inplace=True)
        label_col = next((c for c in chunk.columns if c.lower() == "label"), None)
        present_cols = [col for col in keep_cols if col in chunk.columns]
        if label_col and label_col not in present_cols:
            present_cols.insert(0, label_col)
        chunk = chunk.loc[:, present_cols]
        for col in [c for c in ML_FEATURES if c in chunk.columns]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(np.float32)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"[LOAD] shape: {df.shape}")
    return df

def fix_duplicate_columns(df):
    cols = list(df.columns)
    seen: dict[str, int] = {}
    new_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            if col == "Fwd Header Length":
                new_cols.append("Bwd Header Length")
            else:
                new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df

def apply_column_map(df):
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)
    print(f"[MAP] 컬럼명 변환: {len(rename)}개")
    return df

def filter_bot_benign(df):
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError(f"Label 컬럼 없음")
    print(f"\n[LABEL] 전체 라벨 분포:")
    print(df[label_col].value_counts(dropna=False).to_string())
    label_lower = df[label_col].str.strip().str.lower()
    bot_mask    = label_lower.str.startswith("bot")
    benign_mask = label_lower.isin(["benign"])
    df_f = df[bot_mask | benign_mask].copy()
    df_f["Label_str"]    = df_f[label_col].str.strip()
    df_f["Label_binary"] = bot_mask[bot_mask | benign_mask].astype(np.int32).values
    print(f"\n[FILTER] Bot(1)={df_f['Label_binary'].sum():,}  Benign(0)={(df_f['Label_binary']==0).sum():,}")
    return df_f.reset_index(drop=True)

def derive_missing_features(df):
    def _num(col):
        if col not in df.columns:
            return pd.Series(np.zeros(len(df)), dtype=np.float32)
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "Total Length of Bwd Packets" not in df.columns:
        df["Total Length of Bwd Packets"] = _num("Bwd Packet Length Mean") * _num("Total Backward Packets")
    if "Protocol" not in df.columns:
        df["Protocol"] = 0
    return df

def normalize_protocol_column(df):
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(-1).astype(np.int32)
    return df

def clean_numeric_features(df):
    missing = [c for c in ML_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 {len(missing)}개 누락: {missing}")
    for col in ML_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[ML_FEATURES] = df[ML_FEATURES].fillna(0)
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

def create_flow_data(df):
    X = df[ML_FEATURES].values.astype(np.float32)
    y = df["Label_binary"].values.astype(np.int32)
    print(f"[FLOW] X={X.shape}  Botnet={y.mean():.4f}")
    return X, y

def apply_scaler(X_trainval, X_test):
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"CIC2017 scaler 없음: {SCALER_PATH}\n먼저 preprocess_cicids2017.py 실행")
    cic_scaler   = joblib.load(SCALER_PATH)
    cic_scaler.copy = False
    X_trainval = np.ascontiguousarray(X_trainval, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    X_tv_scaled  = cic_scaler.transform(X_trainval)
    X_te_scaled  = cic_scaler.transform(X_test)
    aligner      = MinMaxScaler(copy=False)
    X_tv_aligned = aligner.fit_transform(X_tv_scaled)
    X_te_aligned = aligner.transform(X_te_scaled)
    joblib.dump(aligner, SAVE_ROOT / "aligner.pkl")
    print(f"[SCALER] CIC2017 MinMaxScaler + Secondary MinMaxScaler 적용 완료")
    return X_tv_aligned, X_te_aligned


# =========================================================
# Chi-square 피처 선택 적용 (CIC2017에서 저장된 인덱스 사용)
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

    np.save(seq_dir / "X_trainval.npy", X_trainval.reshape(-1, n_feat, 1))
    np.save(seq_dir / "y_trainval.npy", y_trainval)
    np.save(seq_dir / "X_test.npy",     X_test.reshape(-1, n_feat, 1))
    np.save(seq_dir / "y_test.npy",     y_test)

    meta = {
        "dataset": "CSE-CIC-IDS2018", "source_file": str(RAW_FILE),
        "botnet_type": "Ares + Zeus (HTTP-based)", "filter": "Bot + Benign only",
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
    print("  CSE-CIC-IDS2018 전처리  (Chi-square 피처 선택 공유)")
    print("=" * 65)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    df = load_raw(RAW_FILE)
    df = fix_duplicate_columns(df)
    df = apply_column_map(df)
    df = filter_bot_benign(df)
    df = derive_missing_features(df)
    df = normalize_protocol_column(df)
    df = clean_numeric_features(df)
    df = apply_log_transform(df)

    df_trainval, df_test = split_random(df)
    X_tv, y_tv = create_flow_data(df_trainval)
    X_te, y_te = create_flow_data(df_test)
    del df, df_trainval, df_test

    # ① CIC2017 Scaler + ② Secondary Scaler
    X_tv, X_te = apply_scaler(X_tv, X_te)

    # ③ Chi-square 피처 선택 (CIC2017 기준 인덱스 적용)
    X_tv, X_te, n_feat = apply_feature_selection(X_tv, X_te)

    save_outputs(X_tv, y_tv, X_te, y_te, n_feat)
    print("\n[DONE]  다음 단계: train_rf.py --dataset cicids2018")


if __name__ == "__main__":
    main()
