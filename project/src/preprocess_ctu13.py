"""
preprocess_ctu13.py

CTU-13 시나리오 9 전처리 — 논문 방식 (Safety 2025 동일)

전처리 방식: flow 1개 = 샘플 1개
  - RF / XGBoost : (n_flows, n_features)
  - CNN-LSTM/GRU : (n_flows, n_features, 1)

Scaler 방식 (D'Hooge et al., 2020 + Safety 2025):
  ① CIC-IDS2017 MinMaxScaler transform only (fit 금지)
  ② Secondary MinMaxScaler (CTU-13 분포 재정렬)
     → 두 데이터셋 간 feature 분포 mismatch 감소
     → 최종 범위 [0, 1] 유지 → threshold 0.5 유효
     → 논문 작성 시 "target dataset 통계 사용" 명시 필요

라벨 방식:
  - Botnet IP 하드코딩 기반
  - Botnet IP → 1 / 그 외 → 0
  - Background 트래픽: CICFlowMeter CSV에는 라벨 없음
    → non-botnet IP 전체를 Normal(0)으로 처리 (허용 가능한 단순화)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
BASE_DIR  = _SRC_DIR.parent

RAW_DIR   = BASE_DIR / "data" / "raw" / "ctu-13"
SAVE_ROOT = BASE_DIR / "data" / "processed" / "ctu13"

# CIC-IDS2017 scenario9 CICFlowMeter CSV
RAW_CSV = RAW_DIR / "scenario9_raw.csv"

# CIC-IDS2017 scaler (transform only)
SCALER_PATH = (
    BASE_DIR / "data" / "processed" / "cicids2017"
    / "seq" / "scaler_flow.pkl"
)

# CTU-13 시나리오 9 봇넷 IP
BOTNET_IPS: set[str] = {
    "147.32.84.165",
    "147.32.84.191",
    "147.32.84.192",
    "147.32.84.193",
    "147.32.84.204",
    "147.32.84.205",
}


# =========================================================
# CICFlowMeter 컬럼명 → CIC-IDS2017 컬럼명 매핑
# =========================================================
COLUMN_MAP: dict[str, str] = {
    "src_ip":            "Source IP",
    "timestamp":         "Timestamp",
    "protocol":          "Protocol",
    "flow_duration":     "Flow Duration",
    "tot_fwd_pkts":      "Total Fwd Packets",
    "tot_bwd_pkts":      "Total Backward Packets",
    "totlen_fwd_pkts":   "Total Length of Fwd Packets",
    "totlen_bwd_pkts":   "Total Length of Bwd Packets",
    "fwd_pkt_len_max":   "Fwd Packet Length Max",
    "fwd_pkt_len_min":   "Fwd Packet Length Min",
    "fwd_pkt_len_mean":  "Fwd Packet Length Mean",
    "fwd_pkt_len_std":   "Fwd Packet Length Std",
    "bwd_pkt_len_max":   "Bwd Packet Length Max",
    "bwd_pkt_len_min":   "Bwd Packet Length Min",
    "bwd_pkt_len_mean":  "Bwd Packet Length Mean",
    "bwd_pkt_len_std":   "Bwd Packet Length Std",
    "flow_byts_s":       "Flow Bytes/s",
    "flow_pkts_s":       "Flow Packets/s",
    "flow_iat_mean":     "Flow IAT Mean",
    "flow_iat_std":      "Flow IAT Std",
    "flow_iat_max":      "Flow IAT Max",
    "flow_iat_min":      "Flow IAT Min",
    "fwd_iat_tot":       "Fwd IAT Total",
    "fwd_iat_mean":      "Fwd IAT Mean",
    "fwd_iat_std":       "Fwd IAT Std",
    "fwd_iat_max":       "Fwd IAT Max",
    "fwd_iat_min":       "Fwd IAT Min",
    "bwd_iat_tot":       "Bwd IAT Total",
    "bwd_iat_mean":      "Bwd IAT Mean",
    "bwd_iat_std":       "Bwd IAT Std",
    "bwd_iat_max":       "Bwd IAT Max",
    "bwd_iat_min":       "Bwd IAT Min",
    "fwd_psh_flags":     "Fwd PSH Flags",
    "bwd_psh_flags":     "Bwd PSH Flags",
    "fwd_urg_flags":     "Fwd URG Flags",
    "bwd_urg_flags":     "Bwd URG Flags",
    "fwd_header_len":    "Fwd Header Length",
    "bwd_header_len":    "Bwd Header Length",
    "fwd_pkts_s":        "Fwd Packets/s",
    "bwd_pkts_s":        "Bwd Packets/s",
    "pkt_len_min":       "Min Packet Length",
    "pkt_len_max":       "Max Packet Length",
    "pkt_len_mean":      "Packet Length Mean",
    "pkt_len_std":       "Packet Length Std",
    "pkt_len_var":       "Packet Length Variance",
    "fin_flag_cnt":      "FIN Flag Count",
    "syn_flag_cnt":      "SYN Flag Count",
    "rst_flag_cnt":      "RST Flag Count",
    "psh_flag_cnt":      "PSH Flag Count",
    "ack_flag_cnt":      "ACK Flag Count",
    "urg_flag_cnt":      "URG Flag Count",
    "cwr_flag_count":    "CWE Flag Count",
    "ece_flag_cnt":      "ECE Flag Count",
    "down_up_ratio":     "Down/Up Ratio",
    "pkt_size_avg":      "Average Packet Size",
    "fwd_seg_size_avg":  "Avg Fwd Segment Size",
    "bwd_seg_size_avg":  "Avg Bwd Segment Size",
    "fwd_byts_b_avg":    "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg":    "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg":  "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg":    "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg":    "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg":  "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts":  "Subflow Fwd Packets",
    "subflow_fwd_byts":  "Subflow Fwd Bytes",
    "subflow_bwd_pkts":  "Subflow Bwd Packets",
    "subflow_bwd_byts":  "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min":  "min_seg_size_forward",
    "active_mean":       "Active Mean",
    "active_std":        "Active Std",
    "active_max":        "Active Max",
    "active_min":        "Active Min",
    "idle_mean":         "Idle Mean",
    "idle_std":          "Idle Std",
    "idle_max":          "Idle Max",
    "idle_min":          "Idle Min",
}


# =========================================================
# ML 피처셋 (77개) — preprocess_cicids2017.py 와 완전히 동일
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
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
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
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
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
# 1. 로드
# =========================================================
def load_raw(path: Path) -> pd.DataFrame:
    print(f"[LOAD] {path}")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"[LOAD] shape: {df.shape}")
    return df


# =========================================================
# 2. 컬럼명 변환
# =========================================================
def apply_column_map(df: pd.DataFrame) -> pd.DataFrame:
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)
    print(f"[MAP] 컬럼명 변환: {len(rename)}개")
    missing = [k for k in COLUMN_MAP if k not in rename]
    if missing:
        print(f"[MAP] 없는 원본 키: {missing}")
    return df


# =========================================================
# 3. 라벨 부여 — Botnet IP 기반
# =========================================================
def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Source IP가 BOTNET_IPS에 있으면 1, 아니면 0
    Background 트래픽은 CICFlowMeter CSV에서 구분 불가
    → non-botnet IP 전체를 Normal(0)으로 처리
    """
    if "Source IP" not in df.columns:
        raise ValueError("'Source IP' 컬럼 없음. COLUMN_MAP 확인 필요")

    df["Label_binary"] = df["Source IP"].apply(
        lambda ip: 1 if str(ip).strip() in BOTNET_IPS else 0
    ).astype(np.int32)

    print(f"\n[LABEL] Botnet IP: {sorted(BOTNET_IPS)}")
    print(f"[LABEL] Botnet(1): {df['Label_binary'].sum():,}")
    print(f"[LABEL] Normal(0): {(df['Label_binary'] == 0).sum():,}")
    print(f"[LABEL] Botnet 비율: {df['Label_binary'].mean():.4f}")
    return df


# =========================================================
# 4. 수치형 정제
# =========================================================
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ML_FEATURES if c not in df.columns]
    if missing:
        print(f"\n[WARN] 없는 피처 {len(missing)}개: {missing}")

    available = [c for c in ML_FEATURES if c in df.columns]
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[available] = df[available].fillna(0)

    if "Protocol" in df.columns:
        df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")
        df["Protocol"] = df["Protocol"].fillna(-1).astype(np.int32)

    return df


# =========================================================
# 5. 로그 변환 — CIC-IDS2017 와 동일
# =========================================================
def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    targets = [c for c in LOG_TRANSFORM_FEATURES if c in df.columns]
    print(f"[LOG] 변환 피처: {len(targets)}개")
    for col in targets:
        df[col] = np.log1p(np.maximum(df[col].values, 0))
    return df


# =========================================================
# 6. Scaler 적용
#    ① CIC2017 MinMaxScaler transform (fit 금지)
#    ② Secondary MinMaxScaler (CTU-13 분포 재정렬)
#       → 최종 범위 [0, 1] 유지 → threshold 0.5 유효
# =========================================================
def apply_scaler_with_alignment(X: np.ndarray) -> tuple[np.ndarray, object]:
    """
    D'Hooge et al. (2020) + Safety 2025 scaler realignment 방식

    ① CIC2017 MinMaxScaler → [0, 1] 변환
    ② Secondary MinMaxScaler → CTU-13 분포 기준 재정렬 후 [0, 1] 유지

    논문 작성 시 명시:
      "MinMaxScaler (D'Hooge et al., 2020) 적용 후
       target dataset 통계를 사용한 secondary MinMaxScaler 적용
       (Safety 2025 방식) — Strict zero-shot 아님"
    """
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"CIC-IDS2017 scaler 없음: {SCALER_PATH}\n"
            "먼저 preprocess_cicids2017.py 를 실행하세요."
        )

    # ① CIC2017 MinMaxScaler
    cic_scaler = joblib.load(SCALER_PATH)
    print(f"\n[SCALER] CIC-IDS2017 MinMaxScaler 로드: {SCALER_PATH}")
    X_scaled = cic_scaler.transform(X).astype(np.float32)

    # ② Secondary MinMaxScaler — CTU-13 분포 재정렬 + [0,1] 유지
    aligner   = MinMaxScaler()
    X_aligned = aligner.fit_transform(X_scaled).astype(np.float32)

    aligner_path = SAVE_ROOT / "aligner.pkl"
    joblib.dump(aligner, aligner_path)

    print(f"[ALIGN] CTU-13 Secondary MinMaxScaler 완료")
    print(f"[ALIGN] 최종 범위: [{X_aligned.min():.4f}, {X_aligned.max():.4f}]")
    print(f"[ALIGN] aligner 저장: {aligner_path}")
    print(f"[SCALER] 최종 shape: {X_aligned.shape}")

    return X_aligned


# =========================================================
# 7. 저장
# =========================================================
def save_outputs(
    X_scaled: np.ndarray,
    y:        np.ndarray,
    df:       pd.DataFrame,
) -> None:
    n_feat = len(ML_FEATURES)

    flat_dir = SAVE_ROOT / "flat"
    seq_dir  = SAVE_ROOT / "seq"
    flat_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    # RF / XGBoost: (n_flows, n_features)
    np.save(flat_dir / "X_test.npy", X_scaled)
    np.save(flat_dir / "y_test.npy", y)

    # CNN-LSTM / GRU: (n_flows, n_features, 1)
    np.save(seq_dir / "X_test.npy", X_scaled.reshape(-1, n_feat, 1))
    np.save(seq_dir / "y_test.npy", y)

    meta = {
        "dataset":       "CTU-13 Scenario 9",
        "botnet_type":   "Neris (IRC-based, 10 bots)",
        "source_csv":    str(RAW_CSV),
        "botnet_ips":    sorted(BOTNET_IPS),
        "preprocessing": {
            "mode":         "flow_based",
            "description":  "flow 1개 = 샘플 1개 (논문들과 동일)",
            "background":   "non-botnet IP → Normal(0) 처리 (구분 불가)",
            "scaler":       "CIC2017 MinMaxScaler + secondary MinMaxScaler (D'Hooge + Safety 2025)",
        },
        "scaler_path":   str(SCALER_PATH),
        "num_features":  n_feat,
        "label_dist": {
            "total":       int(len(y)),
            "botnet":      int(y.sum()),
            "normal":      int((y == 0).sum()),
            "bot_ratio":   float(y.mean()),
        },
        "flat_shape":    list(X_scaled.shape),
        "seq_shape":     [len(X_scaled), n_feat, 1],
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=4, ensure_ascii=False)

    print(f"\n[SAVE] {SAVE_ROOT}")
    print(f"  flat/X_test.npy  shape={X_scaled.shape}             ← RF/XGB")
    print(f"  seq/X_test.npy   shape={X_scaled.reshape(-1, n_feat, 1).shape}  ← CNN-LSTM/GRU")
    print(f"  meta.json")


# =========================================================
# main
# =========================================================
def main() -> None:
    print("=" * 60)
    print("  CTU-13 Scenario 9  전처리  (Flow-Based, Safety 2025)")
    print("=" * 60)
    print(f"  RAW_CSV     = {RAW_CSV}")
    print(f"  SAVE_ROOT   = {SAVE_ROOT}")
    print(f"  SCALER      = {SCALER_PATH}")
    print(f"  BOTNET_IPS  = {sorted(BOTNET_IPS)}")
    print(f"  ML_FEATURES = {len(ML_FEATURES)}개")
    print("=" * 60)

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"CSV 없음: {RAW_CSV}")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. 로드
    df = load_raw(RAW_CSV)

    # 2. 컬럼명 변환
    df = apply_column_map(df)

    # 3. 라벨 부여
    df = assign_labels(df)

    # 4. 수치형 정제
    df = clean_features(df)

    # 5. 로그 변환
    df = apply_log_transform(df)

    # 6. flow 단위 feature 추출
    available = [c for c in ML_FEATURES if c in df.columns]
    X = df[available].values.astype(np.float32)
    y = df["Label_binary"].values.astype(np.int32)

    # 누락 피처 → 0 패딩
    if len(available) < len(ML_FEATURES):
        missing_count = len(ML_FEATURES) - len(available)
        print(f"\n[WARN] 누락 피처 {missing_count}개 → 0으로 패딩")
        X = np.hstack([X, np.zeros((len(X), missing_count), dtype=np.float32)])

    print(f"\n[FLOW] X shape: {X.shape}")
    print(f"[FLOW] Botnet 비율: {y.mean():.4f} ({y.sum():,}/{len(y):,})")

    # 7. CIC2017 scaler + 분포 정렬
    X_scaled = apply_scaler_with_alignment(X)

    # 8. 저장
    save_outputs(X_scaled, y, df)

    print("\n[DONE]")
    print("  다음 단계: evaluate.py 에서 ctu13 경로로 교차 검증 수행")


if __name__ == "__main__":
    main()