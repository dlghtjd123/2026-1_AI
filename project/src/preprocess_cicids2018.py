"""
preprocess_cicids2018.py

CSE-CIC-IDS2018 Friday-02-03-2018 파일에서
Bot + Benign 샘플만 추출하여 CIC-IDS2017 모델 입력 형식으로 변환한다.

전처리 방식: flow 1개 = 샘플 1개 (논문들과 동일)
  - RF / XGBoost : (n_flows, n_features)
  - CNN-LSTM/GRU : (n_flows, n_features, 1)

호환 조건 (preprocess_cicids2017.py 와 반드시 일치):
  - ML_FEATURES (77개)
  - LOG_TRANSFORM_FEATURES
  - Scaler: CIC2017 MinMaxScaler transform only (fit 금지)
           + Secondary MinMaxScaler (CIC2018 분포 정렬)
           → 최종 범위 [0, 1] 유지 → threshold 0.5 유효
           → D'Hooge et al. (2020) + Safety 2025 방식
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

RAW_FILE  = BASE_DIR / "data" / "raw" / "cic-ids2018" / "Friday-02-03-2018.csv"
SAVE_ROOT = BASE_DIR / "data" / "processed" / "cicids2018"

# CIC2017 전처리 시 저장된 scaler 경로
SCALER_PATH = (
    BASE_DIR / "data" / "processed" / "cicids2017"
    / "seq" / "scaler_flow.pkl"
)


# =========================================================
# ML 피처셋 (77개) — preprocess_cicids2017.py 의 ML_FEATURES 와 완전히 동일
# =========================================================
ML_FEATURES: list[str] = [
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
# 로그 변환 대상 — preprocess_cicids2017.py 와 완전히 동일
# =========================================================
LOG_TRANSFORM_FEATURES: list[str] = [
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
# CIC-IDS2018 컬럼명 → CIC-IDS2017 컬럼명 매핑
# =========================================================
COLUMN_MAP: dict[str, str] = {
    "Total Fwd Packet":           "Total Fwd Packets",
    "Total Bwd packets":          "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min":          "Min Packet Length",
    "Packet Length Max":          "Max Packet Length",
    "CWR Flag Count":             "CWE Flag Count",
    "Fwd Segment Size Avg":       "Avg Fwd Segment Size",
    "Bwd Segment Size Avg":       "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg":         "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg":        "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg":          "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg":         "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg":        "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg":          "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes":         "Init_Win_bytes_forward",
    "Bwd Init Win Bytes":         "Init_Win_bytes_backward",
    "Fwd Act Data Pkts":          "act_data_pkt_fwd",
    "Fwd Seg Size Min":           "min_seg_size_forward",
}


# =========================================================
# 1. 로드
# =========================================================
def load_raw(file_path: Path) -> pd.DataFrame:
    print(f"[LOAD] {file_path}")
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=200_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"[LOAD] shape: {df.shape}  columns: {len(df.columns)}")
    return df


# =========================================================
# 2. 중복 컬럼 교정
#    CIC-IDS2018 은 'Fwd Header Length' 가 두 번 등장하는 버그가 있다.
# =========================================================
def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    seen: dict[str, int] = {}
    new_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            if col == "Fwd Header Length":
                new_cols.append("Bwd Header Length")
                print("[FIX] 중복 컬럼 'Fwd Header Length' → 'Bwd Header Length' 교정")
            else:
                new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


# =========================================================
# 3. 컬럼명 변환
# =========================================================
def apply_column_map(df: pd.DataFrame) -> pd.DataFrame:
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)
    print(f"[MAP] 컬럼명 변환: {len(rename)}개")
    unmatched = [k for k in COLUMN_MAP if k not in rename]
    if unmatched:
        print(f"[MAP] 파일에 없는 원본 키: {unmatched}")
    return df


# =========================================================
# 4. Bot + Benign 필터링
# =========================================================
def filter_bot_benign(df: pd.DataFrame) -> pd.DataFrame:
    label_col = next(
        (c for c in df.columns if c.lower() == "label"), None
    )
    if label_col is None:
        raise ValueError(f"Label 컬럼을 찾을 수 없습니다.\n컬럼: {list(df.columns)}")

    print(f"\n[LABEL] 전체 라벨 분포 (label column: '{label_col}'):")
    print(df[label_col].value_counts(dropna=False).to_string())

    if "Attempted Category" in df.columns:
        attempted_mask = df["Attempted Category"] != -1
        n_attempted = attempted_mask.sum()
        if n_attempted > 0:
            print(f"\n[ATTEMPTED] Attempted 플로우 {n_attempted:,}개 → Benign으로 재라벨링")
            df.loc[attempted_mask, label_col] = "BENIGN"

    label_lower = df[label_col].str.strip().str.lower()
    # CIC2017: "Botnet ARES" → startswith("botnet")
    # CIC2018: "Bot"         → startswith("bot")
    # startswith("bot")으로 둘 다 포함
    bot_mask    = label_lower.str.startswith("bot")
    benign_mask = label_lower.isin(["benign"])

    df_f = df[bot_mask | benign_mask].copy()
    df_f["Label_str"]    = df_f[label_col].str.strip()
    df_f["Label_binary"] = bot_mask[bot_mask | benign_mask].astype(np.int32).values

    print(f"\n[FILTER] Bot + Benign 필터링 후:")
    print(df_f["Label_str"].value_counts().to_string())
    print(f"  Bot(1): {df_f['Label_binary'].sum():,}  "
          f"Benign(0): {(df_f['Label_binary'] == 0).sum():,}")

    return df_f.reset_index(drop=True)


# =========================================================
# 5. 누락 피처 파생
# =========================================================
def derive_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.zeros(len(df)), dtype=np.float32)
        return pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "Total Length of Bwd Packets" not in df.columns:
        df["Total Length of Bwd Packets"] = (
            _num("Bwd Packet Length Mean") * _num("Total Backward Packets")
        )
        print("[DERIVE] Total Length of Bwd Packets 생성")

    if "Protocol" not in df.columns:
        df["Protocol"] = 0
        print("[DERIVE] Protocol → 0")

    return df


# =========================================================
# 6. Protocol 처리
# =========================================================
def normalize_protocol_column(df: pd.DataFrame) -> pd.DataFrame:
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")
    df["Protocol"] = df["Protocol"].fillna(-1).astype(np.int32)
    return df


# =========================================================
# 7. 수치형 정제
# =========================================================
def clean_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ML_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼 {len(missing)}개 누락:\n  {missing}\n"
            "fix_duplicate_columns() 또는 derive_missing_features()를 확인하세요."
        )

    for col in ML_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[ML_FEATURES] = df[ML_FEATURES].fillna(0)
    return df


# =========================================================
# 8. 로그 변환 — preprocess_cicids2017.py 와 동일 조건
# =========================================================
def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    targets = [c for c in LOG_TRANSFORM_FEATURES if c in df.columns]
    print(f"[LOG] 변환 피처: {len(targets)}개")
    for col in targets:
        df[col] = np.log1p(np.maximum(df[col].values, 0))
    return df


# =========================================================
# 9. flow 단위 데이터 생성 — preprocess_cicids2017.py 와 동일 방식
#    flow 1개 = 샘플 1개
# =========================================================
def create_flow_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    flow 단위 데이터 생성

    출력:
      X : (n_flows, n_features)        ← RF / XGBoost 용
      X.reshape(-1, n_features, 1)     ← CNN-LSTM / GRU 용 (main 에서 reshape)
      y : (n_flows,)
    """
    X = df[ML_FEATURES].values.astype(np.float32)
    y = df["Label_binary"].values.astype(np.int32)

    print(f"\n[FLOW] X shape: {X.shape}")
    print(f"[FLOW] Botnet 비율: {y.mean():.4f} ({y.sum():,}/{len(y):,})")

    return X, y


# =========================================================
# =========================================================
# 10. Scaler 적용
#     ① CIC2017 MinMaxScaler (transform only — fit 금지)
#     ② Secondary MinMaxScaler (CIC2018 분포 정렬)
#        → 최종 범위 [0, 1] 유지 → threshold 0.5 유효
# =========================================================
def apply_scaler(X: np.ndarray) -> np.ndarray:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"CIC-IDS2017 scaler 없음: {SCALER_PATH}\n"
            "먼저 preprocess_cicids2017.py 를 실행하세요."
        )

    if X.shape[1] != len(ML_FEATURES):
        raise ValueError(
            f"피처 수 불일치: X={X.shape[1]}  ML_FEATURES={len(ML_FEATURES)}"
        )

    # ① CIC2017 MinMaxScaler (transform only)
    cic_scaler = joblib.load(SCALER_PATH)
    print(f"\n[SCALER] CIC-IDS2017 MinMaxScaler 로드: {SCALER_PATH}")
    print("[SCALER] transform only (fit 금지 — 데이터 누수 방지)")
    X_scaled = cic_scaler.transform(X).astype(np.float32)

    # ② Secondary MinMaxScaler (CIC2018 분포 정렬)
    aligner   = MinMaxScaler()
    X_aligned = aligner.fit_transform(X_scaled).astype(np.float32)

    aligner_path = SAVE_ROOT / "aligner.pkl"
    joblib.dump(aligner, aligner_path)

    print(f"[ALIGN] CIC2018 Secondary MinMaxScaler 완료")
    print(f"[ALIGN] 최종 범위: [{X_aligned.min():.4f}, {X_aligned.max():.4f}]")
    print(f"[ALIGN] aligner 저장: {aligner_path}")
    print(f"[SCALER] 최종 shape: {X_aligned.shape}")

    return X_aligned


# =========================================================
# 11. 저장
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

    bot_mask    = df["Label_binary"] == 1
    benign_mask = df["Label_binary"] == 0

    meta = {
        "dataset":         "CSE-CIC-IDS2018",
        "source_file":     str(RAW_FILE),
        "preprocessing": {
            "mode":        "flow_based",
            "description": "flow 1개 = 샘플 1개 (논문들과 동일)",
        },
        "num_features":    n_feat,
        "feature_columns": ML_FEATURES,
        "filter":          "Bot + Benign only",
        "scaler":          str(SCALER_PATH),
        "flow_level": {
            "total":        int(len(df)),
            "bot":          int(bot_mask.sum()),
            "benign":       int(benign_mask.sum()),
            "bot_ratio":    float(bot_mask.mean()),
        },
        "flat_shape": list(X_scaled.shape),
        "seq_shape":  [len(X_scaled), n_feat, 1],
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=4, ensure_ascii=False)

    print(f"\n[SAVE] {SAVE_ROOT}")
    print(f"  flat/X_test.npy  shape={X_scaled.shape}          ← RF/XGB")
    print(f"  seq/X_test.npy   shape={X_scaled.reshape(-1, n_feat, 1).shape}   ← CNN-LSTM/GRU")
    print(f"  meta.json")


# =========================================================
# main
# =========================================================
def main() -> None:
    print("=" * 60)
    print("  CSE-CIC-IDS2018 Bot Preprocessing  (Flow-Based)")
    print("=" * 60)
    print(f"  RAW_FILE    = {RAW_FILE}")
    print(f"  SAVE_ROOT   = {SAVE_ROOT}")
    print(f"  SCALER      = {SCALER_PATH}")
    print(f"  ML_FEATURES = {len(ML_FEATURES)}개")
    print("=" * 60)

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"원본 파일 없음: {RAW_FILE}")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. 로드
    df = load_raw(RAW_FILE)

    # 2. 중복 컬럼 교정
    df = fix_duplicate_columns(df)

    # 3. 컬럼명 변환 (CIC-IDS2018 → CIC-IDS2017 스타일)
    df = apply_column_map(df)

    # 4. Bot + Benign 필터링
    df = filter_bot_benign(df)

    # 5. 누락 피처 파생
    df = derive_missing_features(df)

    # 6. Protocol 처리
    df = normalize_protocol_column(df)

    # 7. 수치형 정제
    df = clean_numeric_features(df)

    # 8. 로그 변환 (CIC2017 과 동일 조건)
    df = apply_log_transform(df)

    # 9. flow 단위 데이터 생성
    X, y = create_flow_data(df)

    # 10. CIC2017 scaler 적용 (transform only)
    X_scaled = apply_scaler(X)

    # 11. 저장
    save_outputs(X_scaled, y, df)

    print("\n[DONE]")
    print("  다음 단계: evaluate.py 에서 flat/ 과 seq/ 경로로 교차 검증 수행")


if __name__ == "__main__":
    main()