# preprocess_ctu13.py
# 목적:
# - CTU-13 scenario1.csv, scenario9.csv를 외부 검증용 입력으로 변환
# - CIC-IDS2017과 교차 검증이 가능한 COMMON_FEATURES 8개 기준으로 변환
# - data/processed/ctu13/{scenario}/winflat, seq 에 저장
#
# [수정 이력]
# 1. TREAT_TO_BOTNET_AS_ATTACK = True 로 변경
# 2. Flow IAT Max 제거 (COMMON_FEATURES 8개)
# 3. groupby 기준을 (Source IP, 날짜)로 변경
# 4. 클래스 불균형 통계를 meta.json에 저장
# 5. BASE_DIR 경로 수정 (_SRC_DIR.parent로 project/ 를 가리키도록)
# 6. SEQ_SCALER_PATH 경로 수정 (cicids/seq_common/ 기준으로)

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 1. 설정
# =========================================================

_SRC_DIR = Path(__file__).resolve().parent   # .../project/src
BASE_DIR = _SRC_DIR.parent                   # .../project

CTU_RAW_DIR = BASE_DIR / "data" / "raw" / "ctu-13"
SAVE_ROOT   = BASE_DIR / "data" / "processed" / "ctu13"

SCENARIOS = [
    "scenario1.csv",
    "scenario9.csv",
]

WINDOW_SIZE = 15
STEP_SIZE   = 5

# preprocess_cicids.py 실행 후 생성되는 CIC common scaler 경로
SEQ_SCALER_PATH  = BASE_DIR / "data" / "processed" / "cicids" / "seq_common" / "scaler_seq_common_w15.pkl"
APPLY_SEQ_SCALER = False   # True로 바꾸면 CIC에서 fit한 scaler를 CTU에 적용한다.

SAVE_SAMPLE_IDS  = True

# To-Botnet 트래픽을 Botnet(1)으로 처리
# C&C 서버 → 감염 호스트 방향의 명령 트래픽이므로 봇넷 탐지 목적상 1이 맞다.
TREAT_TO_BOTNET_AS_ATTACK = True


# =========================================================
# 2. 공통 feature 정의 (CIC-IDS2017과 교차 검증용)
# ---------------------------------------------------------
# preprocess_cicids.py의 COMMON_FEATURES와 반드시 동일해야 한다.
# =========================================================
COMMON_FEATURES = [
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Average Packet Size",
    "Flow IAT Mean",
]


# =========================================================
# 3. 유틸 함수
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """후보 컬럼명 중 실제 존재하는 첫 컬럼을 반환한다."""
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


def parse_protocol(value) -> int:
    """CIC 스타일 숫자 프로토콜로 변환. TCP=6, UDP=17, ICMP=1, 그 외 0"""
    if pd.isna(value):
        return 0

    s = str(value).strip().lower()

    if s in {"tcp", "6"}:   return 6
    if s in {"udp", "17"}:  return 17
    if s in {"icmp", "1"}:  return 1

    try:
        return int(float(s))
    except Exception:
        return 0


def parse_timestamp(series: pd.Series) -> pd.Series:
    """CTU StartTime / Timestamp 파싱."""
    ts = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
    return ts


def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default)


def label_to_binary(label_value: str) -> int:
    """
    CTU-13 레이블 이진화.
    1: From-Botnet, To-Botnet, Botnet, malicious
    0: Normal, Background, 그 외
    """
    if pd.isna(label_value):
        return 0

    s = str(label_value).strip().lower()

    if "to-botnet" in s:
        return 1 if TREAT_TO_BOTNET_AS_ATTACK else 0

    if any(k in s for k in ["from-botnet", "botnet", "malicious"]):
        return 1

    return 0


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0, np.nan)
    out = numerator / den
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================
# 4. CTU 컬럼 정규화
# =========================================================

def normalize_ctu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    CTU-13 raw CSV를 COMMON_FEATURES 기준 표준 컬럼으로 정리한다.

    Flow IAT Mean 근사: Flow Duration / (TotPkts - 1)
    Flow IAT Max는 근사 신뢰도 문제로 COMMON_FEATURES에서 제외.
    """
    df = df.copy()

    timestamp_col   = find_column(df, ["StartTime", "Timestamp", "starttime", "stime"])
    duration_col    = find_column(df, ["Dur", "Duration", "dur"])
    proto_col       = find_column(df, ["Proto", "Protocol", "proto"])
    src_ip_col      = find_column(df, ["SrcAddr", "Source IP", "Src IP", "srcaddr", "src_ip"])
    dst_ip_col      = find_column(df, ["DstAddr", "Destination IP", "Dst IP", "dstaddr", "dst_ip"])
    label_col       = find_column(df, ["Label", "label"])
    total_pkts_col  = find_column(df, ["TotPkts", "Total Packets", "totpkts", "pkts"], required=False)
    total_bytes_col = find_column(df, ["TotBytes", "Total Bytes", "totbytes", "bytes"], required=False)
    src_pkts_col    = find_column(df, ["SrcPkts", "srcpkts", "spkts", "Src Pkts"], required=False)
    dst_pkts_col    = find_column(df, ["DstPkts", "dstpkts", "dpkts", "Dst Pkts"], required=False)

    df["Timestamp"]      = parse_timestamp(df[timestamp_col])
    df["Flow Duration"]  = to_numeric(df[duration_col])
    df["Protocol"]       = df[proto_col].apply(parse_protocol)
    df["Source IP"]      = df[src_ip_col].astype(str)
    df["Destination IP"] = df[dst_ip_col].astype(str)
    df["Label_raw"]      = df[label_col].astype(str)
    df["Label_binary"]   = df["Label_raw"].apply(label_to_binary)

    if total_pkts_col is not None:
        df["TotPkts"] = to_numeric(df[total_pkts_col])
    else:
        raise KeyError("CTU CSV에서 총 패킷 수 컬럼(TotPkts)을 찾지 못했습니다.")

    if total_bytes_col is not None:
        df["TotBytes"] = to_numeric(df[total_bytes_col])
    else:
        raise KeyError("CTU CSV에서 총 바이트 수 컬럼(TotBytes)을 찾지 못했습니다.")

    if src_pkts_col is not None and dst_pkts_col is not None:
        df["Total Fwd Packets"]      = to_numeric(df[src_pkts_col])
        df["Total Backward Packets"] = to_numeric(df[dst_pkts_col])
    else:
        df["Total Fwd Packets"]      = df["TotPkts"]
        df["Total Backward Packets"] = 0.0
        print("[WARN] 방향별 패킷 수 컬럼이 없어 Total Fwd Packets=TotPkts, Total Backward Packets=0 으로 대체합니다.")

    df["Flow Bytes/s"]       = safe_divide(df["TotBytes"], df["Flow Duration"])
    df["Flow Packets/s"]     = safe_divide(df["TotPkts"],  df["Flow Duration"])
    df["Average Packet Size"] = safe_divide(df["TotBytes"], df["TotPkts"])

    denom = (df["TotPkts"] - 1).clip(lower=1)
    df["Flow IAT Mean"] = safe_divide(df["Flow Duration"], denom)

    df = df.replace([np.inf, -np.inf], np.nan)
    df[COMMON_FEATURES] = df[COMMON_FEATURES].fillna(0.0)

    before = len(df)
    df = df.dropna(subset=["Timestamp"]).copy()
    if len(df) < before:
        print(f"[INFO] Timestamp 파싱 실패 row 제거: {before - len(df)}개")

    return df


# =========================================================
# 5. 슬라이딩 윈도우 생성
# =========================================================

def _build_windows_for_group(
    features: np.ndarray,
    label_vals: np.ndarray,
    window_size: int,
    step_size: int,
    scenario_name: str,
    row_ids: np.ndarray,
) -> tuple[list, list, list, list]:
    """한 (Source IP, 날짜) 그룹에 대해 seq / winflat / y / sample_ids를 생성한다."""
    X_seq_list, X_flat_list, y_list, sample_id_list = [], [], [], []

    n = len(features)
    if n < window_size:
        return X_seq_list, X_flat_list, y_list, sample_id_list

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size

        x_seq     = features[start:end]
        x_flat    = x_seq.reshape(-1)
        y         = int(label_vals[start:end].max())
        sample_id = f"{scenario_name}|{row_ids[start]}|{row_ids[end - 1]}"

        X_seq_list.append(x_seq)
        X_flat_list.append(x_flat)
        y_list.append(y)
        sample_id_list.append(sample_id)

    return X_seq_list, X_flat_list, y_list, sample_id_list


# =========================================================
# 6. 시나리오 처리
# =========================================================

def process_one_scenario(csv_path: Path) -> None:
    """단일 CTU-13 시나리오 CSV를 처리하여 seq / winflat / meta.json을 저장한다."""
    scenario_name = csv_path.stem
    print(f"\n[PROCESS] {scenario_name}")

    save_dir    = SAVE_ROOT / scenario_name
    seq_dir     = save_dir / "seq"
    winflat_dir = save_dir / "winflat"

    ensure_dir(seq_dir)
    ensure_dir(winflat_dir)

    df = pd.read_csv(csv_path)
    print(f"[LOAD] rows={len(df):,}, cols={len(df.columns)}")

    df = normalize_ctu_columns(df)

    df["_date"] = df["Timestamp"].dt.date
    df = df.sort_values(["Source IP", "_date", "Timestamp"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    print(f"[INFO] normalized rows={len(df):,}")
    print(f"[INFO] attack rows (Label_binary=1): {int(df['Label_binary'].sum()):,}")
    print(f"[INFO] unique (Source IP, date) groups: {df.groupby(['Source IP', '_date']).ngroups:,}")
    print("\n[LABEL DIST]")
    print(df["Label_raw"].value_counts().head(10))
    print("\n[LABEL_BINARY DIST]")
    print(df["Label_binary"].value_counts())

    X_seq_all, X_flat_all, y_all, sample_ids_all = [], [], [], []
    skipped = 0

    for (src_ip, date), group_df in df.groupby(["Source IP", "_date"], sort=False):
        group_df   = group_df.sort_values("Timestamp").reset_index(drop=True)
        features   = group_df[COMMON_FEATURES].to_numpy(dtype=np.float32)
        label_vals = group_df["Label_binary"].to_numpy(dtype=np.int64)
        row_ids    = group_df["row_id"].astype(str).to_numpy()

        X_seq_list, X_flat_list, y_list, sample_id_list = _build_windows_for_group(
            features, label_vals, WINDOW_SIZE, STEP_SIZE, scenario_name, row_ids
        )

        if not X_seq_list:
            skipped += 1
            continue

        X_seq_all.extend(X_seq_list)
        X_flat_all.extend(X_flat_list)
        y_all.extend(y_list)
        sample_ids_all.extend(sample_id_list)

    print(f"[WINDOW] 스킵된 (Source IP, date) 수: {skipped}")

    if not X_seq_all:
        raise ValueError(
            f"{scenario_name}: 생성된 window가 없습니다. "
            "WINDOW_SIZE가 너무 크거나 group 길이가 너무 짧을 수 있습니다."
        )

    X_seq      = np.asarray(X_seq_all,      dtype=np.float32)
    X_flat     = np.asarray(X_flat_all,     dtype=np.float32)
    y          = np.asarray(y_all,          dtype=np.int64)
    sample_ids = np.asarray(sample_ids_all, dtype=object)

    print(f"[WINDOW] seq shape    : {X_seq.shape}")
    print(f"[WINDOW] winflat shape: {X_flat.shape}")
    print(f"[WINDOW] Botnet 비율  : {y.mean():.4f}  ({y.sum():,}/{len(y):,})")

    # scaler 적용 (옵션)
    if APPLY_SEQ_SCALER:
        if not SEQ_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"SEQ scaler 파일이 없습니다: {SEQ_SCALER_PATH}\n"
                "preprocess_cicids.py를 먼저 실행하세요."
            )
        scaler = joblib.load(SEQ_SCALER_PATH)
        n_samples, seq_len, n_features = X_seq.shape
        X_seq = scaler.transform(X_seq.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        print(f"[SCALE] seq scaler 적용 완료: {SEQ_SCALER_PATH}")

    # 저장
    np.save(seq_dir     / "X.npy", X_seq)
    np.save(seq_dir     / "y.npy", y)
    np.save(winflat_dir / "X.npy", X_flat)
    np.save(winflat_dir / "y.npy", y)

    if SAVE_SAMPLE_IDS:
        np.save(seq_dir     / "sample_ids.npy", sample_ids)
        np.save(winflat_dir / "sample_ids.npy", sample_ids)

    meta = {
        "scenario":              scenario_name,
        "source_csv":            str(csv_path),
        "window_size":           WINDOW_SIZE,
        "step_size":             STEP_SIZE,
        "groupby_key":           "(Source IP, date)",
        "num_features":          len(COMMON_FEATURES),
        "feature_columns":       COMMON_FEATURES,
        "seq_shape":             list(X_seq.shape),
        "winflat_shape":         list(X_flat.shape),
        "num_windows":           int(len(y)),
        "num_attack_windows":    int(y.sum()),
        "botnet_ratio":          float(y.mean()),
        "apply_seq_scaler":      APPLY_SEQ_SCALER,
        "seq_scaler_path":       str(SEQ_SCALER_PATH) if APPLY_SEQ_SCALER else None,
        "treat_to_botnet_as_attack": TREAT_TO_BOTNET_AS_ATTACK,
        "flow_level": {
            "total_flows":       int(len(df)),
            "botnet_flows":      int(df["Label_binary"].sum()),
            "botnet_flow_ratio": float(df["Label_binary"].mean()),
        },
    }

    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"[SAVE] 저장 완료: {save_dir}")
    print(f"[META] Botnet 비율 (window): {meta['botnet_ratio']:.4f}")
    print(f"[META] Botnet 비율 (flow)  : {meta['flow_level']['botnet_flow_ratio']:.4f}")


# =========================================================
# 7. 메인
# =========================================================

def main() -> None:
    print("=== CTU-13 Preprocessing Start ===")
    print(f"[PATH] CTU_RAW_DIR : {CTU_RAW_DIR}")
    print(f"[PATH] SAVE_ROOT   : {SAVE_ROOT}")
    print(f"[CONFIG] WINDOW_SIZE={WINDOW_SIZE}, STEP_SIZE={STEP_SIZE}")
    print(f"[CONFIG] TREAT_TO_BOTNET_AS_ATTACK={TREAT_TO_BOTNET_AS_ATTACK}")
    print(f"[CONFIG] APPLY_SEQ_SCALER={APPLY_SEQ_SCALER}")
    print(f"[CONFIG] COMMON_FEATURES ({len(COMMON_FEATURES)}개): {COMMON_FEATURES}")

    ensure_dir(SAVE_ROOT)

    for filename in SCENARIOS:
        csv_path = CTU_RAW_DIR / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CTU CSV 파일이 없습니다: {csv_path}")
        process_one_scenario(csv_path)

    print("\n=== CTU-13 Preprocessing End ===")
    print("\n[저장된 파일 구조]")
    print("  data/processed/ctu13/")
    for filename in SCENARIOS:
        stem = Path(filename).stem
        print(f"    {stem}/")
        print(f"      seq/     X.npy  y.npy  sample_ids.npy")
        print(f"      winflat/ X.npy  y.npy  sample_ids.npy")
        print(f"      meta.json")


if __name__ == "__main__":
    main()