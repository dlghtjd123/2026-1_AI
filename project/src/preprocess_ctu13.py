# preprocess_ctu13.py
# 목적:
# - CTU-13 scenario1.csv, scenario9.csv를 외부검증용 입력으로 변환
# - CIC와 공통으로 맞출 수 있는 feature만 생성
# - winflat / seq / sample_ids 저장
#
# 중요:
# - 이 코드는 "공통 feature 기반 외부검증"용이다.
# - 현재 저장된 77-feature CIC 모델과는 바로 호환되지 않을 수 있다.
# - 공정한 비교를 위해서는 CIC도 동일한 공통 feature 기준으로 다시 학습하는 것이 맞다.

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 1. 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CTU_RAW_DIR = BASE_DIR / "data" / "raw" / "ctu-13"
SAVE_ROOT = BASE_DIR / "data" / "processed" / "ctu13"

SCENARIOS = [
    "scenario1.csv",
    "scenario9.csv",
]

WINDOW_SIZE = 15
STEP_SIZE = 5

SEQ_SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler_seq_w15.pkl"
APPLY_SEQ_SCALER = False

SAVE_SAMPLE_IDS = True
TREAT_TO_BOTNET_AS_ATTACK = False


# =========================================================
# 2. 공통 feature 정의
# =========================================================
# 아래 feature들은 CTU에서 비교적 안정적으로 만들 수 있는 공통 feature들이다.
# 현재는 9개 기준이다.
# 따라서 seq shape은 (N, 15, 9), winflat shape은 (N, 15*9) 가 된다.
#
# 중요:
# 지금 네 기존 모델이 (N, 15, 77), (N, 1155) 기준으로 학습되었다면
# 이 CTU 데이터와 입력 차원이 다르므로 그대로는 평가할 수 없다.

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


# =========================================================
# 3. 유틸 함수
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """
    후보 컬럼명 중 실제 존재하는 첫 컬럼을 반환
    """
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
    """
    CIC 스타일 숫자 프로토콜로 변환
    TCP=6, UDP=17, ICMP=1, 그 외 0
    """
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
    """
    CTU StartTime / Timestamp 파싱
    """
    # dayfirst=True를 함께 시도하는 것이 CTU 쪽에서 안전한 경우가 많음
    ts = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # 혹시 대부분 실패하면 일반 방식 한 번 더
    if ts.notna().sum() == 0:
        ts = pd.to_datetime(series, errors="coerce")
    return ts


def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default)


def label_to_binary(label_value: str) -> int:
    """
    CTU-13 레이블 이진화
    - From-Botnet / Botnet / malicious 계열 => 1
    - To-Botnet은 옵션에 따라 처리
    - 그 외 normal/background => 0
    """
    if pd.isna(label_value):
        return 0

    s = str(label_value).strip().lower()

    if "to-botnet" in s:
        return 1 if TREAT_TO_BOTNET_AS_ATTACK else 0

    attack_keywords = [
        "from-botnet",
        "botnet",
        "malicious",
    ]
    if any(k in s for k in attack_keywords):
        return 1

    return 0


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0, np.nan)
    out = numerator / den
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_group_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["Source IP"].astype(str) + "|" +
        df["Destination IP"].astype(str) + "|" +
        df["Protocol"].astype(str)
    )


def make_windows_for_group(
    group_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    step_size: int,
    scenario_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[str]]:
    """
    한 group에 대해 seq / winflat / y / sample_ids 생성
    """
    X_seq_list = []
    X_flat_list = []
    y_list = []
    sample_id_list = []

    values = group_df[feature_cols].to_numpy(dtype=np.float32)
    labels = group_df["Label_binary"].to_numpy(dtype=np.int64)
    row_ids = group_df["row_id"].astype(str).to_numpy()

    n = len(group_df)
    if n < window_size:
        return X_seq_list, X_flat_list, y_list, sample_id_list

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size

        x_seq = values[start:end]                       # (window, feature)
        x_flat = x_seq.reshape(-1)                     # (window*feature,)
        y = int(labels[start:end].max())               # window 내 하나라도 공격이면 1
        sample_id = f"{scenario_name}|{row_ids[start]}|{row_ids[end - 1]}"

        X_seq_list.append(x_seq)
        X_flat_list.append(x_flat)
        y_list.append(y)
        sample_id_list.append(sample_id)

    return X_seq_list, X_flat_list, y_list, sample_id_list


# =========================================================
# 4. CTU 컬럼 정규화
# =========================================================

def normalize_ctu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    CTU-13 raw CSV를 공통 처리용 표준 컬럼으로 정리
    """
    df = df.copy()

    timestamp_col = find_column(df, ["StartTime", "Timestamp", "starttime", "stime"])
    duration_col = find_column(df, ["Dur", "Duration", "dur"])
    proto_col = find_column(df, ["Proto", "Protocol", "proto"])

    src_ip_col = find_column(df, ["SrcAddr", "Source IP", "Src IP", "srcaddr", "src_ip"])
    dst_ip_col = find_column(df, ["DstAddr", "Destination IP", "Dst IP", "dstaddr", "dst_ip"])

    label_col = find_column(df, ["Label", "label"])

    # bytes / packets 관련 후보
    total_pkts_col = find_column(df, ["TotPkts", "Total Packets", "totpkts", "pkts"], required=False)
    total_bytes_col = find_column(df, ["TotBytes", "Total Bytes", "totbytes", "bytes"], required=False)
    src_bytes_col = find_column(df, ["SrcBytes", "srcbytes", "sbytes", "Src Bytes"], required=False)

    src_pkts_col = find_column(df, ["SrcPkts", "srcpkts", "spkts", "Src Pkts"], required=False)
    dst_pkts_col = find_column(df, ["DstPkts", "dstpkts", "dpkts", "Dst Pkts"], required=False)

    df["Timestamp"] = parse_timestamp(df[timestamp_col])
    df["Flow Duration"] = to_numeric(df[duration_col])
    df["Protocol"] = df[proto_col].apply(parse_protocol)

    df["Source IP"] = df[src_ip_col].astype(str)
    df["Destination IP"] = df[dst_ip_col].astype(str)

    df["Label_raw"] = df[label_col].astype(str)
    df["Label_binary"] = df["Label_raw"].apply(label_to_binary)

    # 총 패킷 / 바이트
    if total_pkts_col is not None:
        df["TotPkts"] = to_numeric(df[total_pkts_col])
    else:
        raise KeyError("CTU CSV에서 총 패킷 수 컬럼(TotPkts)을 찾지 못했습니다.")

    if total_bytes_col is not None:
        df["TotBytes"] = to_numeric(df[total_bytes_col])
    else:
        raise KeyError("CTU CSV에서 총 바이트 수 컬럼(TotBytes)을 찾지 못했습니다.")

    # 방향별 패킷 수
    # 있으면 사용, 없으면 불완전하지만 fallback 처리
    if src_pkts_col is not None and dst_pkts_col is not None:
        df["Total Fwd Packets"] = to_numeric(df[src_pkts_col])
        df["Total Bwd Packets"] = to_numeric(df[dst_pkts_col])
    else:
        # CTU CSV에 방향별 packet count가 없을 때의 보수적 fallback
        # 정확한 방향별 packet 수가 아니므로 한계가 있다.
        df["Total Fwd Packets"] = df["TotPkts"]
        df["Total Bwd Packets"] = 0.0
        print("[WARN] 방향별 패킷 수 컬럼이 없어 Total Fwd Packets=TotPkts, Total Bwd Packets=0 으로 대체합니다.")

    # rate / packet size
    df["Flow Bytes/s"] = safe_divide(df["TotBytes"], df["Flow Duration"])
    df["Flow Packets/s"] = safe_divide(df["TotPkts"], df["Flow Duration"])
    df["Average Packet Size"] = safe_divide(df["TotBytes"], df["TotPkts"])

    # IAT 근사치
    # 실제 packet timestamp가 없으므로 Dur와 TotPkts를 이용한 근사
    # packet 수가 1 이하이면 IAT Mean은 0 처리
    denom = (df["TotPkts"] - 1).clip(lower=1)
    df["Flow IAT Mean"] = safe_divide(df["Flow Duration"], denom)
    df["Flow IAT Max"] = df["Flow Duration"].copy()

    # 정리
    df = df.replace([np.inf, -np.inf], np.nan)
    df[COMMON_FEATURES] = df[COMMON_FEATURES].fillna(0.0)

    # timestamp 없는 row는 제거
    before = len(df)
    df = df.dropna(subset=["Timestamp"]).copy()
    after = len(df)
    if before != after:
        print(f"[INFO] Timestamp 파싱 실패 row 제거: {before - after}개")

    return df


# =========================================================
# 5. 시나리오 처리
# =========================================================

def process_one_scenario(csv_path: Path) -> None:
    scenario_name = csv_path.stem
    print(f"\n[PROCESS] {scenario_name}")

    save_dir = SAVE_ROOT / scenario_name
    seq_dir = save_dir / "seq"
    winflat_dir = save_dir / "winflat"

    ensure_dir(seq_dir)
    ensure_dir(winflat_dir)

    # CSV 로드
    df = pd.read_csv(csv_path)
    print(f"[LOAD] rows={len(df):,}, cols={len(df.columns)}")

    # 컬럼 정규화
    df = normalize_ctu_columns(df)

    # 정렬 및 group id 생성
    df = df.sort_values(["Source IP", "Destination IP", "Protocol", "Timestamp"]).reset_index(drop=True)
    df["group_id"] = build_group_id(df)
    df["row_id"] = np.arange(len(df))

    print(f"[INFO] normalized rows={len(df):,}")
    print(f"[INFO] attack rows={int(df['Label_binary'].sum()):,}")
    print(f"[INFO] unique groups={df['group_id'].nunique():,}")

    # window 생성
    X_seq_all = []
    X_flat_all = []
    y_all = []
    sample_ids_all = []

    for group_key, group_df in df.groupby("group_id", sort=False):
        group_df = group_df.sort_values("Timestamp").reset_index(drop=True)

        X_seq_list, X_flat_list, y_list, sample_id_list = make_windows_for_group(
            group_df=group_df,
            feature_cols=COMMON_FEATURES,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            scenario_name=scenario_name,
        )

        X_seq_all.extend(X_seq_list)
        X_flat_all.extend(X_flat_list)
        y_all.extend(y_list)
        sample_ids_all.extend(sample_id_list)

    if len(X_seq_all) == 0:
        raise ValueError(
            f"{scenario_name}: 생성된 window가 없습니다. "
            "WINDOW_SIZE가 너무 크거나 group 길이가 너무 짧을 수 있습니다."
        )

    X_seq = np.asarray(X_seq_all, dtype=np.float32)          # (N, W, F)
    X_flat = np.asarray(X_flat_all, dtype=np.float32)        # (N, W*F)
    y = np.asarray(y_all, dtype=np.int64)
    sample_ids = np.asarray(sample_ids_all, dtype=object)

    print(f"[WINDOW] seq shape   : {X_seq.shape}")
    print(f"[WINDOW] winflat shape: {X_flat.shape}")
    print(f"[WINDOW] y shape     : {y.shape}")
    print(f"[WINDOW] attack win  : {int(y.sum()):,}")

    # seq scaler 적용
    if APPLY_SEQ_SCALER:
        if not SEQ_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"SEQ scaler 파일이 없습니다: {SEQ_SCALER_PATH}\n"
                "현재 설정은 APPLY_SEQ_SCALER=True 입니다."
            )
        scaler = joblib.load(SEQ_SCALER_PATH)

        n_samples, seq_len, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_2d = scaler.transform(X_seq_2d)
        X_seq = X_seq_2d.reshape(n_samples, seq_len, n_features)

        print(f"[SCALE] seq scaler 적용 완료: {SEQ_SCALER_PATH}")

    # 저장
    np.save(seq_dir / "X.npy", X_seq)
    np.save(seq_dir / "y.npy", y)

    np.save(winflat_dir / "X.npy", X_flat)
    np.save(winflat_dir / "y.npy", y)

    if SAVE_SAMPLE_IDS:
        np.save(seq_dir / "sample_ids.npy", sample_ids)
        np.save(winflat_dir / "sample_ids.npy", sample_ids)

    meta = {
        "scenario": scenario_name,
        "source_csv": str(csv_path),
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "num_features": len(COMMON_FEATURES),
        "feature_columns": COMMON_FEATURES,
        "seq_shape": list(X_seq.shape),
        "winflat_shape": list(X_flat.shape),
        "num_windows": int(len(y)),
        "num_attack_windows": int(y.sum()),
        "apply_seq_scaler": APPLY_SEQ_SCALER,
        "seq_scaler_path": str(SEQ_SCALER_PATH) if APPLY_SEQ_SCALER else None,
        "treat_to_botnet_as_attack": TREAT_TO_BOTNET_AS_ATTACK,
    }

    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"[SAVE] {save_dir}")


# =========================================================
# 6. 메인
# =========================================================

def main() -> None:
    print("=== CTU-13 Preprocessing Start ===")
    print(f"[RAW_DIR ] {CTU_RAW_DIR}")
    print(f"[SAVE_DIR] {SAVE_ROOT}")
    print(f"[CONFIG ] WINDOW_SIZE={WINDOW_SIZE}, STEP_SIZE={STEP_SIZE}")
    print(f"[CONFIG ] APPLY_SEQ_SCALER={APPLY_SEQ_SCALER}")

    ensure_dir(SAVE_ROOT)

    for filename in SCENARIOS:
        csv_path = CTU_RAW_DIR / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CTU CSV 파일이 없습니다: {csv_path}")

        process_one_scenario(csv_path)

    print("\n=== CTU-13 Preprocessing End ===")


if __name__ == "__main__":
    main()