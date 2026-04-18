import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from typing import Optional


# =========================================================
# 경로 설정
# ---------------------------------------------------------
# 프로젝트 루트 기준으로 원본 데이터 경로와 저장 경로를 정의한다.
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "cic-ids2017")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 내부 네트워크 대역 정의
# ---------------------------------------------------------
# Botnet 탐지는 감염된 내부 host의 반복적 통신 패턴을
# sequence로 구성하는 것이 중요하므로, 내부 IP를 기준으로
# host 단위의 flow sequence를 생성한다.
# =========================================================
INTERNAL_IP_PREFIXES = (
    "192.168.",
    "172.16.",
)


# =========================================================
# 공통 feature 정의
# ---------------------------------------------------------
# RF / XGBoost / CNN-LSTM 모델에 공통으로 사용할 feature 목록이다.
# TrafficLabeling 버전을 사용하는 이유는 Source IP와 Timestamp를
# 포함하고 있어 sequence 기반 모델 구성이 가능하기 때문이다.
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
# 필수 컬럼 정의
# ---------------------------------------------------------
# Source IP와 Timestamp는 sequence 구성에 필요하며,
# 나머지 주요 컬럼은 데이터 정합성 확인을 위해 사용한다.
# =========================================================
REQUIRED_BASE_COLS = [
    "Source IP",
    "Timestamp",
    "Label",
    "Flow Duration",
    "Protocol",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
]


def load_all_csv(raw_dir: str) -> pd.DataFrame:
    """
    원본 CSV 파일 전체를 로드하여 하나의 DataFrame으로 병합한다.

    TrafficLabeling 버전은 날짜별 CSV 파일로 구성되어 있으며,
    파일별 인코딩이 다를 수 있으므로 여러 인코딩을 순차적으로 시도한다.

    Parameters
    ----------
    raw_dir : str
        원본 CSV 파일이 저장된 디렉터리 경로

    Returns
    -------
    df : pd.DataFrame
        병합된 전체 DataFrame
    """
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {raw_dir}")

    print(f"[INFO] CSV 파일 수: {len(csv_files)}")

    df_list = []
    for file_path in csv_files:
        print(f"[LOAD] {os.path.basename(file_path)}")
        for encoding in ["utf-8", "cp1252", "latin-1"]:
            try:
                temp_df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                print(f"       encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] 병합 후 shape: {df.shape}")
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명 앞뒤 공백을 제거한다.

    TrafficLabeling 데이터는 컬럼명에 공백이 포함된 경우가 있어
    후속 컬럼 참조 오류를 방지하기 위해 정규화가 필요하다.

    Parameters
    ----------
    df : pd.DataFrame
        원본 DataFrame

    Returns
    -------
    df : pd.DataFrame
        컬럼명이 정리된 DataFrame
    """
    df.columns = df.columns.str.strip()
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    필수 컬럼 존재 여부를 검증한다.

    Parameters
    ----------
    df : pd.DataFrame
        검증 대상 DataFrame

    Raises
    ------
    ValueError
        필수 컬럼이 하나라도 누락된 경우 발생
    """
    missing = [col for col in REQUIRED_BASE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")


def log_inf_nan_status(df: pd.DataFrame, stage: str) -> None:
    """
    수치형 컬럼 기준 NaN / inf 현황을 출력한다.

    Parameters
    ----------
    df : pd.DataFrame
        확인 대상 DataFrame
    stage : str
        출력 시점 식별 문자열
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    nan_counts = df[numeric_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

    inf_counts = pd.Series(dtype="int64")
    if len(numeric_cols) > 0:
        inf_mask = df[numeric_cols].apply(lambda col: np.isinf(col).sum())
        inf_counts = inf_mask[inf_mask > 0].sort_values(ascending=False)

    print(f"\n[CHECK:{stage}] NaN 컬럼 수: {len(nan_counts)}")
    if len(nan_counts) > 0:
        print("[CHECK] NaN 상위 컬럼:")
        print(nan_counts.head(10))

    print(f"[CHECK:{stage}] inf 컬럼 수: {len(inf_counts)}")
    if len(inf_counts) > 0:
        print("[CHECK] inf 상위 컬럼:")
        print(inf_counts.head(10))


def log_label_distribution(df: pd.DataFrame, stage: str) -> None:
    """
    라벨 분포를 출력한다.

    Parameters
    ----------
    df : pd.DataFrame
        확인 대상 DataFrame
    stage : str
        출력 시점 식별 문자열
    """
    print(f"\n[LABEL DIST:{stage}]")
    print(df["Label"].value_counts(dropna=False))


def basic_cleaning(
    df: pd.DataFrame,
    fill_numeric_na_with_zero: bool = True,
    drop_bad_timestamp: bool = True,
) -> pd.DataFrame:
    """
    CIC-IDS2017 TrafficLabeling 데이터의 기본 정제를 수행한다.

    수행 내용:
    - 문자열 컬럼 공백 제거
    - Timestamp를 datetime으로 변환
    - Timestamp / Label / Source IP 결측 처리
    - inf 값을 NaN으로 변환
    - 수치형 NaN 값을 0으로 대체

    Parameters
    ----------
    df : pd.DataFrame
        원본 병합 DataFrame
    fill_numeric_na_with_zero : bool, default=True
        수치형 NaN 값을 0으로 대체할지 여부
    drop_bad_timestamp : bool, default=True
        Timestamp 파싱 실패 행을 제거할지 여부

    Returns
    -------
    df : pd.DataFrame
        정제 완료된 DataFrame
    """
    before_rows = len(df)

    print("\n[CLEAN] 정제 시작")
    log_label_distribution(df, "before_cleaning")
    log_inf_nan_status(df, "before_cleaning")

    # 문자열 컬럼의 불필요한 공백 제거
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    # Timestamp를 day-first 형식으로 파싱
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )

    parsed_timestamp_na = df["Timestamp"].isna().sum()
    print(f"\n[CLEAN] Timestamp 전체 NaT 행 수: {parsed_timestamp_na:,}")

    # Timestamp 정합성을 유지하기 위해 파싱 실패 행은 제거하거나 보간한다
    if drop_bad_timestamp:
        before_ts_drop = len(df)
        df = df.dropna(subset=["Timestamp"])
        after_ts_drop = len(df)
        print(f"[CLEAN] Timestamp NaT 제거 행 수: {before_ts_drop - after_ts_drop:,}")
    else:
        df["Timestamp"] = df["Timestamp"].ffill()
        remain_nat = df["Timestamp"].isna().sum()
        print(f"[CLEAN] Timestamp ffill 후 남은 NaT 행 수: {remain_nat:,}")

    # 필수 식별 컬럼 결측 제거
    before_required_drop = len(df)
    df = df.dropna(subset=["Label", "Source IP"])
    after_required_drop = len(df)
    print(f"[CLEAN] Label/Source IP 결측 제거 행 수: {before_required_drop - after_required_drop:,}")

    # 빈 문자열 형태의 결측도 제거
    before_empty_drop = len(df)
    df = df[
        (df["Label"].astype(str).str.strip() != "")
        & (df["Source IP"].astype(str).str.strip() != "")
    ]
    after_empty_drop = len(df)
    print(f"[CLEAN] Label/Source IP 빈 문자열 제거 행 수: {before_empty_drop - after_empty_drop:,}")

    # 수치형 inf 값을 NaN으로 변환
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_before = 0
    if len(numeric_cols) > 0:
        inf_before = np.isinf(df[numeric_cols].to_numpy()).sum()

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols_after_replace = df.select_dtypes(include=[np.number]).columns
    nan_after_inf_replace = df[numeric_cols_after_replace].isna().sum().sum()

    print(f"[CLEAN] inf → NaN 변환 개수: {inf_before:,}")
    print(f"[CLEAN] 현재 수치형 전체 NaN 개수: {nan_after_inf_replace:,}")

    log_inf_nan_status(df, "after_inf_to_nan")

    # 모델 입력을 위해 수치형 NaN을 0으로 대체
    if fill_numeric_na_with_zero:
        nan_before_fill = df[numeric_cols_after_replace].isna().sum().sum()
        df[numeric_cols_after_replace] = df[numeric_cols_after_replace].fillna(0)
        nan_after_fill = df[numeric_cols_after_replace].isna().sum().sum()

        print(f"[CLEAN] 수치형 NaN -> 0 대체 개수: {nan_before_fill:,}")
        print(f"[CLEAN] 수치형 NaN 잔여 개수: {nan_after_fill:,}")
    else:
        print("[CLEAN] 수치형 NaN을 0으로 대체하지 않음")

    after_rows = len(df)

    print(f"\n[CLEAN] 제거된 전체 행 수: {before_rows - after_rows:,}")
    print(f"[CLEAN] 정제 후 shape: {df.shape}")

    log_label_distribution(df, "after_cleaning")
    log_inf_nan_status(df, "after_cleaning")

    return df


def create_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Botnet 탐지를 위한 이진 라벨을 생성한다.

    Label에 'bot'이 포함되면 1(Botnet),
    그 외 모든 클래스는 0(Non-Botnet)으로 변환한다.

    Parameters
    ----------
    df : pd.DataFrame
        정제 완료된 DataFrame

    Returns
    -------
    df : pd.DataFrame
        Label_binary 컬럼이 추가된 DataFrame
    """
    df["Label_binary"] = df["Label"].astype(str).str.lower().apply(
        lambda x: 1 if "bot" in x else 0
    )

    print("[LABEL] Label 분포:")
    print(df["Label"].value_counts())
    print("\n[LABEL] Label_binary 분포:")
    print(df["Label_binary"].value_counts())

    return df


def encode_protocol(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """
    Protocol 값을 정수형으로 변환한다.

    Protocol은 원래 숫자 의미를 가지므로,
    가능하면 원래 값을 유지하는 방향으로 처리한다.

    Parameters
    ----------
    df : pd.DataFrame
        Protocol 컬럼을 포함한 DataFrame

    Returns
    -------
    df : pd.DataFrame
        Protocol 컬럼이 정수형으로 변환된 DataFrame
    protocol_encoder : Optional[LabelEncoder]
        현재 구현에서는 사용하지 않으므로 None 반환
    """
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")

    bad_protocol = df["Protocol"].isna().sum()
    print(f"[ENCODE] Protocol 숫자 변환 실패 수: {bad_protocol:,}")

    df["Protocol"] = df["Protocol"].fillna(-1).astype(np.int32)

    print("[ENCODE] Protocol unique values:")
    print(sorted(df["Protocol"].unique().tolist())[:20])

    return df, None


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Timestamp 기준으로 전체 데이터를 시간순 정렬한다.

    Parameters
    ----------
    df : pd.DataFrame
        Timestamp 컬럼을 포함한 DataFrame

    Returns
    -------
    df : pd.DataFrame
        Timestamp 기준으로 정렬된 DataFrame
    """
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print("[ORDER] Timestamp 기준 시간 정렬 완료")
    return df


def is_internal_ip(ip: str) -> bool:
    """
    입력 IP가 내부 네트워크 대역인지 판별한다.

    Parameters
    ----------
    ip : str
        확인할 IP 주소

    Returns
    -------
    bool
        내부 IP 대역이면 True, 아니면 False
    """
    return str(ip).startswith(INTERNAL_IP_PREFIXES)


def analyze_flow_distribution(df: pd.DataFrame) -> None:
    """
    Source IP 기준 flow 수 분포를 분석한다.

    전체 분포와 내부 IP 분포를 각각 확인하여
    sequence 구성에 사용할 window size 결정의 근거로 활용한다.

    Parameters
    ----------
    df : pd.DataFrame
        전처리 완료된 DataFrame
    """
    all_counts = df.groupby("Source IP").size()

    print("\n[FLOW DIST] 전체 src_ip 당 flow 수 분포:")
    print(all_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

    internal_mask = df["Source IP"].apply(is_internal_ip)
    internal_df = df[internal_mask]
    internal_counts = internal_df.groupby("Source IP").size()

    print(f"\n[FLOW DIST] 내부 IP flow 수: {internal_mask.sum():,}개")
    print(f"[FLOW DIST] 내부 IP 종류: {internal_counts.shape[0]}개")

    print("\n[FLOW DIST] 내부 src_ip 당 flow 수 분포:")
    print(internal_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

    botnet_internal = internal_df[internal_df["Label_binary"] == 1]
    total_bot = int(df["Label_binary"].sum())
    internal_bot = len(botnet_internal)

    print(f"\n[FLOW DIST] 내부 IP 중 Botnet flow 수: {internal_bot:,}")
    if total_bot > 0:
        print(f"[FLOW DIST] 전체 Botnet 대비 내부 IP 비율: {internal_bot / total_bot * 100:.1f}%")
    else:
        print("[FLOW DIST] 전체 Botnet 수가 0이라 비율 계산 생략")

    median_val = internal_counts.median()

    if median_val < 5:
        recommended = 3
    elif median_val < 20:
        recommended = 5
    elif median_val < 50:
        recommended = 10
    else:
        recommended = 15

    candidates = sorted(set([max(3, recommended - 5), recommended, recommended + 5]))

    print(f"\n[FLOW DIST] → 추천 window_size: {recommended}")
    print(f"[FLOW DIST] → 실험 후보: {', '.join(map(str, candidates))}")


def create_flat_data(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    RF / XGBoost용 flat 데이터를 생성한다.

    단일 flow 기반 분류를 위해 전체 데이터를 그대로 사용하며,
    sequence 구성 없이 feature 행렬과 라벨 벡터를 반환한다.

    Parameters
    ----------
    df : pd.DataFrame
        전처리 완료된 DataFrame
    feature_cols : list[str]
        모델 입력에 사용할 feature 컬럼 목록

    Returns
    -------
    X : np.ndarray
        shape = (n_samples, n_features)
        단일 flow 기반 입력 feature 배열
    y : np.ndarray
        shape = (n_samples,)
        각 flow의 이진 라벨 배열
    """
    valid_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"[WARN] flat 데이터에서 없는 컬럼 (스킵): {missing_cols}")

    X = df[valid_cols].values.astype(np.float32)
    y = df["Label_binary"].values.astype(np.int32)

    print(f"\n[FLAT] X shape: {X.shape}")
    print(f"[FLAT] y shape: {y.shape}")
    print(f"[FLAT] Botnet 비율: {y.mean():.4f}")
    print(f"[FLAT] Botnet 수: {y.sum():,} / 전체: {len(y):,}")

    return X, y


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 15,
    step_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CNN-LSTM 입력용 sequence 데이터를 생성한다.

    내부 IP 기준으로 flow를 그룹화한 뒤 Timestamp 순으로 정렬하고,
    sliding window를 적용하여 host 단위의 연속적인 행동 흐름을 sequence로 구성한다.

    라벨링 기준:
    - window 내 Botnet flow가 하나라도 존재하면 sequence label = 1
    - 그렇지 않으면 sequence label = 0

    Parameters
    ----------
    df : pd.DataFrame
        전처리 완료된 전체 DataFrame
    feature_cols : list[str]
        sequence 입력에 사용할 feature 컬럼 목록
    window_size : int, default=15
        하나의 sequence를 구성하는 flow 개수
    step_size : int, default=5
        sliding window 이동 간격

    Returns
    -------
    X : np.ndarray
        shape = (n_sequences, window_size, n_features)
        CNN-LSTM 입력용 sequence feature 배열
    y : np.ndarray
        shape = (n_sequences,)
        각 sequence에 대한 이진 라벨 배열
        (0 = Normal, 1 = Botnet)
    """
    valid_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"[WARN] sequence 생성에서 없는 컬럼 (스킵): {missing_cols}")

    # 내부 IP만 선택하여 host 단위 sequence 구성
    internal_mask = df["Source IP"].apply(is_internal_ip)
    df_internal = df[internal_mask].copy()

    print(f"\n[SEQ] 내부 IP flow 수: {len(df_internal):,}")
    print(f"[SEQ] 내부 IP 종류: {df_internal['Source IP'].nunique()}개")

    sequences = []
    labels = []
    skipped = 0

    grouped = df_internal.groupby("Source IP")

    for src_ip, group in grouped:
        # 각 host의 flow를 시간 순으로 정렬
        group = group.sort_values("Timestamp").reset_index(drop=True)

        features = group[valid_cols].values
        label_vals = group["Label_binary"].values
        n_flows = len(features)

        # sequence 길이보다 짧은 host는 제외
        if n_flows < window_size:
            skipped += 1
            continue

        # sliding window 기반 sequence 생성
        for start in range(0, n_flows - window_size + 1, step_size):
            end = start + window_size
            seq = features[start:end]
            seq_labels = label_vals[start:end]

            # window 내 Botnet flow가 하나라도 존재하면 Botnet sequence로 라벨링
            seq_label = 1 if seq_labels.sum() > 0 else 0

            sequences.append(seq)
            labels.append(seq_label)

    print(f"\n[SEQ] window_size={window_size}, step_size={step_size}")
    print(f"[SEQ] 생성된 sequence 수: {len(sequences):,}")
    print(f"[SEQ] flow 부족으로 스킵된 src_ip 수: {skipped}")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print(f"[SEQ] X shape: {X.shape}")
    print(f"[SEQ] y shape: {y.shape}")
    print(f"[SEQ] Botnet sequence 비율: {y.mean():.4f}")
    print(f"[SEQ] Botnet sequence 수: {y.sum():,} / 전체: {len(y):,}")

    return X, y


def save_outputs(
    df: pd.DataFrame,
    protocol_encoder: Optional[LabelEncoder],
    save_dir: str,
) -> None:
    """
    전처리 결과 DataFrame과 부가 정보를 저장한다.

    저장 항목:
    - cicids2017_traffic.parquet
    - protocol_label_encoder.pkl (필요한 경우)

    Parameters
    ----------
    df : pd.DataFrame
        저장할 전처리 결과 DataFrame
    protocol_encoder : Optional[LabelEncoder]
        Protocol 인코더 객체
    save_dir : str
        저장 디렉터리 경로
    """
    save_cols = (
        ["Source IP", "Timestamp", "Label", "Label_binary"]
        + [col for col in ML_FEATURES if col in df.columns]
    )

    df[save_cols].to_parquet(
        os.path.join(save_dir, "cicids2017_traffic.parquet"),
        index=False
    )

    if protocol_encoder is not None:
        joblib.dump(
            protocol_encoder,
            os.path.join(save_dir, "protocol_label_encoder.pkl")
        )
        print("[SAVE] Protocol encoder 저장 완료")
    else:
        print("[SAVE] Protocol encoder 저장 생략 (원값 유지 방식 사용)")

    print("[SAVE] 전처리 데이터 저장 완료")


def save_numpy(
    data: np.ndarray,
    label: np.ndarray,
    save_dir: str,
    name: str,
) -> None:
    """
    numpy 배열 형태의 입력 데이터와 라벨을 저장한다.

    Parameters
    ----------
    data : np.ndarray
        저장할 입력 배열
    label : np.ndarray
        저장할 라벨 배열
    save_dir : str
        저장 디렉터리 경로
    name : str
        파일명 식별자
    """
    np.save(os.path.join(save_dir, f"X_{name}.npy"), data)
    np.save(os.path.join(save_dir, f"y_{name}.npy"), label)
    print(f"[SAVE] X_{name}.npy / y_{name}.npy 저장 완료")


def preview_data(df: pd.DataFrame) -> None:
    """
    전처리 결과의 일부 샘플과 전체 shape를 출력한다.

    Parameters
    ----------
    df : pd.DataFrame
        미리보기 대상 DataFrame
    """
    print("\n[PREVIEW] 상위 5행")
    print(df[["Source IP", "Timestamp", "Label", "Label_binary",
              "Flow Duration", "Protocol"]].head())
    print(f"\n[PREVIEW] 전체 shape: {df.shape}")


def main():
    """
    전처리 파이프라인 전체를 실행한다.

    수행 순서:
    1. CSV 로드 및 컬럼 정리
    2. 데이터 정제
    3. 이진 라벨 생성
    4. Protocol 처리 및 시간 정렬
    5. 전처리 결과 저장
    6. flow 분포 분석
    7. flat 데이터 및 sequence 데이터 생성/저장
    """
    print("=== CIC-IDS2017 TrafficLabeling Preprocessing Start ===")
    print(f"[PATH] RAW_DIR : {RAW_DIR}")
    print(f"[PATH] SAVE_DIR: {SAVE_DIR}")

    WINDOW_SIZE = 15
    STEP_SIZE = 5

    print(f"\n[CONFIG] 최종 WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"[CONFIG] 최종 STEP_SIZE  : {STEP_SIZE}")

    df = load_all_csv(RAW_DIR)
    df = normalize_column_names(df)
    validate_columns(df)

    df = basic_cleaning(
        df,
        fill_numeric_na_with_zero=True,
        drop_bad_timestamp=True,
    )

    df = create_binary_label(df)
    df, protocol_encoder = encode_protocol(df)
    df = sort_by_time(df)

    preview_data(df)
    save_outputs(df, protocol_encoder, SAVE_DIR)

    analyze_flow_distribution(df)

    print("\n[STEP] RF / XGBoost 용 flat 데이터 생성")
    X_flat, y_flat = create_flat_data(df, ML_FEATURES)
    save_numpy(X_flat, y_flat, SAVE_DIR, "flat")

    print(f"\n[STEP] CNN-LSTM 용 sequence 데이터 생성 (window_size={WINDOW_SIZE})")
    X_seq, y_seq = create_sequences(
        df=df,
        feature_cols=ML_FEATURES,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
    )
    save_numpy(X_seq, y_seq, SAVE_DIR, f"seq_w{WINDOW_SIZE}")

    print("\n=== Preprocessing Done ===")
    print("\n[저장된 파일]")
    print("  cicids2017_traffic.parquet           ← 전처리된 전체 데이터")
    print("  X_flat.npy / y_flat.npy              ← RF / XGBoost 입력")
    print(f"  X_seq_w{WINDOW_SIZE}.npy / y_seq_w{WINDOW_SIZE}.npy    ← CNN-LSTM 입력")

    print("\n[NEXT STEP]")
    print("  1. baseline_models.py → RF / XGBoost 원본 학습 (증강 없음)")
    print("  2. wgan_gp.py         → WGAN-GP로 Botnet 데이터 증강")
    print("  3. cnn_lstm.py        → CNN-LSTM 학습")
    print("  4. evaluate.py        → 성능 비교표 생성")


if __name__ == "__main__":
    main()