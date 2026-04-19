import os
import glob
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from typing import Optional


# =========================================================
# 경로 설정
# ---------------------------------------------------------
# __file__ 기준으로 두 단계 상위 디렉터리를 프로젝트 루트로 삼는다.
# raw 데이터는 data/raw/cic-ids2017/, 전처리 결과는 data/processed/에 저장한다.
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "cic-ids2017")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 내부 네트워크 대역 정의
# ---------------------------------------------------------
# CIC-IDS2017 실험 환경에서 감염 대상이 되는 내부 호스트 대역이다.
# sequence 생성 시 외부 IP(공격자, 외부 서버 등)는 제외하고
# 내부 호스트 단위로만 행동 흐름을 구성한다.
# =========================================================
INTERNAL_IP_PREFIXES = (
    "192.168.",
    "172.16.",
)


# =========================================================
# 모델 입력 feature 목록
# ---------------------------------------------------------
# RF / XGBoost / CNN-LSTM 세 모델이 공통으로 사용하는 feature 집합이다.
# CICFlowMeter가 출력하는 flow 수준 통계값으로 구성되어 있으며,
# TrafficLabeling 버전을 기준으로 한다.
# (MachineLearningCVE 버전은 Source IP / Timestamp가 없어 sequence 구성 불가)
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
# 필수 컬럼 목록
# ---------------------------------------------------------
# 파이프라인 실행 전 존재 여부를 검증할 컬럼이다.
# Source IP / Timestamp는 sequence 구성에 반드시 필요하고,
# 나머지는 데이터 정합성 확인 및 정제 단계에서 사용한다.
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


# =========================================================
# 라벨 매핑 정의
# ---------------------------------------------------------
# Botnet으로 분류할 라벨만 명시적으로 정의하고,
# 나머지는 모두 0(Normal)으로 처리한다.
# 0으로 처리된 라벨은 _label_counter에 기록되어
# 파이프라인 종료 후 분포를 확인할 수 있다.
# =========================================================
BOTNET_LABELS = {"bot", "botnet"}

# 실행 중 등장한 모든 라벨의 빈도를 수집한다 (1=Botnet 포함)
_label_counter: Counter = Counter()


# =========================================================
# CSV 로드
# =========================================================

def load_all_csv(raw_dir: str) -> pd.DataFrame:
    """
    디렉터리 내 모든 CSV 파일을 읽어 하나의 DataFrame으로 병합한다.

    CIC-IDS2017 TrafficLabeling 버전은 날짜별로 CSV가 분리되어 있고
    파일마다 인코딩이 다를 수 있으므로, utf-8 → cp1252 → latin-1 순으로
    인코딩을 순차 시도한다. 세 가지 모두 실패하거나 다른 오류가 발생하면
    해당 파일에서 ValueError를 발생시키고 파이프라인을 중단한다.

    Parameters
    ----------
    raw_dir : str
        CSV 파일이 위치한 디렉터리 경로

    Returns
    -------
    pd.DataFrame
        전체 CSV를 행 방향으로 이어 붙인 DataFrame.
        인덱스는 0부터 재설정된다.

    Raises
    ------
    FileNotFoundError
        raw_dir에 CSV 파일이 하나도 없을 때
    ValueError
        특정 파일의 모든 인코딩 시도가 실패했을 때
    """
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {raw_dir}")

    print(f"[INFO] CSV 파일 수: {len(csv_files)}")

    df_list = []
    for file_path in csv_files:
        print(f"[LOAD] {os.path.basename(file_path)}")
        temp_df = None

        for encoding in ["utf-8", "cp1252", "latin-1"]:
            try:
                temp_df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                print(f"       encoding: {encoding}")
                break
            except UnicodeDecodeError:
                # 다음 인코딩 후보로 넘어간다
                continue
            except Exception as e:
                # UnicodeDecodeError 외의 오류(손상된 파일, 구분자 문제 등)는
                # 재시도해도 의미가 없으므로 즉시 루프를 탈출한다
                print(f"[WARN] {os.path.basename(file_path)} 로드 실패 ({encoding}): {e}")
                break

        # 모든 인코딩 시도가 실패한 경우 파이프라인을 중단한다
        if temp_df is None:
            raise ValueError(f"모든 인코딩 시도 실패: {file_path}")

        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] 병합 후 shape: {df.shape}")
    return df


# =========================================================
# 컬럼명 정규화 / 검증
# =========================================================

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 컬럼명의 앞뒤 공백을 제거한다.

    CICFlowMeter 출력 파일은 컬럼명에 선행·후행 공백이 포함된 경우가 있어
    이후 컬럼 참조 시 KeyError가 발생할 수 있다.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        컬럼명이 strip된 DataFrame (원본 수정)
    """
    df.columns = df.columns.str.strip()
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    REQUIRED_BASE_COLS에 정의된 필수 컬럼이 모두 존재하는지 검증한다.

    파이프라인 초반에 호출하여 후속 단계에서 KeyError가 발생하는 상황을 방지한다.
    하나라도 누락된 컬럼이 있으면 즉시 ValueError를 발생시킨다.

    Parameters
    ----------
    df : pd.DataFrame

    Raises
    ------
    ValueError
        누락된 필수 컬럼이 하나 이상 있을 때
    """
    missing = [col for col in REQUIRED_BASE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")


# =========================================================
# 상태 로깅 유틸리티
# =========================================================

def log_inf_nan_status(df: pd.DataFrame, stage: str) -> None:
    """
    수치형 컬럼의 NaN / inf 개수를 출력한다.

    전처리 전·후 등 여러 시점에서 호출하여 데이터 품질 변화를 추적하는 용도이다.
    NaN과 inf가 없는 컬럼은 출력하지 않는다.

    Parameters
    ----------
    df : pd.DataFrame
    stage : str
        출력 시점을 구분하는 식별자 (예: "before_cleaning", "after_inf_to_nan")
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
    Label 컬럼의 클래스 분포를 출력한다.

    정제 전·후 라벨 변화(행 제거로 인한 클래스 소실 등)를 확인하는 용도이다.

    Parameters
    ----------
    df : pd.DataFrame
    stage : str
        출력 시점 식별자 (예: "before_cleaning", "after_cleaning")
    """
    print(f"\n[LABEL DIST:{stage}]")
    print(df["Label"].value_counts(dropna=False))


# =========================================================
# 기본 정제
# =========================================================

def basic_cleaning(
    df: pd.DataFrame,
    fill_numeric_na_with_zero: bool = True,
    drop_bad_timestamp: bool = True,
) -> pd.DataFrame:
    """
    원시 데이터에 대한 기본 정제를 수행한다.

    수행 순서:
    1. 문자열 컬럼 공백 제거
    2. Timestamp → datetime 변환 (day-first, mixed format)
    3. Timestamp 파싱 실패 행 제거 또는 forward-fill
    4. Label / Source IP 결측 및 빈 문자열 행 제거
    5. 수치형 inf → NaN 변환
    6. 수치형 NaN → 0 대체 (옵션)

    Parameters
    ----------
    df : pd.DataFrame
        병합된 원시 DataFrame
    fill_numeric_na_with_zero : bool, default=True
        True이면 수치형 NaN을 0으로 대체한다.
        False이면 NaN을 그대로 유지한다 (모델이 직접 처리해야 함).
    drop_bad_timestamp : bool, default=True
        True이면 Timestamp 파싱 실패 행을 제거한다.
        False이면 앞 행의 값으로 forward-fill한다.

    Returns
    -------
    pd.DataFrame
        정제 완료된 DataFrame
    """
    before_rows = len(df)

    print("\n[CLEAN] 정제 시작")
    log_label_distribution(df, "before_cleaning")
    log_inf_nan_status(df, "before_cleaning")

    # 1. 문자열 컬럼 공백 제거
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    # 2. Timestamp 파싱
    # CIC-IDS2017은 날짜가 day-first(DD/MM/YYYY) 형식이며
    # 파일마다 포맷이 미묘하게 달라 format="mixed"로 처리한다
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )

    parsed_timestamp_na = df["Timestamp"].isna().sum()
    print(f"\n[CLEAN] Timestamp 전체 NaT 행 수: {parsed_timestamp_na:,}")

    # 3. Timestamp 파싱 실패 처리
    if drop_bad_timestamp:
        before_ts_drop = len(df)
        df = df.dropna(subset=["Timestamp"])
        after_ts_drop = len(df)
        print(f"[CLEAN] Timestamp NaT 제거 행 수: {before_ts_drop - after_ts_drop:,}")
    else:
        # 제거 대신 직전 유효 Timestamp로 채운다
        df["Timestamp"] = df["Timestamp"].ffill()
        remain_nat = df["Timestamp"].isna().sum()
        print(f"[CLEAN] Timestamp ffill 후 남은 NaT 행 수: {remain_nat:,}")

    # 4. Label / Source IP 결측 및 빈 문자열 제거
    # Label이 없으면 라벨링 자체가 불가능하고,
    # Source IP가 없으면 host 단위 sequence를 구성할 수 없다
    before_required_drop = len(df)
    df = df.dropna(subset=["Label", "Source IP"])
    after_required_drop = len(df)
    print(f"[CLEAN] Label/Source IP 결측 제거 행 수: {before_required_drop - after_required_drop:,}")

    before_empty_drop = len(df)
    df = df[
        (df["Label"].astype(str).str.strip() != "")
        & (df["Source IP"].astype(str).str.strip() != "")
    ]
    after_empty_drop = len(df)
    print(f"[CLEAN] Label/Source IP 빈 문자열 제거 행 수: {before_empty_drop - after_empty_drop:,}")

    # 5. inf → NaN 변환
    # CICFlowMeter는 Flow Bytes/s 등에서 division-by-zero로 inf를 출력할 수 있다
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

    # 6. 수치형 NaN → 0 대체
    # RF / XGBoost / CNN-LSTM 모두 NaN 입력을 허용하지 않으므로
    # 기본값 0으로 채운다. 도메인 지식상 해당 기능이 측정되지 않은 것을
    # 0(활동 없음)으로 해석하는 것이 자연스럽다.
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


# =========================================================
# 이진 라벨 생성
# =========================================================

def _map_binary_label(label: str, strict: bool = False) -> int:
    """
    단일 라벨 문자열을 이진 정수(0 또는 1)로 변환하는 내부 매핑 함수.

    BOTNET_LABELS에 속하면 1, 그 외는 모두 0으로 처리한다.
    0으로 처리된 라벨은 _label_counter에 기록되어
    create_binary_label 종료 시 분포를 확인할 수 있다.

    strict=True이면 BOTNET_LABELS에 없는 라벨에서 즉시 ValueError를 발생시킨다.
    데이터셋을 처음 파악할 때 사용하고, 실험 단계에서는 False로 둔다.

    Parameters
    ----------
    label : str
        원본 Label 컬럼의 단일 값
    strict : bool, default=False
        True이면 미등록 라벨에서 예외 발생, False이면 0으로 처리 후 기록

    Returns
    -------
    int
        1 (Botnet) 또는 0 (Normal)

    Raises
    ------
    ValueError
        strict=True이고 BOTNET_LABELS에 없는 라벨일 때
    """
    label = str(label).strip().lower()

    if label in BOTNET_LABELS:
        return 1

    # Botnet이 아닌 모든 라벨은 0으로 처리하되 카운터에 기록한다
    _label_counter[label] += 1
    if strict:
        raise ValueError(f"알 수 없는 라벨 값: '{label}'")
    return 0


def create_binary_label(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Label 컬럼을 기반으로 이진 라벨 컬럼(Label_binary)을 생성한다.

    BOTNET_LABELS에 해당하면 1, 나머지는 모두 0으로 처리한다.
    0으로 처리된 라벨 목록은 _label_counter를 통해 파이프라인 종료 후 확인 가능하다.

    Parameters
    ----------
    df : pd.DataFrame
        basic_cleaning이 완료된 DataFrame
    strict : bool, default=False
        True이면 미등록 라벨 등장 시 즉시 예외 발생 (데이터 구조 파악용)
        False이면 0으로 처리 후 카운터에 기록 (실험/학습용)

    Returns
    -------
    pd.DataFrame
        Label_binary 컬럼이 추가된 DataFrame
    """
    df["Label_binary"] = df["Label"].map(
        lambda x: _map_binary_label(x, strict=strict)
    )

    print("[LABEL] 0으로 처리된 라벨 분포 (Non-Botnet):")
    for label, count in _label_counter.most_common():
        print(f"  '{label}': {count:,}건")

    print("\n[LABEL] Label_binary 분포:")
    print(df["Label_binary"].value_counts())

    return df


# =========================================================
# Protocol 인코딩
# =========================================================

def encode_protocol(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """
    Protocol 컬럼을 정수형(int32)으로 변환한다.

    Protocol은 IANA 프로토콜 번호(TCP=6, UDP=17 등)를 의미하는 수치형 값이므로
    LabelEncoder를 사용하지 않고 원래 숫자 값을 그대로 유지한다.
    숫자로 변환할 수 없는 값(문자열 등)은 -1로 채운다.

    현재 LabelEncoder를 사용하지 않으므로 두 번째 반환값은 항상 None이다.
    다른 데이터셋과의 인터페이스 일관성을 위해 반환 시그니처는 유지한다.

    Parameters
    ----------
    df : pd.DataFrame
        Protocol 컬럼을 포함한 DataFrame

    Returns
    -------
    df : pd.DataFrame
        Protocol 컬럼이 int32로 변환된 DataFrame
    None
        LabelEncoder 미사용 (인터페이스 일관성용 자리 표시자)
    """
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")

    bad_protocol = df["Protocol"].isna().sum()
    print(f"[ENCODE] Protocol 숫자 변환 실패 수: {bad_protocol:,}")

    df["Protocol"] = df["Protocol"].fillna(-1).astype(np.int32)

    print("[ENCODE] Protocol unique values:")
    print(sorted(df["Protocol"].unique().tolist())[:20])

    return df, None


# =========================================================
# 시간 정렬
# =========================================================

def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Timestamp 오름차순으로 전체 데이터를 정렬한다.

    여러 CSV를 병합하면 날짜 순서가 뒤섞일 수 있으므로
    sequence 생성 전에 전역 시간 순서를 보장한다.
    각 host 내부 정렬은 create_sequences에서 별도로 수행한다.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Timestamp 오름차순으로 정렬된 DataFrame. 인덱스는 재설정된다.
    """
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print("[ORDER] Timestamp 기준 시간 정렬 완료")
    return df


# =========================================================
# 내부 IP 판별
# =========================================================

def is_internal_ip(ip: str) -> bool:
    """
    IP 주소가 INTERNAL_IP_PREFIXES에 정의된 내부 네트워크 대역인지 판별한다.

    Parameters
    ----------
    ip : str
        확인할 IP 주소 문자열

    Returns
    -------
    bool
        내부 대역이면 True, 아니면 False
    """
    return str(ip).startswith(INTERNAL_IP_PREFIXES)


# =========================================================
# flow 분포 분석
# =========================================================

def analyze_flow_distribution(df: pd.DataFrame) -> None:
    """
    Source IP 기준 flow 수 분포를 분석하고 적절한 window_size 후보를 제안한다.

    내부 IP 호스트의 flow 수 중앙값을 기준으로 window_size 권장값을 산출한다.
    결과는 출력만 하며 파이프라인에 자동 반영하지 않는다.
    (window_size는 모델 성능·클래스 불균형·sequence 의미 보존을 함께 고려해야 하므로
    실험자가 직접 결정하는 방식을 유지한다)

    권장값 산출 기준:
        중앙값 < 5  → window_size = 3
        중앙값 < 20 → window_size = 5
        중앙값 < 50 → window_size = 10
        그 외        → window_size = 15

    Parameters
    ----------
    df : pd.DataFrame
        Label_binary 컬럼이 포함된 전처리 완료 DataFrame
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


# =========================================================
# flat 데이터 생성 (RF / XGBoost)
# =========================================================

def create_flat_data(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    RF / XGBoost 학습용 2D feature 배열을 생성한다.

    sequence를 구성하지 않고 각 flow를 독립적인 샘플로 취급한다.
    feature_cols 중 실제로 존재하지 않는 컬럼은 경고 후 스킵한다.

    Parameters
    ----------
    df : pd.DataFrame
        전처리 완료된 DataFrame
    feature_cols : list[str]
        모델 입력에 사용할 feature 컬럼 목록

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float32
        단일 flow 기반 feature 행렬
    y : np.ndarray, shape (n_samples,), dtype int32
        각 flow의 이진 라벨 (0=Normal, 1=Botnet)
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


# =========================================================
# sequence 데이터 생성 (CNN-LSTM)
# =========================================================

def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 15,
    step_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CNN-LSTM 학습용 3D sequence 배열을 생성한다.

    내부 IP 호스트별로 flow를 Timestamp 순으로 정렬한 뒤
    sliding window를 적용하여 호스트 단위 행동 흐름을 sequence로 구성한다.
    flow 수가 window_size에 미치지 못하는 호스트는 sequence를 생성하지 않는다.

    라벨링 기준:
        window 내 Botnet flow가 1개 이상 → sequence label = 1
        window 내 모두 Normal          → sequence label = 0

    Parameters
    ----------
    df : pd.DataFrame
        전처리 완료된 전체 DataFrame
    feature_cols : list[str]
        sequence 입력에 사용할 feature 컬럼 목록
    window_size : int, default=15
        sequence 하나를 구성하는 연속 flow 개수
    step_size : int, default=5
        sliding window 이동 간격 (step_size < window_size이면 overlap 발생)

    Returns
    -------
    X : np.ndarray, shape (n_sequences, window_size, n_features), dtype float32
        CNN-LSTM 입력 sequence 배열
    y : np.ndarray, shape (n_sequences,), dtype int32
        각 sequence의 이진 라벨 (0=Normal, 1=Botnet)
    """
    valid_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"[WARN] sequence 생성에서 없는 컬럼 (스킵): {missing_cols}")

    # 외부 IP(공격자 서버 등)는 감염 호스트 관점의 행동 패턴을 담지 않으므로 제외
    internal_mask = df["Source IP"].apply(is_internal_ip)
    df_internal = df[internal_mask].copy()

    print(f"\n[SEQ] 내부 IP flow 수: {len(df_internal):,}")
    print(f"[SEQ] 내부 IP 종류: {df_internal['Source IP'].nunique()}개")

    sequences = []
    labels = []
    skipped = 0

    grouped = df_internal.groupby("Source IP")

    for src_ip, group in grouped:
        group = group.sort_values("Timestamp").reset_index(drop=True)

        features = group[valid_cols].values
        label_vals = group["Label_binary"].values
        n_flows = len(features)

        if n_flows < window_size:
            skipped += 1
            continue

        for start in range(0, n_flows - window_size + 1, step_size):
            end = start + window_size
            seq = features[start:end]
            seq_labels = label_vals[start:end]
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


# =========================================================
# 저장
# =========================================================

def save_outputs(
    df: pd.DataFrame,
    protocol_encoder: Optional[LabelEncoder],
    save_dir: str,
) -> None:
    """
    전처리 완료 DataFrame을 parquet 형식으로 저장한다.

    저장 컬럼: Source IP, Timestamp, Label, Label_binary + ML_FEATURES 교집합.
    protocol_encoder가 None이 아닌 경우(LabelEncoder 사용 시) pkl로 함께 저장한다.
    현재 구현에서는 Protocol 원값을 유지하므로 encoder는 저장하지 않는다.

    Parameters
    ----------
    df : pd.DataFrame
    protocol_encoder : Optional[LabelEncoder]
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
    numpy 배열 쌍(X, y)을 .npy 파일로 저장한다.

    저장 파일명: X_{name}.npy, y_{name}.npy

    Parameters
    ----------
    data : np.ndarray
        입력 feature 배열
    label : np.ndarray
        라벨 배열
    save_dir : str
        저장 디렉터리 경로
    name : str
        파일명 식별자 (예: "flat", "seq_w15")
    """
    np.save(os.path.join(save_dir, f"X_{name}.npy"), data)
    np.save(os.path.join(save_dir, f"y_{name}.npy"), label)
    print(f"[SAVE] X_{name}.npy / y_{name}.npy 저장 완료")


# =========================================================
# 미리보기
# =========================================================

def preview_data(df: pd.DataFrame) -> None:
    """
    전처리 결과의 주요 컬럼 상위 5행과 전체 shape를 출력한다.

    파이프라인 실행 중 중간 결과를 눈으로 빠르게 확인하는 용도이다.

    Parameters
    ----------
    df : pd.DataFrame
    """
    print("\n[PREVIEW] 상위 5행")
    print(df[["Source IP", "Timestamp", "Label", "Label_binary",
              "Flow Duration", "Protocol"]].head())
    print(f"\n[PREVIEW] 전체 shape: {df.shape}")


# =========================================================
# 메인 파이프라인
# =========================================================

def main():
    """
    전처리 파이프라인 전체를 순서대로 실행한다.

    수행 순서:
    1. CSV 로드 및 컬럼 정규화
    2. 필수 컬럼 검증
    3. 기본 정제 (결측·inf·Timestamp 처리)
    4. 이진 라벨 생성
    5. Protocol 정수 변환 및 시간 정렬
    6. 전처리 결과 parquet 저장
    7. flow 분포 분석 (window_size 결정 참고용)
    8. flat 데이터 생성 및 저장 (RF / XGBoost)
    9. sequence 데이터 생성 및 저장 (CNN-LSTM)
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

    df = create_binary_label(df, strict=False)  # 데이터 파악 중이면 strict=True로 변경
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