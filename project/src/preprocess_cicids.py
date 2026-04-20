import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Optional


# =========================================================
# 경로 설정
# ---------------------------------------------------------
# __file__ 기준으로 두 단계 상위 디렉터리를 프로젝트 루트로 삼는다.
# raw 데이터는 data/raw/cic-ids2017/, 전처리 결과는 data/processed/에 저장한다.
# =========================================================
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../project/src
BASE_DIR  = os.path.dirname(_SRC_DIR)                    # .../project

RAW_DIR  = os.path.join(BASE_DIR, "data", "raw", "cic-ids2017")
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
# BOTNET_LABELS     : 1로 매핑할 라벨 집합
# KNOWN_NON_BOT_LABELS : 0으로 매핑할 것이 확실한 라벨 집합
#
# strict=True일 때는 두 집합 모두에 없는 "진짜 unknown 라벨"에서만 예외를 발생시킨다.
# (이전 구현은 bot이 아닌 모든 라벨에서 예외를 던지는 문제가 있었다)
#
# 이 프로젝트의 task 정의:
#   Botnet(감염 호스트의 C&C 통신) vs 나머지 모든 트래픽
#   나머지에는 BENIGN뿐만 아니라 DDoS, PortScan 등 다른 공격도 포함된다.
#   발표/논문에서 "정상 vs 봇넷"으로 표현하면 부정확하다.
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

# 0으로 처리된 unknown 라벨 빈도 수집용 전역 카운터
# create_binary_label() 시작 시 항상 clear()하므로 재실행 시 누적되지 않는다.
_label_counter: Counter = Counter()


# =========================================================
# CSV 로드
# =========================================================

def load_all_csv(raw_dir: str) -> pd.DataFrame:
    """
    디렉터리 내 모든 CSV 파일을 읽어 하나의 DataFrame으로 병합한다.

    CIC-IDS2017 TrafficLabeling 버전은 날짜별로 CSV가 분리되어 있고
    파일마다 인코딩이 다를 수 있으므로 utf-8 → cp1252 → latin-1 순으로
    인코딩을 순차 시도한다.

    UnicodeDecodeError가 아닌 다른 오류(손상된 파일, 구분자 문제 등)는
    재시도해도 의미가 없으므로 즉시 루프를 탈출하고,
    실제 원인 예외를 ValueError.__cause__에 포함하여 올린다.

    Parameters
    ----------
    raw_dir : str
        CSV 파일이 위치한 디렉터리 경로

    Returns
    -------
    pd.DataFrame
        전체 CSV를 행 방향으로 이어 붙인 DataFrame. 인덱스는 0부터 재설정된다.

    Raises
    ------
    FileNotFoundError
        raw_dir에 CSV 파일이 하나도 없을 때
    ValueError
        특정 파일의 모든 인코딩 시도가 실패했을 때
        (실제 원인 예외를 __cause__로 포함)
    """
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {raw_dir}")

    print(f"[INFO] CSV 파일 수: {len(csv_files)}")

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
                # UnicodeDecodeError 외의 오류는 재시도해도 해결되지 않으므로 즉시 중단
                last_error = e
                print(f"[WARN] {os.path.basename(file_path)} 로드 실패 ({encoding}): {e}")
                break

        if temp_df is None:
            raise ValueError(f"파일 로드 실패: {os.path.basename(file_path)}") from last_error

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
    """
    df.columns = df.columns.str.strip()
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    REQUIRED_BASE_COLS에 정의된 필수 컬럼이 모두 존재하는지 검증한다.

    파이프라인 초반에 호출하여 후속 단계에서 KeyError가 발생하는 상황을 방지한다.

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
    1. Label / Source IP 결측 및 빈 문자열 제거
       (astype(str) 변환 전에 수행해야 NaN이 "nan" 문자열로 변환되기 전에 제거 가능)
    2. 문자열 컬럼 공백 제거
    3. ML_FEATURES 수치형 강제 변환 (mixed type → numeric)
    4. Timestamp → datetime 변환 (day-first, mixed format)
       (생략하면 sort_by_time()이 문자열 정렬이 되고 .dt.date 호출에서 오류 발생)
    5. Timestamp 파싱 실패 행 제거 또는 forward-fill
    6. 수치형 inf → NaN 변환
    7. 수치형 NaN → 0 대체 (옵션)

    Parameters
    ----------
    df : pd.DataFrame
        병합된 원시 DataFrame
    fill_numeric_na_with_zero : bool, default=True
        True이면 수치형 NaN을 0으로 대체한다.
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

    # 1. Label / Source IP 결측 제거 (str 변환 전에 수행)
    before_required_drop = len(df)
    df = df.dropna(subset=["Label", "Source IP"])
    print(f"\n[CLEAN] Label/Source IP NaN 제거 행 수: {before_required_drop - len(df):,}")

    before_empty_drop = len(df)
    df = df[
        (df["Label"].astype(str).str.strip() != "")
        & (df["Label"].astype(str).str.strip().str.lower() != "nan")
        & (df["Source IP"].astype(str).str.strip() != "")
        & (df["Source IP"].astype(str).str.strip().str.lower() != "nan")
    ]
    print(f"[CLEAN] Label/Source IP 빈 문자열 제거 행 수: {before_empty_drop - len(df):,}")

    # 2. 문자열 컬럼 공백 제거
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # 3. ML_FEATURES 수치형 강제 변환
    # CICFlowMeter 출력에서 일부 feature 컬럼이 mixed type으로 읽혀 object로 남는 경우가 있다.
    # 이 상태로 두면 fillna(0)과 astype(np.float32)에서 오류가 발생한다.
    converted, failed = 0, []
    for col in [c for c in ML_FEATURES if c in df.columns]:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted += 1
            if df[col].isna().sum() > 0:
                failed.append(col)
    if converted > 0:
        print(f"[CLEAN] ML_FEATURES object → numeric 강제 변환 컬럼 수: {converted}")
    if failed:
        print(f"[WARN]  변환 후 NaN 발생 컬럼: {failed}")

    # 4. Timestamp → datetime 변환
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )
    parsed_na = df["Timestamp"].isna().sum()
    print(f"\n[CLEAN] Timestamp 전체 NaT 행 수: {parsed_na:,}")

    # 5. Timestamp 파싱 실패 처리
    if drop_bad_timestamp:
        before_ts = len(df)
        df = df.dropna(subset=["Timestamp"])
        print(f"[CLEAN] Timestamp NaT 제거 행 수: {before_ts - len(df):,}")
    else:
        df["Timestamp"] = df["Timestamp"].ffill()
        print(f"[CLEAN] Timestamp ffill 후 남은 NaT 행 수: {df['Timestamp'].isna().sum():,}")

    # 6. inf → NaN 변환
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_before = np.isinf(df[numeric_cols].to_numpy()).sum() if len(numeric_cols) > 0 else 0
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"[CLEAN] inf → NaN 변환 개수: {inf_before:,}")
    print(f"[CLEAN] 현재 수치형 전체 NaN 개수: {df[numeric_cols].isna().sum().sum():,}")
    log_inf_nan_status(df, "after_inf_to_nan")

    # 7. 수치형 NaN → 0 대체
    if fill_numeric_na_with_zero:
        nan_before = df[numeric_cols].isna().sum().sum()
        df[numeric_cols] = df[numeric_cols].fillna(0)
        print(f"[CLEAN] 수치형 NaN -> 0 대체 개수: {nan_before:,}")
        print(f"[CLEAN] 수치형 NaN 잔여 개수: {df[numeric_cols].isna().sum().sum():,}")
    else:
        print("[CLEAN] 수치형 NaN을 0으로 대체하지 않음")

    print(f"\n[CLEAN] 제거된 전체 행 수: {before_rows - len(df):,}")
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

    매핑 우선순위:
        1) BOTNET_LABELS          → 1
        2) KNOWN_NON_BOT_LABELS   → 0
        3) 그 외 (unknown)        → strict=True이면 ValueError, False이면 0 + 카운터 기록

    strict=True의 의미:
        BOTNET_LABELS도 KNOWN_NON_BOT_LABELS도 아닌 "진짜 미등록 라벨"에서만 예외가 발생한다.
        기존에 알려진 non-bot 라벨은 strict 여부와 무관하게 항상 0으로 처리된다.

    Parameters
    ----------
    label : str
    strict : bool, default=False

    Returns
    -------
    int : 1 (Botnet) 또는 0 (Non-Botnet)

    Raises
    ------
    ValueError
        strict=True이고 두 집합 모두에 없는 unknown 라벨일 때
    """
    label = str(label).strip().lower()

    if label in BOTNET_LABELS:
        return 1
    if label in KNOWN_NON_BOT_LABELS:
        return 0

    # 두 집합 모두에 없는 unknown 라벨
    _label_counter[label] += 1
    if strict:
        raise ValueError(
            f"알 수 없는 라벨 값: '{label}'\n"
            "BOTNET_LABELS 또는 KNOWN_NON_BOT_LABELS에 추가하거나 strict=False로 실행하세요."
        )
    return 0


def create_binary_label(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Label 컬럼을 기반으로 이진 라벨 컬럼(Label_binary)을 생성한다.

    호출 시마다 _label_counter를 초기화하므로 같은 프로세스에서 재실행해도
    이전 실행의 카운트가 누적되지 않는다.

    Parameters
    ----------
    df : pd.DataFrame
    strict : bool, default=False
        True이면 unknown 라벨 등장 시 즉시 예외 발생 (데이터 구조 파악용)
        False이면 0으로 처리 후 카운터에 기록 (실험/학습용)

    Returns
    -------
    pd.DataFrame
        Label_binary 컬럼이 추가된 DataFrame
    """
    _label_counter.clear()  # 재실행 시 이전 카운트 누적 방지

    df["Label_binary"] = df["Label"].map(
        lambda x: _map_binary_label(x, strict=strict)
    )

    if _label_counter:
        print("[LABEL] Unknown 라벨 (0으로 처리 / KNOWN_NON_BOT_LABELS 추가 고려):")
        for lbl, cnt in _label_counter.most_common():
            print(f"  '{lbl}': {cnt:,}건")
    else:
        print("[LABEL] Unknown 라벨 없음")

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
    숫자로 변환할 수 없는 값은 -1로 채운다.
    두 번째 반환값은 항상 None이며 인터페이스 일관성을 위한 자리 표시자이다.
    """
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")
    print(f"[ENCODE] Protocol 숫자 변환 실패 수: {df['Protocol'].isna().sum():,}")
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
    """
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print("[ORDER] Timestamp 기준 시간 정렬 완료")
    return df


# =========================================================
# 내부 IP 판별
# =========================================================

def is_internal_ip(ip: str) -> bool:
    """IP 주소가 INTERNAL_IP_PREFIXES에 정의된 내부 네트워크 대역인지 판별한다."""
    return str(ip).startswith(INTERNAL_IP_PREFIXES)


# =========================================================
# flow 분포 분석
# =========================================================

def analyze_flow_distribution(df: pd.DataFrame) -> None:
    """
    Source IP 기준 flow 수 분포를 분석하고 적절한 window_size 후보를 제안한다.

    결과는 출력만 하며 파이프라인에 자동 반영하지 않는다.
    window_size는 모델 성능·클래스 불균형·sequence 의미 보존을 함께 고려해야 하므로
    실험자가 직접 결정하는 방식을 유지한다.

    권장값 산출 기준:
        중앙값 < 5  → 3  /  중앙값 < 20 → 5
        중앙값 < 50 → 10 /  그 외        → 15
    """
    all_counts = df.groupby("Source IP").size()
    print("\n[FLOW DIST] 전체 src_ip 당 flow 수 분포:")
    print(all_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

    internal_mask   = df["Source IP"].apply(is_internal_ip)
    internal_df     = df[internal_mask]
    internal_counts = internal_df.groupby("Source IP").size()

    print(f"\n[FLOW DIST] 내부 IP flow 수: {internal_mask.sum():,}개")
    print(f"[FLOW DIST] 내부 IP 종류: {internal_counts.shape[0]}개")
    print("\n[FLOW DIST] 내부 src_ip 당 flow 수 분포:")
    print(internal_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

    total_bot    = int(df["Label_binary"].sum())
    internal_bot = int(internal_df["Label_binary"].sum())
    print(f"\n[FLOW DIST] 내부 IP 중 Botnet flow 수: {internal_bot:,}")
    if total_bot > 0:
        print(f"[FLOW DIST] 전체 Botnet 대비 내부 IP 비율: {internal_bot / total_bot * 100:.1f}%")

    median_val = internal_counts.median()
    if   median_val < 5:  recommended = 3
    elif median_val < 20: recommended = 5
    elif median_val < 50: recommended = 10
    else:                 recommended = 15

    candidates = sorted({max(3, recommended - 5), recommended, recommended + 5})
    print(f"\n[FLOW DIST] → 추천 window_size: {recommended}")
    print(f"[FLOW DIST] → 실험 후보: {', '.join(map(str, candidates))}")


# =========================================================
# host 그룹 단위 train / val / test split
# =========================================================

def split_host_groups(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    내부 IP 호스트를 (Source IP, 날짜) 단위로 train / val / test에 분리한다.

    왜 host 그룹 단위로 split하는가:
        sequence는 같은 (host, 날짜) 그룹 안에서 sliding window로 생성된다.
        flow 단위로 random split하면 동일한 그룹에서 나온 겹치는 sequence가
        train과 test 양쪽에 동시에 들어가 data leakage가 발생한다.
        따라서 "split 먼저 → 각 subset에서 sequence 생성"의 순서를 지켜야 한다.

    split 방식:
        전체 그룹 → (train+val) / test → train / val
        stratify: 그룹 내 Botnet 존재 여부 (클래스 불균형 보존)

    외부 IP flow는 flat 모델(RF/XGBoost) 학습에 포함되므로
    train subset에만 추가한다. val/test는 내부 IP 호스트 탐지 성능 측정에 집중한다.

    Parameters
    ----------
    df : pd.DataFrame
        Label_binary 컬럼이 포함된 전처리 완료 DataFrame
    val_ratio : float, default=0.1
        전체 대비 validation 그룹 비율
    test_ratio : float, default=0.2
        전체 대비 test 그룹 비율
    random_state : int, default=42

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
    """
    internal_mask = df["Source IP"].apply(is_internal_ip)
    df_internal   = df[internal_mask].copy()
    df_external   = df[~internal_mask]

    df_internal["_date"]      = df_internal["Timestamp"].dt.date
    df_internal["_group_key"] = (
        df_internal["Source IP"].astype(str) + "_" + df_internal["_date"].astype(str)
    )

    # 그룹별 Botnet 존재 여부 (stratify 기준)
    group_info = (
        df_internal.groupby("_group_key")["Label_binary"]
        .sum()
        .gt(0)
        .astype(int)
        .reset_index()
    )

    print("[SPLIT] 그룹 라벨 분포:")
    print(group_info["Label_binary"].value_counts().sort_index())

    group_keys    = group_info["_group_key"].tolist()
    group_has_bot = group_info["Label_binary"].tolist()

    # 1차 분리: (train+val) / test
    keys_trainval, keys_test, _, _ = train_test_split(
        group_keys, group_has_bot,
        test_size=test_ratio,
        random_state=random_state,
        stratify=group_has_bot,
    )

    # 2차 분리: train / val
    group_label_map = dict(zip(group_keys, group_has_bot))
    labels_trainval = [group_label_map[k] for k in keys_trainval]
    adjusted_val    = val_ratio / (1.0 - test_ratio)
    keys_train, keys_val, _, _ = train_test_split(
        keys_trainval, labels_trainval,
        test_size=adjusted_val,
        random_state=random_state,
        stratify=labels_trainval,
    )

    keys_train_set = set(keys_train)
    keys_val_set   = set(keys_val)
    keys_test_set  = set(keys_test)

    internal_train = df_internal[df_internal["_group_key"].isin(keys_train_set)]
    internal_val   = df_internal[df_internal["_group_key"].isin(keys_val_set)]
    internal_test  = df_internal[df_internal["_group_key"].isin(keys_test_set)]

    # 임시 컬럼 제거
    drop_cols = ["_date", "_group_key"]
    internal_train = internal_train.drop(columns=drop_cols)
    internal_val   = internal_val.drop(columns=drop_cols)
    internal_test  = internal_test.drop(columns=drop_cols)

    # 모두 내부 IP 만으로 split
    df_train = internal_train.reset_index(drop=True)
    df_val   = internal_val.reset_index(drop=True)
    df_test  = internal_test.reset_index(drop=True)

    print(f"\n[SPLIT] 호스트 그룹 수: {len(group_keys):,}")
    print(f"[SPLIT] train 그룹: {len(keys_train):,} / val 그룹: {len(keys_val):,} / test 그룹: {len(keys_test):,}")
    print(f"[SPLIT] train flow: {len(df_train):,} / val flow: {len(df_val):,} / test flow: {len(df_test):,}")
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  {name} Botnet 비율: {subset['Label_binary'].mean():.4f}")

    return df_train, df_val, df_test


# =========================================================
# flat 데이터 생성 (RF / XGBoost)
# =========================================================

def create_flat_data(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    RF / XGBoost 학습용 2D feature 배열을 생성한다.

    트리 계열 모델은 feature scaling이 필요 없으므로 raw 값을 그대로 반환한다.
    scaling은 호출 측에서 필요에 따라 별도로 적용한다.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float32
    y : np.ndarray, shape (n_samples,), dtype int32
    """
    valid_cols   = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"[WARN] flat 데이터에서 없는 컬럼 (스킵): {missing_cols}")

    X = df[valid_cols].values.astype(np.float32)
    y = df["Label_binary"].values.astype(np.int32)

    print(f"\n[FLAT] X shape: {X.shape}  Botnet 비율: {y.mean():.4f}  ({y.sum():,}/{len(y):,})")
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

    groupby 기준: (Source IP, 날짜)
        Source IP만으로 groupby하면 서로 다른 날짜의 트래픽이 하나의 sequence에 섞여
        시간적 연속성이 깨진다. 날짜를 추가해 같은 날의 flow만 묶는다.

    이 함수는 split 이후 각 subset(train/val/test)에 대해 개별 호출해야 한다.
    전체 데이터에 먼저 sequence를 생성한 뒤 split하면 겹치는 sequence가
    train과 test 양쪽에 들어가 data leakage가 발생한다.

    라벨링: window 내 Botnet flow ≥ 1 → 1, 전부 Normal → 0

    Parameters
    ----------
    df : pd.DataFrame
        split_host_groups()로 분리된 subset DataFrame
    feature_cols : list[str]
    window_size : int, default=15
    step_size : int, default=5

    Returns
    -------
    X : np.ndarray, shape (n_sequences, window_size, n_features), dtype float32
    y : np.ndarray, shape (n_sequences,), dtype int32
    """
    valid_cols   = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"[WARN] sequence 생성에서 없는 컬럼 (스킵): {missing_cols}")

    internal_mask = df["Source IP"].apply(is_internal_ip)
    df_internal   = df[internal_mask].copy()
    df_internal["_date"] = df_internal["Timestamp"].dt.date

    print(f"\n[SEQ] 내부 IP flow 수: {len(df_internal):,} / 종류: {df_internal['Source IP'].nunique()}개")

    sequences, labels = [], []
    skipped = 0

    for (src_ip, date), group in df_internal.groupby(["Source IP", "_date"]):
        group      = group.sort_values("Timestamp").reset_index(drop=True)
        features   = group[valid_cols].values
        label_vals = group["Label_binary"].values
        n_flows    = len(features)

        if n_flows < window_size:
            skipped += 1
            continue

        for start in range(0, n_flows - window_size + 1, step_size):
            end = start + window_size
            sequences.append(features[start:end])
            labels.append(1 if label_vals[start:end].sum() > 0 else 0)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels,    dtype=np.int32)

    print(f"[SEQ] window={window_size}, step={step_size}  생성: {len(sequences):,}  스킵: {skipped}")
    print(f"[SEQ] X shape: {X.shape}  Botnet 비율: {y.mean():.4f}  ({y.sum():,}/{len(y):,})")

    return X, y

# =========================================================
# window-flat 데이터 생성 (RF / XGBoost 공정 비교용)
# =========================================================

def create_window_flat_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 15,
    step_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RF / XGBoost용 window-flat 데이터를 생성한다.

    CNN-LSTM용 sequence 생성과 동일한 기준으로 window를 만들되,
    각 window를 1차원 벡터로 펼쳐서(flatten) 반환한다.

    이렇게 하면 같은 window 샘플을
    - CNN-LSTM에는 (window_size, n_features) 형태로,
    - RF/XGBoost에는 (window_size * n_features,) 형태로
    사용할 수 있어 공정 비교가 가능해진다.

    groupby 기준: (Source IP, 날짜)
        Source IP만으로 groupby하면 서로 다른 날짜의 트래픽이 하나의 window에 섞여
        시간적 연속성이 깨질 수 있으므로, 날짜를 함께 사용한다.

    라벨링 기준:
        window 내 Botnet flow가 1개 이상 → window label = 1
        window 내 모두 Non-Botnet       → window label = 0

    Parameters
    ----------
    df : pd.DataFrame
        split_host_groups()로 분리된 subset DataFrame
    feature_cols : list[str]
        window 생성에 사용할 feature 컬럼 목록
    window_size : int, default=15
        하나의 window를 구성하는 연속 flow 개수
    step_size : int, default=5
        sliding window 이동 간격

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size * n_features), dtype float32
        RF / XGBoost 입력용 flattened window 배열
    y : np.ndarray, shape (n_windows,), dtype int32
        각 window의 이진 라벨
    """
    valid_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"[WARN] window-flat 생성에서 없는 컬럼 (스킵): {missing_cols}")

    internal_mask = df["Source IP"].apply(is_internal_ip)
    df_internal = df[internal_mask].copy()
    df_internal["_date"] = df_internal["Timestamp"].dt.date

    print(
        f"\n[WIN-FLAT] 내부 IP flow 수: {len(df_internal):,} / "
        f"종류: {df_internal['Source IP'].nunique()}개"
    )

    windows = []
    labels = []
    skipped = 0

    grouped = df_internal.groupby(["Source IP", "_date"])

    for (src_ip, date), group in grouped:
        group = group.sort_values("Timestamp").reset_index(drop=True)

        features = group[valid_cols].values
        label_vals = group["Label_binary"].values
        n_flows = len(features)

        if n_flows < window_size:
            skipped += 1
            continue

        for start in range(0, n_flows - window_size + 1, step_size):
            end = start + window_size

            window = features[start:end]
            window_label = 1 if label_vals[start:end].sum() > 0 else 0

            windows.append(window.reshape(-1))
            labels.append(window_label)

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print(f"[WIN-FLAT] window_size={window_size}, step_size={step_size}")
    print(f"[WIN-FLAT] 생성된 window 수: {len(windows):,}")
    print(f"[WIN-FLAT] flow 부족으로 스킵된 (src_ip, date) 수: {skipped}")
    print(f"[WIN-FLAT] X shape: {X.shape}")
    print(f"[WIN-FLAT] y shape: {y.shape}")
    print(f"[WIN-FLAT] Botnet 비율: {y.mean():.4f}")
    print(f"[WIN-FLAT] Botnet 수: {y.sum():,} / 전체: {len(y):,}")

    return X, y

# =========================================================
# scaler fit + 저장 (CNN-LSTM 전용)
# =========================================================

def fit_and_save_scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_dir: str,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    StandardScaler를 train에만 fit하고 val / test는 transform만 적용한다.

    트리 계열 모델(RF / XGBoost)은 scaling이 필요 없으므로 이 함수를 사용하지 않는다.
    CNN-LSTM처럼 gradient 기반 모델에서 사용한다.

    3D sequence 배열 (n, window, features)은 (n*window, features)로 reshape한 뒤
    fit/transform하고 원래 shape으로 복원한다.
    fit된 scaler는 pkl로 저장하여 추론 시 동일한 변환을 재현한다.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
    save_dir : str
    name : str  파일명 식별자 (예: "seq_w15")

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray, dtype float32
    """
    scaler = StandardScaler()
    is_3d  = X_train.ndim == 3

    if is_3d:
        n_win, n_feat = X_train.shape[1], X_train.shape[2]
        X_tr = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(-1, n_win, n_feat)
        X_va = scaler.transform(X_val.reshape(-1, n_feat)).reshape(-1, n_win, n_feat)
        X_te = scaler.transform(X_test.reshape(-1, n_feat)).reshape(-1, n_win, n_feat)
    else:
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)
        X_te = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(save_dir, f"scaler_{name}.pkl"))
    print(f"[SCALER] scaler_{name}.pkl 저장 완료")

    return X_tr.astype(np.float32), X_va.astype(np.float32), X_te.astype(np.float32)

# =========================================================
# 시각 자료 저장
# =========================================================

def analyze_flow_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """
    Source IP 기준 flow 수 분포를 분석하고 시각화하여 저장한다.
    """
    # 1. 데이터 준비
    internal_mask = df["Source IP"].apply(is_internal_ip)
    internal_counts = df[internal_mask].groupby("Source IP").size().rename("flow_count")
    
    print("\n[FLOW DIST] 내부 src_ip 당 flow 수 분포:")
    stats = internal_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
    print(stats)

    # 2. 시각화 설정
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- (1) Histogram (Distribution) ---
    # 데이터의 편차가 매우 클 것이므로 log_scale=True를 권장합니다.
    sns.histplot(internal_counts, bins=50, kde=True, ax=axes[0], log_scale=True, color='skyblue')
    axes[0].set_title("Distribution of Flow Counts per Internal IP (Log Scale)")
    axes[0].set_xlabel("Number of Flows (Log Scale)")
    axes[0].set_ylabel("Frequency (Number of IPs)")
    
    # 윈도우 사이즈 후보 가이드라인 표시 (예: median, 90th percentile)
    median_val = internal_counts.median()
    axes[0].axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.1f}')
    axes[0].legend()

    # --- (2) Boxplot (Outliers & Quartiles) ---
    sns.boxplot(x=internal_counts, ax=axes[1], color='lightgreen')
    axes[1].set_xscale("log") # 박스플롯도 로그 스케일 적용
    axes[1].set_title("Boxplot of Flow Counts per Internal IP (Log Scale)")
    axes[1].set_xlabel("Number of Flows (Log Scale)")

    plt.tight_layout()
    
    # 3. 파일 저장
    plot_path = os.path.join(save_dir, "flow_distribution.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[FLOW DIST] 시각화 결과 저장 완료: {plot_path}")
    
    # 추천 window_size 로직 (기존 유지)
    if median_val < 5: recommended = 3
    elif median_val < 20: recommended = 5
    elif median_val < 50: recommended = 10
    else: recommended = 15
    
    print(f"→ 추천 window_size: {recommended} (Median 기반)")

# =========================================================
# 저장
# =========================================================

def save_outputs(
    df: pd.DataFrame,
    protocol_encoder: Optional[LabelEncoder],
    save_dir: str,
) -> None:
    """전처리 완료 DataFrame을 parquet 형식으로 저장한다."""
    save_cols = (
        ["Source IP", "Timestamp", "Label", "Label_binary"]
        + [col for col in ML_FEATURES if col in df.columns]
    )
    df[save_cols].to_parquet(os.path.join(save_dir, "cicids2017_traffic.parquet"), index=False)

    if protocol_encoder is not None:
        joblib.dump(protocol_encoder, os.path.join(save_dir, "protocol_label_encoder.pkl"))
        print("[SAVE] Protocol encoder 저장 완료")
    else:
        print("[SAVE] Protocol encoder 저장 생략 (원값 유지 방식 사용)")

    print("[SAVE] 전처리 데이터 저장 완료")


def save_numpy(data: np.ndarray, label: np.ndarray, save_dir: str, split_name: str) -> None:
    """numpy 배열 쌍(X, y)을 .npy 파일로 저장한다."""
    np.save(os.path.join(save_dir, f"X_{split_name}.npy"), data)
    np.save(os.path.join(save_dir, f"y_{split_name}.npy"), label)
    print(f"[SAVE] {save_dir}/X_{split_name}.npy / y_{split_name}.npy 저장 완료")


# =========================================================
# 미리보기
# =========================================================

def preview_data(df: pd.DataFrame) -> None:
    """전처리 결과의 주요 컬럼 상위 5행과 전체 shape를 출력한다."""
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
    1.  CSV 로드 및 컬럼 정규화
    2.  필수 컬럼 검증
    3.  기본 정제 (결측·inf·Timestamp 처리, ML_FEATURES 수치형 강제 변환)
    4.  이진 라벨 생성
    5.  Protocol 정수 변환 및 시간 정렬
    6.  전처리 결과 parquet 저장
    7.  flow 분포 분석 (window_size 결정 참고용)
    8.  host 그룹 단위 train / val / test split
    9.  window-flat 데이터 생성 및 저장 (RF / XGBoost)
    10. sequence 생성 → scaler fit → 저장 (CNN-LSTM)
    """
    print("=== CIC-IDS2017 TrafficLabeling Preprocessing Start ===")
    print(f"[PATH] RAW_DIR : {RAW_DIR}")
    print(f"[PATH] SAVE_DIR: {SAVE_DIR}")

    WINDOW_SIZE = 15
    STEP_SIZE = 5

    WINFLAT_DIR = os.path.join(SAVE_DIR, "winflat")
    SEQ_DIR = os.path.join(SAVE_DIR, "seq")

    os.makedirs(WINFLAT_DIR, exist_ok=True)
    os.makedirs(SEQ_DIR, exist_ok=True)

    print(f"\n[CONFIG] WINDOW_SIZE: {WINDOW_SIZE} / STEP_SIZE: {STEP_SIZE}")
    print(f"[PATH] WINFLAT_DIR: {WINFLAT_DIR}")
    print(f"[PATH] SEQ_DIR    : {SEQ_DIR}")

    # ── 공통 전처리 ──────────────────────────────────────────
    df = load_all_csv(RAW_DIR)
    df = normalize_column_names(df)
    validate_columns(df)

    df = basic_cleaning(df, fill_numeric_na_with_zero=True, drop_bad_timestamp=True)
    df = create_binary_label(df, strict=False)
    df, protocol_encoder = encode_protocol(df)
    df = sort_by_time(df)

    preview_data(df)
    save_outputs(df, protocol_encoder, SAVE_DIR)
    analyze_flow_distribution(df, SAVE_DIR)

    # ── host 그룹 단위 split ─────────────────────────────────
    # sequence 생성 전에 split해야 data leakage를 방지할 수 있다.
    df_train, df_val, df_test = split_host_groups(df)

    # ── RF / XGBoost용 window-flat 데이터 저장 ───────────────
    print(f"\n[STEP] RF / XGBoost 용 window-flat 데이터 생성 (window_size={WINDOW_SIZE})")
    for split_name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X_win, y_win = create_window_flat_data(
            df=subset,
            feature_cols=ML_FEATURES,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
        )
        save_numpy(X_win, y_win, WINFLAT_DIR, split_name)

    # ── CNN-LSTM용 sequence 데이터 생성 및 저장 ──────────────
    seq_tag = f"seq_w{WINDOW_SIZE}"
    print(f"\n[STEP] CNN-LSTM 용 sequence 데이터 생성 (window_size={WINDOW_SIZE})")

    X_tr, y_tr = create_sequences(df_train, ML_FEATURES, WINDOW_SIZE, STEP_SIZE)
    X_va, y_va = create_sequences(df_val, ML_FEATURES, WINDOW_SIZE, STEP_SIZE)
    X_te, y_te = create_sequences(df_test, ML_FEATURES, WINDOW_SIZE, STEP_SIZE)

    X_tr, X_va, X_te = fit_and_save_scaler(X_tr, X_va, X_te, SAVE_DIR, seq_tag)

    save_numpy(X_tr, y_tr, SEQ_DIR, "train")
    save_numpy(X_va, y_va, SEQ_DIR, "val")
    save_numpy(X_te, y_te, SEQ_DIR, "test")

    print("\n[저장된 파일]")
    print("  cicids2017_traffic.parquet")
    print("  scaler_seq_w15.pkl")
    print("  ")
    print("  [winflat]")
    print("    X_train.npy / y_train.npy")
    print("    X_val.npy   / y_val.npy")
    print("    X_test.npy  / y_test.npy")
    print("  ")
    print("  [seq]")
    print("    X_train.npy / y_train.npy")
    print("    X_val.npy   / y_val.npy")
    print("    X_test.npy  / y_test.npy")

    print("\n[NEXT STEP]")
    print("  1. baseline_models.py → RF / XGBoost 학습")
    print("  2. wgan_gp.py         → WGAN-GP로 Botnet 데이터 증강")
    print("  3. cnn_lstm.py        → CNN-LSTM 학습")
    print("  4. evaluate.py        → 성능 비교표 생성")


if __name__ == "__main__":
    main()