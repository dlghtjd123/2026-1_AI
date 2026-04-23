import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(_SRC_DIR)

RAW_DIR   = os.path.join(BASE_DIR, "data", "raw", "cic-ids2017")
SAVE_DIR  = os.path.join(BASE_DIR, "data", "processed", "cicids")
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 내부 네트워크 대역 정의
# =========================================================
INTERNAL_IP_PREFIXES = (
    "192.168.",
    "172.16.",
)


# =========================================================
# 공통 feature 목록 (8개) — CTU-13 교차검증용
# ---------------------------------------------------------
# CTU-13에서도 생성 가능한 feature만 추린 목록.
# preprocess_ctu13.py의 COMMON_FEATURES와 반드시 동일해야 한다.
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
# 모델 입력 feature 목록 (77개) — CIC 단독 평가용
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
# Task 정의:
#   Botnet(감염 호스트의 C&C 통신) = 1
#   나머지 모든 트래픽 (BENIGN 포함, DDoS·PortScan 등 타 공격도 포함) = 0
#
# 목적: 봇넷 탐지율(Recall for Botnet class)을 주 지표로 삼는다.
# 따라서 non-botnet 공격을 0으로 두는 것은 의도적 설계다.
# 논문에서 반드시 "Botnet vs. all other traffic" 으로 명시할 것.
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

_label_counter: Counter = Counter()


# =========================================================
# CSV 로드
# =========================================================

def load_all_csv(raw_dir: str) -> pd.DataFrame:
    """
    디렉터리 내 모든 CSV 파일을 읽어 하나의 DataFrame으로 병합한다.

    utf-8 → cp1252 → latin-1 순으로 인코딩을 순차 시도한다.
    UnicodeDecodeError 외의 오류는 재시도해도 해결되지 않으므로 즉시 중단한다.
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
    """모든 컬럼명의 앞뒤 공백을 제거한다."""
    df.columns = df.columns.str.strip()
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    REQUIRED_BASE_COLS에 정의된 필수 컬럼이 모두 존재하는지 검증한다.

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
    """수치형 컬럼의 NaN / inf 개수를 출력한다."""
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
    """Label 컬럼의 클래스 분포를 출력한다."""
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
    2. 문자열 컬럼 공백 제거
    3. ML_FEATURES 수치형 강제 변환
    4. Timestamp → datetime 변환
    5. Timestamp 파싱 실패 행 제거 또는 forward-fill
    6. 수치형 inf → NaN 변환
    7. 수치형 NaN → 0 대체 (옵션)
    """
    before_rows = len(df)

    print("\n[CLEAN] 정제 시작")
    log_label_distribution(df, "before_cleaning")
    log_inf_nan_status(df, "before_cleaning")

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

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

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

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )
    parsed_na = df["Timestamp"].isna().sum()
    print(f"\n[CLEAN] Timestamp 전체 NaT 행 수: {parsed_na:,}")

    if drop_bad_timestamp:
        before_ts = len(df)
        df = df.dropna(subset=["Timestamp"])
        print(f"[CLEAN] Timestamp NaT 제거 행 수: {before_ts - len(df):,}")
    else:
        df["Timestamp"] = df["Timestamp"].ffill()
        print(f"[CLEAN] Timestamp ffill 후 남은 NaT 행 수: {df['Timestamp'].isna().sum():,}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_before = np.isinf(df[numeric_cols].to_numpy()).sum() if len(numeric_cols) > 0 else 0
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"[CLEAN] inf → NaN 변환 개수: {inf_before:,}")
    print(f"[CLEAN] 현재 수치형 전체 NaN 개수: {df[numeric_cols].isna().sum().sum():,}")
    log_inf_nan_status(df, "after_inf_to_nan")

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

    1 (Botnet)   : BOTNET_LABELS에 속하는 라벨
    0 (Non-Bot)  : KNOWN_NON_BOT_LABELS에 속하는 라벨 (BENIGN + 타 공격 포함)

    strict=True이면 두 집합 모두에 없는 unknown 라벨에서 ValueError 발생.
    """
    label = str(label).strip().lower()

    if label in BOTNET_LABELS:
        return 1
    if label in KNOWN_NON_BOT_LABELS:
        return 0

    _label_counter[label] += 1
    if strict:
        raise ValueError(
            f"알 수 없는 라벨 값: '{label}'\n"
            "BOTNET_LABELS 또는 KNOWN_NON_BOT_LABELS에 추가하거나 strict=False로 실행하세요."
        )
    return 0


def create_binary_label(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Label 컬럼을 기반으로 이진 라벨 컬럼(Label_binary)을 생성한다."""
    _label_counter.clear()

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
# Protocol 정규화
# ---------------------------------------------------------
# [수정] 함수명을 encode_protocol → normalize_protocol_column으로 변경.
# Protocol은 IANA 프로토콜 번호(TCP=6, UDP=17 등) 자체가 의미 있는 수치이므로
# LabelEncoder 없이 원값을 int32로 보존한다.
# 이전 버전에서 LabelEncoder를 반환 인터페이스로 유지하면서 항상 None을 반환하던
# 혼란스러운 구조를 제거하였다.
# =========================================================

def normalize_protocol_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Protocol 컬럼을 int32로 정규화한다.

    IANA 프로토콜 번호(TCP=6, UDP=17 등)를 그대로 사용한다.
    숫자로 변환할 수 없는 값은 -1로 채운다.
    """
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")
    print(f"[PROTOCOL] 숫자 변환 실패 수: {df['Protocol'].isna().sum():,}")
    df["Protocol"] = df["Protocol"].fillna(-1).astype(np.int32)
    print("[PROTOCOL] unique values:")
    print(sorted(df["Protocol"].unique().tolist())[:20])
    return df


# =========================================================
# 시간 정렬
# =========================================================

def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamp 오름차순으로 전체 데이터를 정렬한다."""
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
# host 그룹 단위 train / val / test split
# ---------------------------------------------------------
# [수정] 외부 IP 처리 관련 주석 및 미구현 코드 제거.
#
# sequence 기반 모델(CNN-LSTM)과 window-flat 모델(RF/XGBoost) 모두
# 내부 호스트의 시간 연속적 flow 흐름을 학습 대상으로 한다.
# 외부 IP flow는 host 단위 시간 연속성을 보장하기 어렵고,
# CTU-13 외부 검증과의 일관성을 위해서도 내부 IP만 사용한다.
# =========================================================

def split_host_groups(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    내부 IP 호스트를 (Source IP, 날짜) 단위로 train / val / test에 분리한다.

    split 방식:
        전체 그룹 → (train+val) / test → train / val
        stratify: 그룹 내 Botnet 존재 여부

    Parameters
    ----------
    df : pd.DataFrame
    val_ratio : float, default=0.1
    test_ratio : float, default=0.2
    random_state : int, default=42

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
        모두 내부 IP flow만 포함한다.
    """
    internal_mask = df["Source IP"].apply(is_internal_ip)
    df_internal   = df[internal_mask].copy()

    n_external = (~internal_mask).sum()
    print(f"[SPLIT] 외부 IP flow {n_external:,}개 제외 (내부 IP 전용 학습)")

    df_internal["_date"]      = df_internal["Timestamp"].dt.date
    df_internal["_group_key"] = (
        df_internal["Source IP"].astype(str) + "_" + df_internal["_date"].astype(str)
    )

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

    keys_trainval, keys_test, _, _ = train_test_split(
        group_keys, group_has_bot,
        test_size=test_ratio,
        random_state=random_state,
        stratify=group_has_bot,
    )

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

    drop_cols = ["_date", "_group_key"]

    df_train = (
        df_internal[df_internal["_group_key"].isin(keys_train_set)]
        .drop(columns=drop_cols)
        .reset_index(drop=True)
    )
    df_val = (
        df_internal[df_internal["_group_key"].isin(keys_val_set)]
        .drop(columns=drop_cols)
        .reset_index(drop=True)
    )
    df_test = (
        df_internal[df_internal["_group_key"].isin(keys_test_set)]
        .drop(columns=drop_cols)
        .reset_index(drop=True)
    )

    print(f"\n[SPLIT] 호스트 그룹 수: {len(group_keys):,}")
    print(f"[SPLIT] train 그룹: {len(keys_train):,} / val 그룹: {len(keys_val):,} / test 그룹: {len(keys_test):,}")
    print(f"[SPLIT] train flow: {len(df_train):,} / val flow: {len(df_val):,} / test flow: {len(df_test):,}")
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  {name} Botnet 비율: {subset['Label_binary'].mean():.4f}")

    return df_train, df_val, df_test


# =========================================================
# 슬라이딩 윈도우 공통 내부 함수
# ---------------------------------------------------------
# [추가] create_sequences와 create_window_flat_data가 중복으로 구현하던
# sliding window 핵심 로직을 하나의 내부 함수로 통합하였다.
#
# [윈도우 라벨링 전략 - 논문 명시 필요]
# window 내 Botnet flow가 1개 이상이면 window label = 1 로 처리한다.
# 이 전략은 Recall(봇넷 탐지율)을 높이는 방향으로 작동한다.
# 논문에서 WGAN-GP 증강 효과를 분석할 때 이 라벨링 방식이
# 기저 탐지율에 미치는 영향을 별도로 서술해야 한다.
# =========================================================

def _build_windows_for_group(
    features: np.ndarray,
    label_vals: np.ndarray,
    window_size: int,
    step_size: int,
) -> tuple[list[np.ndarray], list[int]]:
    """
    단일 그룹(host, date)의 flow 배열로부터 sliding window를 생성한다.

    flow 수가 window_size 미만이면 빈 리스트를 반환한다.

    Parameters
    ----------
    features : np.ndarray, shape (n_flows, n_features)
    label_vals : np.ndarray, shape (n_flows,)
    window_size : int
    step_size : int

    Returns
    -------
    windows : list of np.ndarray, shape (window_size, n_features) each
    labels  : list of int (0 or 1)
    """
    windows: list[np.ndarray] = []
    labels:  list[int]        = []
    n_flows = len(features)

    if n_flows < window_size:
        return windows, labels

    for start in range(0, n_flows - window_size + 1, step_size):
        end = start + window_size
        windows.append(features[start:end])
        labels.append(1 if label_vals[start:end].sum() > 0 else 0)

    return windows, labels


# =========================================================
# flat 데이터 생성 (RF / XGBoost - flow 단위)
# =========================================================

def create_flat_data(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    RF / XGBoost 학습용 flow 단위 2D feature 배열을 생성한다.

    트리 계열 모델은 feature scaling이 필요 없으므로 raw 값을 그대로 반환한다.
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
# ---------------------------------------------------------
# [수정] _build_windows_for_group 공통 함수를 사용하도록 리팩터링.
# groupby 기준: (Source IP, 날짜)
# =========================================================

def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 15,
    step_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CNN-LSTM 학습용 3D sequence 배열을 생성한다.

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

    all_windows: list[np.ndarray] = []
    all_labels:  list[int]        = []
    skipped = 0

    for (src_ip, date), group in df_internal.groupby(["Source IP", "_date"]):
        group    = group.sort_values("Timestamp").reset_index(drop=True)
        features = group[valid_cols].values
        labels   = group["Label_binary"].values

        windows, labels_w = _build_windows_for_group(features, labels, window_size, step_size)

        if not windows:
            skipped += 1
            continue

        all_windows.extend(windows)
        all_labels.extend(labels_w)

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels,  dtype=np.int32)

    print(f"[SEQ] window={window_size}, step={step_size}  생성: {len(all_windows):,}  스킵: {skipped}")
    print(f"[SEQ] X shape: {X.shape}  Botnet 비율: {y.mean():.4f}  ({y.sum():,}/{len(y):,})")

    return X, y


# =========================================================
# window-flat 데이터 생성 (RF / XGBoost 공정 비교용)
# ---------------------------------------------------------
# [수정] _build_windows_for_group 공통 함수를 사용하도록 리팩터링.
# =========================================================

def create_window_flat_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 15,
    step_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RF / XGBoost용 window-flat 데이터를 생성한다.

    CNN-LSTM용 sequence와 동일한 기준으로 window를 만들되,
    각 window를 1차원 벡터로 펼쳐서(flatten) 반환한다.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    window_size : int, default=15
    step_size : int, default=5

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size * n_features), dtype float32
    y : np.ndarray, shape (n_windows,), dtype int32
    """
    valid_cols   = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"[WARN] window-flat 생성에서 없는 컬럼 (스킵): {missing_cols}")

    internal_mask = df["Source IP"].apply(is_internal_ip)
    df_internal   = df[internal_mask].copy()
    df_internal["_date"] = df_internal["Timestamp"].dt.date

    print(
        f"\n[WIN-FLAT] 내부 IP flow 수: {len(df_internal):,} / "
        f"종류: {df_internal['Source IP'].nunique()}개"
    )

    all_flat:   list[np.ndarray] = []
    all_labels: list[int]        = []
    skipped = 0

    for (src_ip, date), group in df_internal.groupby(["Source IP", "_date"]):
        group    = group.sort_values("Timestamp").reset_index(drop=True)
        features = group[valid_cols].values
        labels   = group["Label_binary"].values

        windows, labels_w = _build_windows_for_group(features, labels, window_size, step_size)

        if not windows:
            skipped += 1
            continue

        all_flat.extend([w.reshape(-1) for w in windows])
        all_labels.extend(labels_w)

    X = np.array(all_flat,   dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"[WIN-FLAT] window_size={window_size}, step_size={step_size}")
    print(f"[WIN-FLAT] 생성된 window 수: {len(all_flat):,}")
    print(f"[WIN-FLAT] flow 부족으로 스킵된 (src_ip, date) 수: {skipped}")
    print(f"[WIN-FLAT] X shape: {X.shape}")
    print(f"[WIN-FLAT] y shape: {y.shape}")
    print(f"[WIN-FLAT] Botnet 비율: {y.mean():.4f}")
    print(f"[WIN-FLAT] Botnet 수: {y.sum():,} / 전체: {len(y):,}")

    return X, y


# =========================================================
# scaler fit + 저장 (CNN-LSTM 전용)
# ---------------------------------------------------------
# [수정] save_dir은 main()에서 SEQ_DIR을 전달한다.
#   이전 버전은 SAVE_DIR 루트에 저장하여 winflat용 scaler와 혼용될 위험이 있었다.
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

    3D sequence 배열 (n, window, features)은 (n*window, features)로 reshape한 뒤
    fit/transform하고 원래 shape으로 복원한다.
    fit된 scaler는 save_dir에 pkl로 저장한다.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
    save_dir : str
        scaler를 저장할 디렉터리. main()에서 SEQ_DIR을 전달할 것.
    name : str

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

    scaler_path = os.path.join(save_dir, f"scaler_{name}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[SCALER] {scaler_path} 저장 완료")

    return X_tr.astype(np.float32), X_va.astype(np.float32), X_te.astype(np.float32)


# =========================================================
# 클래스 불균형 통계 저장
# ---------------------------------------------------------
# [추가] WGAN-GP 증강 정당화를 위한 split별 Botnet 비율을 JSON으로 저장한다.
# 증강 전후 탐지율 비교 표를 만들 때 이 파일이 baseline 불균형 수치가 된다.
# =========================================================

def save_split_stats(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    y_win_train: np.ndarray,
    y_win_val: np.ndarray,
    y_win_test: np.ndarray,
    y_seq_train: np.ndarray,
    y_seq_val: np.ndarray,
    y_seq_test: np.ndarray,
    save_dir: str,
) -> None:
    """
    split별 flow 단위 및 window 단위 클래스 불균형 통계를 JSON으로 저장한다.

    저장 내용:
    - flow_level: split별 전체 flow 수, Botnet flow 수, Botnet 비율
    - window_flat: split별 전체 window 수, Botnet window 수, Botnet 비율
    - seq: split별 전체 sequence 수, Botnet sequence 수, Botnet 비율

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        split_host_groups()로 분리된 subset DataFrame
    y_win_* : np.ndarray
        create_window_flat_data()가 반환한 window 단위 라벨
    y_seq_* : np.ndarray
        create_sequences()가 반환한 sequence 단위 라벨
    save_dir : str
    """
    def _stats(df: pd.DataFrame, y_win: np.ndarray, y_seq: np.ndarray) -> dict:
        return {
            "flow_level": {
                "total":        int(len(df)),
                "botnet":       int(df["Label_binary"].sum()),
                "botnet_ratio": float(df["Label_binary"].mean()),
            },
            "window_flat": {
                "total":        int(len(y_win)),
                "botnet":       int(y_win.sum()),
                "botnet_ratio": float(y_win.mean()),
            },
            "seq": {
                "total":        int(len(y_seq)),
                "botnet":       int(y_seq.sum()),
                "botnet_ratio": float(y_seq.mean()),
            },
        }

    stats = {
        "train": _stats(df_train, y_win_train, y_seq_train),
        "val":   _stats(df_val,   y_win_val,   y_seq_val),
        "test":  _stats(df_test,  y_win_test,  y_seq_test),
    }

    out_path = os.path.join(save_dir, "split_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print(f"\n[STATS] 클래스 불균형 통계 저장 완료: {out_path}")
    for split_name, s in stats.items():
        print(
            f"  {split_name}: flow Botnet {s['flow_level']['botnet_ratio']:.4f} | "
            f"window Botnet {s['window_flat']['botnet_ratio']:.4f} | "
            f"seq Botnet {s['seq']['botnet_ratio']:.4f}"
        )


# =========================================================
# 시각 자료 저장
# =========================================================

def analyze_flow_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """Source IP 기준 flow 수 분포를 분석하고 시각화하여 저장한다."""
    internal_mask   = df["Source IP"].apply(is_internal_ip)
    internal_counts = df[internal_mask].groupby("Source IP").size().rename("flow_count")

    print("\n[FLOW DIST] 내부 src_ip 당 flow 수 분포:")
    stats = internal_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
    print(stats)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(internal_counts, bins=50, kde=True, ax=axes[0], log_scale=True, color="skyblue")
    axes[0].set_title("Distribution of Flow Counts per Internal IP (Log Scale)")
    axes[0].set_xlabel("Number of Flows (Log Scale)")
    axes[0].set_ylabel("Frequency (Number of IPs)")

    median_val = internal_counts.median()
    axes[0].axvline(median_val, color="red", linestyle="--", label=f"Median: {median_val:.1f}")
    axes[0].legend()

    sns.boxplot(x=internal_counts, ax=axes[1], color="lightgreen")
    axes[1].set_xscale("log")
    axes[1].set_title("Boxplot of Flow Counts per Internal IP (Log Scale)")
    axes[1].set_xlabel("Number of Flows (Log Scale)")

    plt.tight_layout()

    plot_path = os.path.join(save_dir, "flow_distribution.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[FLOW DIST] 시각화 결과 저장 완료: {plot_path}")

    if median_val < 5:      recommended = 3
    elif median_val < 20:   recommended = 5
    elif median_val < 50:   recommended = 10
    else:                   recommended = 15

    print(f"→ 추천 window_size: {recommended} (Median 기반)")


# =========================================================
# 저장
# =========================================================

def save_outputs(df: pd.DataFrame, save_dir: str) -> None:
    """전처리 완료 DataFrame을 parquet 형식으로 저장한다."""
    save_cols = (
        ["Source IP", "Timestamp", "Label", "Label_binary"]
        + [col for col in ML_FEATURES if col in df.columns]
    )
    out_path = os.path.join(save_dir, "cicids2017_traffic.parquet")
    df[save_cols].to_parquet(out_path, index=False)
    print(f"[SAVE] 전처리 데이터 저장 완료: {out_path}")


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
    3.  기본 정제
    4.  이진 라벨 생성
    5.  Protocol 정규화 및 시간 정렬
    6.  전처리 결과 parquet 저장
    7.  flow 분포 분석
    8.  host 그룹 단위 train / val / test split
    9.  window-flat 데이터 저장 (RF/XGBoost)
    10. sequence 생성 → scaler fit → 저장 (CNN-LSTM)
    11. 클래스 불균형 통계 저장
    """
    print("=== CIC-IDS2017 TrafficLabeling Preprocessing Start ===")
    print(f"[PATH] RAW_DIR : {RAW_DIR}")
    print(f"[PATH] SAVE_DIR: {SAVE_DIR}")

    WINDOW_SIZE = 15
    STEP_SIZE   = 5

    # 디렉터리 구조
    # data/processed/cicids/
    #   ├── cicids2017_traffic.parquet
    #   ├── split_stats.json
    #   ├── flow_distribution.png
    #   ├── winflat/   X_train/val/test.npy  y_train/val/test.npy
    #   └── seq/       X_train/val/test.npy  y_train/val/test.npy  scaler_seq_w15.pkl

    WINFLAT_DIR        = os.path.join(SAVE_DIR, "winflat")
    SEQ_DIR            = os.path.join(SAVE_DIR, "seq")
    WINFLAT_COMMON_DIR = os.path.join(SAVE_DIR, "winflat_common")
    SEQ_COMMON_DIR     = os.path.join(SAVE_DIR, "seq_common")

    for d in [WINFLAT_DIR, SEQ_DIR, WINFLAT_COMMON_DIR, SEQ_COMMON_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"\n[CONFIG] WINDOW_SIZE: {WINDOW_SIZE} / STEP_SIZE: {STEP_SIZE}")
    print(f"[CONFIG] ML_FEATURES: {len(ML_FEATURES)}개 / COMMON_FEATURES: {len(COMMON_FEATURES)}개")

    # ── 공통 전처리 ──────────────────────────────────────────
    df = load_all_csv(RAW_DIR)
    df = normalize_column_names(df)
    validate_columns(df)

    df = basic_cleaning(df, fill_numeric_na_with_zero=True, drop_bad_timestamp=True)
    df = create_binary_label(df, strict=False)
    df = normalize_protocol_column(df)    # [수정] encode_protocol → normalize_protocol_column
    df = sort_by_time(df)

    preview_data(df)
    save_outputs(df, SAVE_DIR)
    analyze_flow_distribution(df, SAVE_DIR)

    # ── host 그룹 단위 split ─────────────────────────────────
    df_train, df_val, df_test = split_host_groups(df)

    # ── [ML_FEATURES 77개] window-flat 저장 ─────────────────
    print(f"\n[STEP] RF/XGBoost window-flat ({len(ML_FEATURES)}개 features)")
    y_win_splits = {}
    for split_name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X_win, y_win = create_window_flat_data(subset, ML_FEATURES, WINDOW_SIZE, STEP_SIZE)
        save_numpy(X_win, y_win, WINFLAT_DIR, split_name)
        y_win_splits[split_name] = y_win

    # ── [ML_FEATURES 77개] sequence + scaler 저장 ────────────
    seq_tag = f"seq_w{WINDOW_SIZE}"
    print(f"\n[STEP] CNN-LSTM sequence ({len(ML_FEATURES)}개 features)")

    X_tr, y_tr = create_sequences(df_train, ML_FEATURES, WINDOW_SIZE, STEP_SIZE)
    X_va, y_va = create_sequences(df_val,   ML_FEATURES, WINDOW_SIZE, STEP_SIZE)
    X_te, y_te = create_sequences(df_test,  ML_FEATURES, WINDOW_SIZE, STEP_SIZE)

    # [수정] scaler를 SEQ_DIR에 저장 (이전 버전은 SAVE_DIR 루트에 저장하여 혼용 위험이 있었음)
    X_tr, X_va, X_te = fit_and_save_scaler(X_tr, X_va, X_te, SEQ_DIR, seq_tag)

    save_numpy(X_tr, y_tr, SEQ_DIR, "train")
    save_numpy(X_va, y_va, SEQ_DIR, "val")
    save_numpy(X_te, y_te, SEQ_DIR, "test")

    y_seq_splits = {"train": y_tr, "val": y_va, "test": y_te}

    # ── [COMMON_FEATURES 8개] winflat 저장 (교차검증용) ──────
    print(f"\n[STEP] RF/XGBoost window-flat ({len(COMMON_FEATURES)}개 features, 교차검증용)")
    for split_name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X_win_c, y_win_c = create_window_flat_data(subset, COMMON_FEATURES, WINDOW_SIZE, STEP_SIZE)
        save_numpy(X_win_c, y_win_c, WINFLAT_COMMON_DIR, split_name)

    # ── [COMMON_FEATURES 8개] sequence + scaler 저장 (교차검증용)
    common_seq_tag = f"seq_common_w{WINDOW_SIZE}"
    print(f"\n[STEP] CNN-LSTM sequence ({len(COMMON_FEATURES)}개 features, 교차검증용)")

    X_tr_c, y_tr_c = create_sequences(df_train, COMMON_FEATURES, WINDOW_SIZE, STEP_SIZE)
    X_va_c, y_va_c = create_sequences(df_val,   COMMON_FEATURES, WINDOW_SIZE, STEP_SIZE)
    X_te_c, y_te_c = create_sequences(df_test,  COMMON_FEATURES, WINDOW_SIZE, STEP_SIZE)

    X_tr_c, X_va_c, X_te_c = fit_and_save_scaler(
        X_tr_c, X_va_c, X_te_c, SEQ_COMMON_DIR, common_seq_tag
    )

    save_numpy(X_tr_c, y_tr_c, SEQ_COMMON_DIR, "train")
    save_numpy(X_va_c, y_va_c, SEQ_COMMON_DIR, "val")
    save_numpy(X_te_c, y_te_c, SEQ_COMMON_DIR, "test")

    # ── 클래스 불균형 통계 저장 ──────────────────────────────
    save_split_stats(
        df_train, df_val, df_test,
        y_win_splits["train"], y_win_splits["val"], y_win_splits["test"],
        y_seq_splits["train"], y_seq_splits["val"], y_seq_splits["test"],
        SAVE_DIR,
    )

    print("\n[저장된 파일 구조]")
    print("  data/processed/cicids/")
    print("    cicids2017_traffic.parquet")
    print("    split_stats.json")
    print("    flow_distribution.png")
    print("    winflat/         X_train/val/test.npy  ← 77 features (CIC 단독 평가)")
    print("    seq/             X_train/val/test.npy  ← 77 features (CIC 단독 평가)")
    print("    winflat_common/  X_train/val/test.npy  ← 8 features  (CTU 교차검증)")
    print("    seq_common/      X_train/val/test.npy  ← 8 features  (CTU 교차검증)")

    print("\n[NEXT STEP]")
    print("  ① train_rf/xgb/cnn_lstm.py  MODE=full   → CIC 단독 평가 모델 학습")
    print("  ② train_rf/xgb/cnn_lstm.py  MODE=common → CTU 교차검증 모델 학습")
    print("  ③ preprocess_ctu13.py                   → CTU-13 전처리")
    print("  ④ evaluate.py                           → 성능 비교표 생성")


if __name__ == "__main__":
    main()