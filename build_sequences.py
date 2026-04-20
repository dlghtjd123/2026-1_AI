# build_sequences.py
# 목적:
# - 공통 row parquet(CIC / CTU)를 읽어 동일 규칙으로 window 생성
# - CIC: train / val / test split 후 seq / winflat / sample_ids 저장
# - CTU: scenario별 external validation용 seq / winflat / sample_ids 저장
# - seq scaler는 CIC train에만 fit하고 나머지는 transform만 적용
#
# 출력 예시:
# data/processed/common_build/
#   ├─ scaler_seq_w15.pkl
#   ├─ cic/
#   │   ├─ seq/
#   │   │   ├─ X_train.npy
#   │   │   ├─ y_train.npy
#   │   │   ├─ sample_ids_train.npy
#   │   │   ├─ X_val.npy
#   │   │   ├─ y_val.npy
#   │   │   ├─ sample_ids_val.npy
#   │   │   ├─ X_test.npy
#   │   │   ├─ y_test.npy
#   │   │   └─ sample_ids_test.npy
#   │   ├─ winflat/
#   │   │   ├─ X_train.npy
#   │   │   ├─ y_train.npy
#   │   │   ├─ sample_ids_train.npy
#   │   │   ├─ X_val.npy
#   │   │   ├─ y_val.npy
#   │   │   ├─ sample_ids_val.npy
#   │   │   ├─ X_test.npy
#   │   │   ├─ y_test.npy
#   │   │   └─ sample_ids_test.npy
#   │   └─ meta.json
#   └─ ctu13/
#       ├─ scenario1/
#       │   ├─ seq/
#       │   │   ├─ X.npy
#       │   │   ├─ y.npy
#       │   │   └─ sample_ids.npy
#       │   ├─ winflat/
#       │   │   ├─ X.npy
#       │   │   ├─ y.npy
#       │   │   └─ sample_ids.npy
#       │   └─ meta.json
#       └─ scenario9/
#           └─ ...

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. 경로 / 설정
# =========================================================

def resolve_project_root() -> Path:
    """
    현재 파일 위치를 기준으로 프로젝트 루트를 유추한다.
    - build_sequences.py가 프로젝트 루트에 있으면 그 위치 사용
    - src/ 아래에 있으면 상위 디렉터리 사용
    """
    here = Path(__file__).resolve().parent
    if (here / "data").exists():
        return here
    if (here.parent / "data").exists():
        return here.parent
    return here


BASE_DIR = resolve_project_root()

CIC_COMMON_PATH = BASE_DIR / "data" / "processed" / "cic_common.parquet"
CTU_COMMON_DIR = BASE_DIR / "data" / "processed" / "ctu13"
BUILD_ROOT = BASE_DIR / "data" / "processed" / "common_build"

CTU_SCENARIOS = [
    "scenario1_common.parquet",
    "scenario9_common.parquet",
]

WINDOW_SIZE = 15
STEP_SIZE = 5

VAL_RATIO = 0.1
TEST_RATIO = 0.2
RANDOM_STATE = 42

# group_id 기본 기준: Source IP + Destination IP + Protocol
# 날짜까지 강제로 끊고 싶으면 True로 바꾸면 된다.
APPEND_DATE_TO_GROUP = False

# seq scaler는 CIC train에만 fit
SAVE_SEQ_SCALER = True
SEQ_SCALER_FILENAME = f"scaler_seq_w{WINDOW_SIZE}.pkl"

# 공통 feature 9개
FEATURE_COLS = [
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

REQUIRED_BASE_COLS = [
    "Timestamp",
    "Source IP",
    "Destination IP",
    "Label_raw",
    "Label_binary",
]


# =========================================================
# 2. 유틸
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_common_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    df = pd.read_parquet(path)

    required = REQUIRED_BASE_COLS + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"공통 parquet 컬럼 누락: {missing}\npath={path}")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Timestamp", "Source IP", "Destination IP"]).reset_index(drop=True)
    if before != len(df):
        print(f"[INFO] {path.name}: Timestamp/주소 결측 row 제거 {before - len(df):,}개")

    # 수치형 안정화
    for col in FEATURE_COLS + ["Label_binary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["Label_binary"] = df["Label_binary"].fillna(0).astype(np.int64)

    return df


def build_group_id(df: pd.DataFrame, append_date: bool = False) -> pd.Series:
    base = (
        df["Source IP"].astype(str).str.strip() + "|" +
        df["Destination IP"].astype(str).str.strip() + "|" +
        df["Protocol"].astype(str).str.strip()
    )

    if append_date:
        dates = pd.to_datetime(df["Timestamp"], errors="coerce").dt.date.astype(str)
        base = base + "|" + dates

    return base


def can_stratify(labels: list[int] | np.ndarray) -> bool:
    """
    train_test_split(stratify=...)가 가능한 최소 조건 검사
    """
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 2:
        return False
    return counts.min() >= 2


def split_cic_groups(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    append_date_to_group: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CIC 공통 row 데이터에서 group 단위로 train / val / test 분할

    주의:
    - split은 반드시 window 생성 전에 수행
    - 같은 group에서 나온 window가 train/test에 동시에 들어가면 leakage 발생
    """
    df = df.copy()
    df["group_id"] = build_group_id(df, append_date=append_date_to_group)

    # 그룹별 공격 포함 여부
    group_info = (
        df.groupby("group_id", as_index=False)["Label_binary"]
        .max()
        .rename(columns={"Label_binary": "has_attack"})
    )

    group_ids = group_info["group_id"].tolist()
    group_labels = group_info["has_attack"].astype(int).tolist()

    print(f"[SPLIT] total groups: {len(group_ids):,}")
    print("[SPLIT] group label distribution:")
    print(group_info["has_attack"].value_counts().sort_index())

    # 1차: train+val / test
    stratify_first: Optional[list[int]] = group_labels if can_stratify(group_labels) else None
    if stratify_first is None:
        print("[WARN] 1차 group split에서 stratify를 사용할 수 없어 random split으로 진행합니다.")

    groups_trainval, groups_test = train_test_split(
        group_ids,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify_first,
    )

    # 2차: train / val
    group_label_map = dict(zip(group_ids, group_labels))
    trainval_labels = [group_label_map[g] for g in groups_trainval]

    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    stratify_second: Optional[list[int]] = trainval_labels if can_stratify(trainval_labels) else None
    if stratify_second is None:
        print("[WARN] 2차 group split에서 stratify를 사용할 수 없어 random split으로 진행합니다.")

    groups_train, groups_val = train_test_split(
        groups_trainval,
        test_size=adjusted_val_ratio,
        random_state=random_state,
        stratify=stratify_second,
    )

    train_set = set(groups_train)
    val_set = set(groups_val)
    test_set = set(groups_test)

    df_train = df[df["group_id"].isin(train_set)].copy().reset_index(drop=True)
    df_val = df[df["group_id"].isin(val_set)].copy().reset_index(drop=True)
    df_test = df[df["group_id"].isin(test_set)].copy().reset_index(drop=True)

    print(f"[SPLIT] train groups: {len(train_set):,}")
    print(f"[SPLIT] val groups  : {len(val_set):,}")
    print(f"[SPLIT] test groups : {len(test_set):,}")
    print(f"[SPLIT] train rows  : {len(df_train):,}")
    print(f"[SPLIT] val rows    : {len(df_val):,}")
    print(f"[SPLIT] test rows   : {len(df_test):,}")

    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        ratio = subset["Label_binary"].mean() if len(subset) > 0 else 0.0
        print(f"  {name} attack row ratio: {ratio:.4f}")

    return df_train, df_val, df_test


# =========================================================
# 3. window 생성
# =========================================================

def make_windows_for_group(
    group_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    step_size: int,
    dataset_name: str,
    split_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[str]]:
    """
    한 group에 대해 seq / winflat / y / sample_ids 생성
    """
    X_seq_list: list[np.ndarray] = []
    X_flat_list: list[np.ndarray] = []
    y_list: list[int] = []
    sample_id_list: list[str] = []

    group_df = group_df.sort_values("Timestamp").reset_index(drop=True)

    values = group_df[feature_cols].to_numpy(dtype=np.float32)
    labels = group_df["Label_binary"].to_numpy(dtype=np.int64)
    row_ids = group_df["row_id"].astype(str).to_numpy()
    group_id = str(group_df["group_id"].iloc[0])

    n = len(group_df)
    if n < window_size:
        return X_seq_list, X_flat_list, y_list, sample_id_list

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size

        x_seq = values[start:end]         # (window, feature)
        x_flat = x_seq.reshape(-1)        # (window*feature,)
        y = int(labels[start:end].max())  # window 내 하나라도 attack이면 1

        sample_id = (
            f"{dataset_name}|{split_name}|{group_id}|"
            f"{row_ids[start]}|{row_ids[end - 1]}"
        )

        X_seq_list.append(x_seq)
        X_flat_list.append(x_flat)
        y_list.append(y)
        sample_id_list.append(sample_id)

    return X_seq_list, X_flat_list, y_list, sample_id_list


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    step_size: int,
    dataset_name: str,
    split_name: str,
    append_date_to_group: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    DataFrame 전체에 대해 group_id 기준으로 sliding window 생성
    """
    df = df.copy()
    df["group_id"] = build_group_id(df, append_date=append_date_to_group)
    df = df.sort_values(["group_id", "Timestamp"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    X_seq_all: list[np.ndarray] = []
    X_flat_all: list[np.ndarray] = []
    y_all: list[int] = []
    sample_ids_all: list[str] = []

    skipped_groups = 0

    for _, group_df in df.groupby("group_id", sort=False):
        if len(group_df) < window_size:
            skipped_groups += 1
            continue

        X_seq_list, X_flat_list, y_list, sample_id_list = make_windows_for_group(
            group_df=group_df,
            feature_cols=feature_cols,
            window_size=window_size,
            step_size=step_size,
            dataset_name=dataset_name,
            split_name=split_name,
        )

        X_seq_all.extend(X_seq_list)
        X_flat_all.extend(X_flat_list)
        y_all.extend(y_list)
        sample_ids_all.extend(sample_id_list)

    if len(X_seq_all) == 0:
        raise ValueError(
            f"{dataset_name}/{split_name}: 생성된 window가 없습니다. "
            f"WINDOW_SIZE={window_size}가 너무 크거나 group이 너무 짧을 수 있습니다."
        )

    X_seq = np.asarray(X_seq_all, dtype=np.float32)      # (N, W, F)
    X_flat = np.asarray(X_flat_all, dtype=np.float32)    # (N, W*F)
    y = np.asarray(y_all, dtype=np.int64)
    sample_ids = np.asarray(sample_ids_all, dtype=object)

    print(f"[WINDOW] {dataset_name}/{split_name}")
    print(f"  seq shape    : {X_seq.shape}")
    print(f"  winflat shape: {X_flat.shape}")
    print(f"  y shape      : {y.shape}")
    print(f"  attack win   : {int(y.sum()):,} / {len(y):,}")
    print(f"  skipped group: {skipped_groups:,}")

    return X_seq, X_flat, y, sample_ids


# =========================================================
# 4. scaler
# =========================================================

def fit_seq_scaler(X_train_seq: np.ndarray) -> StandardScaler:
    """
    seq scaler는 CIC train seq에만 fit
    """
    if X_train_seq.ndim != 3:
        raise ValueError(f"X_train_seq는 3차원이어야 합니다. 현재 shape={X_train_seq.shape}")

    _, _, n_features = X_train_seq.shape
    scaler = StandardScaler()
    scaler.fit(X_train_seq.reshape(-1, n_features))
    return scaler


def transform_seq_with_scaler(X_seq: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    if X_seq.ndim != 3:
        raise ValueError(f"X_seq는 3차원이어야 합니다. 현재 shape={X_seq.shape}")

    n_samples, seq_len, n_features = X_seq.shape
    X_2d = X_seq.reshape(-1, n_features)
    X_2d = scaler.transform(X_2d)
    X_out = X_2d.reshape(n_samples, seq_len, n_features)
    return X_out.astype(np.float32)


# =========================================================
# 5. 저장
# =========================================================

def save_cic_split_outputs(
    root_dir: Path,
    split_name: str,
    X_seq: np.ndarray,
    X_flat: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
) -> None:
    seq_dir = root_dir / "seq"
    winflat_dir = root_dir / "winflat"
    ensure_dir(seq_dir)
    ensure_dir(winflat_dir)

    np.save(seq_dir / f"X_{split_name}.npy", X_seq)
    np.save(seq_dir / f"y_{split_name}.npy", y)
    np.save(seq_dir / f"sample_ids_{split_name}.npy", sample_ids)

    np.save(winflat_dir / f"X_{split_name}.npy", X_flat)
    np.save(winflat_dir / f"y_{split_name}.npy", y)
    np.save(winflat_dir / f"sample_ids_{split_name}.npy", sample_ids)

    print(f"[SAVE] CIC {split_name} 저장 완료 -> {root_dir}")


def save_ctu_scenario_outputs(
    scenario_root: Path,
    X_seq: np.ndarray,
    X_flat: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
) -> None:
    seq_dir = scenario_root / "seq"
    winflat_dir = scenario_root / "winflat"
    ensure_dir(seq_dir)
    ensure_dir(winflat_dir)

    np.save(seq_dir / "X.npy", X_seq)
    np.save(seq_dir / "y.npy", y)
    np.save(seq_dir / "sample_ids.npy", sample_ids)

    np.save(winflat_dir / "X.npy", X_flat)
    np.save(winflat_dir / "y.npy", y)
    np.save(winflat_dir / "sample_ids.npy", sample_ids)

    print(f"[SAVE] CTU scenario 저장 완료 -> {scenario_root}")


def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =========================================================
# 6. CIC 처리
# =========================================================

def process_cic() -> tuple[StandardScaler, dict]:
    print("\n=== CIC build start ===")

    df_cic = load_common_parquet(CIC_COMMON_PATH)
    print(f"[LOAD] CIC rows={len(df_cic):,}")

    df_train, df_val, df_test = split_cic_groups(
        df=df_cic,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_STATE,
        append_date_to_group=APPEND_DATE_TO_GROUP,
    )

    X_tr_seq, X_tr_flat, y_tr, sid_tr = build_windows(
        df=df_train,
        feature_cols=FEATURE_COLS,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        dataset_name="cic",
        split_name="train",
        append_date_to_group=APPEND_DATE_TO_GROUP,
    )

    X_va_seq, X_va_flat, y_va, sid_va = build_windows(
        df=df_val,
        feature_cols=FEATURE_COLS,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        dataset_name="cic",
        split_name="val",
        append_date_to_group=APPEND_DATE_TO_GROUP,
    )

    X_te_seq, X_te_flat, y_te, sid_te = build_windows(
        df=df_test,
        feature_cols=FEATURE_COLS,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        dataset_name="cic",
        split_name="test",
        append_date_to_group=APPEND_DATE_TO_GROUP,
    )

    # seq scaler fit -> CIC train only
    seq_scaler = fit_seq_scaler(X_tr_seq)
    X_tr_seq_scaled = transform_seq_with_scaler(X_tr_seq, seq_scaler)
    X_va_seq_scaled = transform_seq_with_scaler(X_va_seq, seq_scaler)
    X_te_seq_scaled = transform_seq_with_scaler(X_te_seq, seq_scaler)

    ensure_dir(BUILD_ROOT)
    scaler_path = BUILD_ROOT / SEQ_SCALER_FILENAME
    if SAVE_SEQ_SCALER:
        joblib.dump(seq_scaler, scaler_path)
        print(f"[SAVE] seq scaler 저장 완료 -> {scaler_path}")

    cic_root = BUILD_ROOT / "cic"
    save_cic_split_outputs(cic_root, "train", X_tr_seq_scaled, X_tr_flat, y_tr, sid_tr)
    save_cic_split_outputs(cic_root, "val", X_va_seq_scaled, X_va_flat, y_va, sid_va)
    save_cic_split_outputs(cic_root, "test", X_te_seq_scaled, X_te_flat, y_te, sid_te)

    cic_meta = {
        "source": str(CIC_COMMON_PATH),
        "feature_columns": FEATURE_COLS,
        "num_features": len(FEATURE_COLS),
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "append_date_to_group": APPEND_DATE_TO_GROUP,
        "group_rule": "Source IP + Destination IP + Protocol"
                      + (" + date" if APPEND_DATE_TO_GROUP else ""),
        "seq_scaler_path": str(scaler_path) if SAVE_SEQ_SCALER else None,
        "splits": {
            "train": {
                "seq_shape": list(X_tr_seq_scaled.shape),
                "winflat_shape": list(X_tr_flat.shape),
                "num_windows": int(len(y_tr)),
                "num_attack_windows": int(y_tr.sum()),
            },
            "val": {
                "seq_shape": list(X_va_seq_scaled.shape),
                "winflat_shape": list(X_va_flat.shape),
                "num_windows": int(len(y_va)),
                "num_attack_windows": int(y_va.sum()),
            },
            "test": {
                "seq_shape": list(X_te_seq_scaled.shape),
                "winflat_shape": list(X_te_flat.shape),
                "num_windows": int(len(y_te)),
                "num_attack_windows": int(y_te.sum()),
            },
        },
    }
    save_json(cic_root / "meta.json", cic_meta)

    print("=== CIC build end ===")
    return seq_scaler, cic_meta


# =========================================================
# 7. CTU 처리
# =========================================================

def process_ctu(seq_scaler: StandardScaler) -> None:
    print("\n=== CTU build start ===")

    ctu_root = BUILD_ROOT / "ctu13"

    for filename in CTU_SCENARIOS:
        parquet_path = CTU_COMMON_DIR / filename
        scenario_name = Path(filename).stem.replace("_common", "")

        df_ctu = load_common_parquet(parquet_path)
        print(f"\n[LOAD] {scenario_name} rows={len(df_ctu):,}")

        X_seq, X_flat, y, sample_ids = build_windows(
            df=df_ctu,
            feature_cols=FEATURE_COLS,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            dataset_name="ctu13",
            split_name=scenario_name,
            append_date_to_group=APPEND_DATE_TO_GROUP,
        )

        X_seq_scaled = transform_seq_with_scaler(X_seq, seq_scaler)

        scenario_root = ctu_root / scenario_name
        save_ctu_scenario_outputs(scenario_root, X_seq_scaled, X_flat, y, sample_ids)

        meta = {
            "source": str(parquet_path),
            "scenario": scenario_name,
            "feature_columns": FEATURE_COLS,
            "num_features": len(FEATURE_COLS),
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "append_date_to_group": APPEND_DATE_TO_GROUP,
            "group_rule": "Source IP + Destination IP + Protocol"
                          + (" + date" if APPEND_DATE_TO_GROUP else ""),
            "seq_scaler_path": str(BUILD_ROOT / SEQ_SCALER_FILENAME),
            "seq_shape": list(X_seq_scaled.shape),
            "winflat_shape": list(X_flat.shape),
            "num_windows": int(len(y)),
            "num_attack_windows": int(y.sum()),
        }
        save_json(scenario_root / "meta.json", meta)

    print("\n=== CTU build end ===")


# =========================================================
# 8. 메인
# =========================================================

def main() -> None:
    print("=== build_sequences.py start ===")
    print(f"[BASE_DIR] {BASE_DIR}")
    print(f"[CIC_COMMON] {CIC_COMMON_PATH}")
    print(f"[CTU_COMMON_DIR] {CTU_COMMON_DIR}")
    print(f"[BUILD_ROOT] {BUILD_ROOT}")
    print(f"[CONFIG] WINDOW_SIZE={WINDOW_SIZE}, STEP_SIZE={STEP_SIZE}")
    print(f"[CONFIG] VAL_RATIO={VAL_RATIO}, TEST_RATIO={TEST_RATIO}")
    print(f"[CONFIG] APPEND_DATE_TO_GROUP={APPEND_DATE_TO_GROUP}")
    print(f"[CONFIG] FEATURES={FEATURE_COLS}")

    ensure_dir(BUILD_ROOT)

    seq_scaler, _ = process_cic()
    process_ctu(seq_scaler)

    print("\n=== build_sequences.py end ===")
    print("[NEXT]")
    print("1. RF / XGBoost는 common_build/cic/winflat 사용")
    print("2. CNN-LSTM은 common_build/cic/seq 사용")
    print("3. CTU external validation은 common_build/ctu13/* 사용")


if __name__ == "__main__":
    main()