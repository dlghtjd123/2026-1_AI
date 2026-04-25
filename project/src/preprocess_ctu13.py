"""
preprocess_ctu13.py

목적:
- cicflowmeter로 생성한 CTU-13 raw CSV를 CIC-IDS2017과 동일한 형태로 변환
- 컬럼명 매핑 → 봇넷 IP 기반 레이블링 → split_host_groups → winflat/seq 저장

[사용 시나리오]
- scenario9만 사용
  이유: scenario1은 봇넷 호스트가 단 하루만 활동하여 봇넷 그룹이 1개뿐이다.
       train/val/test 분리 시 봇넷이 특정 split에만 편중되어
       val 기반 threshold 탐색이 불가능하다.
  scenario9는 봇넷 그룹이 12개로 정상적인 split이 가능하다.

봇넷 IP:
- scenario9: 147.32.84.165, 147.32.84.191, 147.32.84.192,
             147.32.84.193, 147.32.84.204, 147.32.84.205

[변경 이력]
- CTU NetFlow CSV(8 features) → CICFlowMeter CSV(77 features)로 전환
- preprocess_cicids.py와 동일한 ML_FEATURES 사용
- COMMON_FEATURES, 8개 제한 제거
- groupby 기준: (Source IP, 날짜) — CIC와 동일
- scenario1 제외
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
BASE_DIR  = _SRC_DIR.parent

RAW_DIR   = BASE_DIR / "data" / "raw" / "ctu-13"
SAVE_ROOT = BASE_DIR / "data" / "processed" / "ctu13"

SCENARIOS = {
    "scenario9": {
        "csv":        RAW_DIR / "scenario9_raw.csv",
        "botnet_ips": {
            "147.32.84.165", "147.32.84.191", "147.32.84.192",
            "147.32.84.193", "147.32.84.204", "147.32.84.205",
        },
    },
}

WINDOW_SIZE = 15
STEP_SIZE   = 5


# =========================================================
# cicflowmeter 컬럼명 → CIC-IDS2017 컬럼명 매핑
# =========================================================
COLUMN_MAP = {
    "src_ip":           "Source IP",
    "timestamp":        "Timestamp",
    "protocol":         "Protocol",
    "flow_duration":    "Flow Duration",
    "tot_fwd_pkts":     "Total Fwd Packets",
    "tot_bwd_pkts":     "Total Backward Packets",
    "totlen_fwd_pkts":  "Total Length of Fwd Packets",
    "totlen_bwd_pkts":  "Total Length of Bwd Packets",
    "fwd_pkt_len_max":  "Fwd Packet Length Max",
    "fwd_pkt_len_min":  "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std":  "Fwd Packet Length Std",
    "bwd_pkt_len_max":  "Bwd Packet Length Max",
    "bwd_pkt_len_min":  "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std":  "Bwd Packet Length Std",
    "flow_byts_s":      "Flow Bytes/s",
    "flow_pkts_s":      "Flow Packets/s",
    "flow_iat_mean":    "Flow IAT Mean",
    "flow_iat_std":     "Flow IAT Std",
    "flow_iat_max":     "Flow IAT Max",
    "flow_iat_min":     "Flow IAT Min",
    "fwd_iat_tot":      "Fwd IAT Total",
    "fwd_iat_mean":     "Fwd IAT Mean",
    "fwd_iat_std":      "Fwd IAT Std",
    "fwd_iat_max":      "Fwd IAT Max",
    "fwd_iat_min":      "Fwd IAT Min",
    "bwd_iat_tot":      "Bwd IAT Total",
    "bwd_iat_mean":     "Bwd IAT Mean",
    "bwd_iat_std":      "Bwd IAT Std",
    "bwd_iat_max":      "Bwd IAT Max",
    "bwd_iat_min":      "Bwd IAT Min",
    "fwd_psh_flags":    "Fwd PSH Flags",
    "bwd_psh_flags":    "Bwd PSH Flags",
    "fwd_urg_flags":    "Fwd URG Flags",
    "bwd_urg_flags":    "Bwd URG Flags",
    "fwd_header_len":   "Fwd Header Length",
    "bwd_header_len":   "Bwd Header Length",
    "fwd_pkts_s":       "Fwd Packets/s",
    "bwd_pkts_s":       "Bwd Packets/s",
    "pkt_len_min":      "Min Packet Length",
    "pkt_len_max":      "Max Packet Length",
    "pkt_len_mean":     "Packet Length Mean",
    "pkt_len_std":      "Packet Length Std",
    "pkt_len_var":      "Packet Length Variance",
    "fin_flag_cnt":     "FIN Flag Count",
    "syn_flag_cnt":     "SYN Flag Count",
    "rst_flag_cnt":     "RST Flag Count",
    "psh_flag_cnt":     "PSH Flag Count",
    "ack_flag_cnt":     "ACK Flag Count",
    "urg_flag_cnt":     "URG Flag Count",
    "cwr_flag_count":   "CWE Flag Count",
    "ece_flag_cnt":     "ECE Flag Count",
    "down_up_ratio":    "Down/Up Ratio",
    "pkt_size_avg":     "Average Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_byts_b_avg":   "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg":   "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg":   "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg":   "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts":"Init_Win_bytes_forward",
    "init_bwd_win_byts":"Init_Win_bytes_backward",
    "fwd_act_data_pkts":"act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean":      "Active Mean",
    "active_std":       "Active Std",
    "active_max":       "Active Max",
    "active_min":       "Active Min",
    "idle_mean":        "Idle Mean",
    "idle_std":         "Idle Std",
    "idle_max":         "Idle Max",
    "idle_min":         "Idle Min",
}

ML_FEATURES = [
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
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
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


# =========================================================
# 슬라이딩 윈도우 공통 함수
# =========================================================
def _build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
) -> tuple[list, list]:
    windows, ys = [], []
    n = len(features)
    if n < window_size:
        return windows, ys
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        windows.append(features[start:end])
        ys.append(1 if labels[start:end].sum() > 0 else 0)
    return windows, ys


# =========================================================
# 시나리오 처리
# =========================================================
def process_one_scenario(scenario_name: str, config: dict) -> None:
    csv_path   = config["csv"]
    botnet_ips = config["botnet_ips"]

    print(f"\n{'='*60}")
    print(f"[PROCESS] {scenario_name}")
    print(f"{'='*60}")

    save_dir    = SAVE_ROOT / scenario_name
    winflat_dir = save_dir / "winflat"
    seq_dir     = save_dir / "seq"

    for d in [winflat_dir, seq_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── CSV 로드 ─────────────────────────────────────────
    print(f"[LOAD] {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[LOAD] shape: {df.shape}")

    # ── 컬럼명 매핑 ──────────────────────────────────────
    df = df.rename(columns=COLUMN_MAP)
    print(f"[MAP] 컬럼명 CIC 형식으로 변환 완료")

    # ── Timestamp 파싱 ────────────────────────────────────
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed")
    before = len(df)
    df = df.dropna(subset=["Timestamp", "Source IP"]).reset_index(drop=True)
    print(f"[CLEAN] Timestamp/Source IP 결측 제거: {before - len(df):,}행")

    # ── 수치형 정제 ───────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # ── 레이블 생성 ───────────────────────────────────────
    df["Label_binary"] = df["Source IP"].apply(
        lambda ip: 1 if str(ip) in botnet_ips else 0
    )
    print(f"[LABEL] Botnet: {df['Label_binary'].sum():,} / 전체: {len(df):,}")
    print(f"[LABEL] Botnet 비율: {df['Label_binary'].mean():.4f}")

    # ── 시간 정렬 ─────────────────────────────────────────
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ── (Source IP, 날짜) 기준 split ─────────────────────
    df["_date"]      = df["Timestamp"].dt.date
    df["_group_key"] = df["Source IP"].astype(str) + "_" + df["_date"].astype(str)

    group_info = (
        df.groupby("_group_key")["Label_binary"]
        .sum().gt(0).astype(int).reset_index()
    )
    group_keys    = group_info["_group_key"].tolist()
    group_has_bot = group_info["Label_binary"].tolist()

    print(f"\n[SPLIT] 전체 그룹 수: {len(group_keys):,}")
    print(f"[SPLIT] Botnet 포함 그룹: {sum(group_has_bot):,}")

    keys_trainval, keys_test, _, _ = train_test_split(
        group_keys, group_has_bot,
        test_size=0.2, random_state=42, stratify=group_has_bot,
    )
    label_map       = dict(zip(group_keys, group_has_bot))
    labels_trainval = [label_map[k] for k in keys_trainval]
    keys_train, keys_val, _, _ = train_test_split(
        keys_trainval, labels_trainval,
        test_size=0.125, random_state=42, stratify=labels_trainval,
    )

    train_set = set(keys_train)
    val_set   = set(keys_val)
    test_set  = set(keys_test)

    df_train = df[df["_group_key"].isin(train_set)].drop(columns=["_date", "_group_key"]).reset_index(drop=True)
    df_val   = df[df["_group_key"].isin(val_set)].drop(columns=["_date", "_group_key"]).reset_index(drop=True)
    df_test  = df[df["_group_key"].isin(test_set)].drop(columns=["_date", "_group_key"]).reset_index(drop=True)

    print(f"[SPLIT] train: {len(df_train):,} / val: {len(df_val):,} / test: {len(df_test):,}")
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"  {name} Botnet 비율: {subset['Label_binary'].mean():.4f}")

    # ── window-flat / seq 생성 ────────────────────────────
    valid_cols = [c for c in ML_FEATURES if c in df.columns]
    missing    = [c for c in ML_FEATURES if c not in df.columns]
    if missing:
        print(f"[WARN] 없는 컬럼 (스킵): {missing}")

    def make_windows(subset: pd.DataFrame):
        subset = subset.copy()
        subset["_date"] = pd.to_datetime(subset["Timestamp"]).dt.date
        all_flat, all_seq, all_y = [], [], []
        skipped = 0
        for (src_ip, date), group in subset.groupby(["Source IP", "_date"]):
            group    = group.sort_values("Timestamp").reset_index(drop=True)
            features = group[valid_cols].values.astype(np.float32)
            labels   = group["Label_binary"].values
            wins, ys = _build_windows(features, labels, WINDOW_SIZE, STEP_SIZE)
            if not wins:
                skipped += 1
                continue
            all_seq.extend(wins)
            all_flat.extend([w.reshape(-1) for w in wins])
            all_y.extend(ys)
        return (
            np.array(all_seq,  dtype=np.float32),
            np.array(all_flat, dtype=np.float32),
            np.array(all_y,    dtype=np.int32),
            skipped,
        )

    print(f"\n[WINDOW] window_size={WINDOW_SIZE}, step_size={STEP_SIZE}")
    splits = {}
    for split_name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X_seq, X_flat, y, skipped = make_windows(subset)
        print(f"  {split_name}: seq {X_seq.shape} / flat {X_flat.shape} / Botnet {y.mean():.4f} / 스킵 {skipped}")
        splits[split_name] = (X_seq, X_flat, y)

    # ── seq scaler (train 기준 fit) ───────────────────────
    scaler  = StandardScaler()
    n, w, f = splits["train"][0].shape
    X_tr_scaled = scaler.fit_transform(
        splits["train"][0].reshape(-1, f)
    ).reshape(n, w, f).astype(np.float32)

    def scale_seq(X):
        n, w, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, w, f).astype(np.float32)

    X_va_scaled = scale_seq(splits["val"][0])
    X_te_scaled = scale_seq(splits["test"][0])

    joblib.dump(scaler, seq_dir / f"scaler_seq_w{WINDOW_SIZE}.pkl")
    print(f"[SCALER] seq scaler 저장: seq/scaler_seq_w{WINDOW_SIZE}.pkl")

    # ── 저장 ─────────────────────────────────────────────
    for split_name, (X_seq, X_flat, y) in splits.items():
        np.save(winflat_dir / f"X_{split_name}.npy", X_flat)
        np.save(winflat_dir / f"y_{split_name}.npy", y)

    np.save(seq_dir / "X_train.npy", X_tr_scaled)
    np.save(seq_dir / "X_val.npy",   X_va_scaled)
    np.save(seq_dir / "X_test.npy",  X_te_scaled)
    for split_name in ["train", "val", "test"]:
        np.save(seq_dir / f"y_{split_name}.npy", splits[split_name][2])

    # ── meta.json ─────────────────────────────────────────
    meta = {
        "scenario":         scenario_name,
        "source_csv":       str(csv_path),
        "botnet_ips":       list(botnet_ips),
        "window_size":      WINDOW_SIZE,
        "step_size":        STEP_SIZE,
        "num_features":     len(valid_cols),
        "feature_columns":  valid_cols,
        "groupby_key":      "(Source IP, date)",
        "flow_level": {
            "total":        int(len(df)),
            "botnet":       int(df["Label_binary"].sum()),
            "botnet_ratio": float(df["Label_binary"].mean()),
        },
        "splits": {
            s: {
                "winflat_shape": list(splits[s][1].shape),
                "seq_shape":     list(splits[s][0].shape),
                "botnet_ratio":  float(splits[s][2].mean()),
            }
            for s in ["train", "val", "test"]
        },
    }
    with open(save_dir / "meta.json", "w", encoding="utf-8") as f_:
        json.dump(meta, f_, indent=4, ensure_ascii=False)

    print(f"\n[SAVE] {save_dir}")
    print(f"[META] Botnet 비율 (flow): {meta['flow_level']['botnet_ratio']:.4f}")


# =========================================================
# 메인
# =========================================================
def main() -> None:
    print("=== CTU-13 Preprocessing Start ===")
    print(f"[PATH] RAW_DIR  : {RAW_DIR}")
    print(f"[PATH] SAVE_ROOT: {SAVE_ROOT}")
    print(f"[CONFIG] WINDOW_SIZE={WINDOW_SIZE}, STEP_SIZE={STEP_SIZE}")
    print(f"[CONFIG] ML_FEATURES: {len(ML_FEATURES)}개 (CIC-IDS2017과 동일)")
    print(f"[CONFIG] 사용 시나리오: scenario9 (scenario1 제외 — 봇넷 그룹 1개)")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    for scenario_name, config in SCENARIOS.items():
        if not config["csv"].exists():
            raise FileNotFoundError(f"CSV 파일이 없습니다: {config['csv']}")
        process_one_scenario(scenario_name, config)

    print("\n=== CTU-13 Preprocessing End ===")
    print("\n[저장된 파일 구조]")
    print("  data/processed/ctu13/")
    print("    scenario9/")
    print("      winflat/ X_train/val/test.npy  y_train/val/test.npy")
    print("      seq/     X_train/val/test.npy  y_train/val/test.npy  scaler_seq_w15.pkl")
    print("      meta.json")

    print("\n[NEXT STEP]")
    print("  evaluate.py 실행 → CTU 교차검증 (77 features)")


if __name__ == "__main__":
    main()