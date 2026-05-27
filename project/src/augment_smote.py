"""
augment_smote.py

CIC-IDS2017 Bot 클래스 SMOTE 증강 — K-fold 대응

증강 대상: trainval 데이터만 (test 불변)
저장 경로: data/processed/cicids2017_smote/
  flat/ X_trainval.npy  y_trainval.npy  (RF/XGB용)
  seq/  X_trainval.npy  y_trainval.npy  (CNN-LSTM/GRU용)
  test는 원본에서 복사
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE


# =========================================================
# 인자 파싱
# =========================================================
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--dataset",
    type=str,
    default="cicids2017",
    choices=["cicids2017", "cicids2018", "ctu13"],
    help="증강할 데이터셋 선택",
)
DATASET = _parser.parse_args().dataset


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent

SRC_ROOT  = _PROJECT / "data" / "processed" / DATASET
SAVE_ROOT = _PROJECT / "data" / "processed" / f"{DATASET}_smote"

SRC_FLAT  = SRC_ROOT / "flat"
SRC_SEQ   = SRC_ROOT / "seq"

SAVE_FLAT = SAVE_ROOT / "flat"
SAVE_SEQ  = SAVE_ROOT / "seq"


# =========================================================
# 설정
# =========================================================
RANDOM_STATE = 42
TARGET_RATIO = 0.1      # 증강 후 목표 Bot 비율


# =========================================================
# main
# =========================================================
def main() -> None:
    print("=" * 60)
    print("  SMOTE Augmentation — CIC-IDS2017 Bot 클래스")
    print("=" * 60)
    print(f"  SRC_ROOT  = {SRC_ROOT}")
    print(f"  SAVE_ROOT = {SAVE_ROOT}")
    print(f"  TARGET_RATIO (Bot 비율) = {TARGET_RATIO}")
    print("=" * 60)

    for d in [SAVE_FLAT, SAVE_SEQ]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 원본 trainval 로드 ────────────────────────────────
    X_trainval = np.load(SRC_FLAT / "X_trainval.npy")
    y_trainval = np.load(SRC_FLAT / "y_trainval.npy").astype(int)

    n_feat = X_trainval.shape[1]

    print(f"\n[LOAD] X_trainval: {X_trainval.shape}")
    print(f"[LOAD] Bot(1): {y_trainval.sum():,}  "
          f"Normal(0): {(y_trainval == 0).sum():,}  "
          f"Bot 비율: {y_trainval.mean():.4f}")

    # ── SMOTE 적용 ────────────────────────────────────────
    n_majority = (y_trainval == 0).sum()
    n_current  = y_trainval.sum()
    n_target   = int(n_majority * TARGET_RATIO / (1 - TARGET_RATIO))

    if n_target <= n_current:
        print(f"\n[INFO] 이미 목표 비율 달성 — SMOTE 불필요")
        return

    print(f"\n[SMOTE] 생성할 Bot 샘플: {n_target - n_current:,}개")
    print(f"[SMOTE] 증강 후 Bot: {n_target:,} / Normal: {n_majority:,}")

    smote = SMOTE(
        sampling_strategy={1: n_target},
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    X_aug, y_aug = smote.fit_resample(X_trainval, y_trainval)

    X_aug = X_aug.astype(np.float32)
    y_aug = y_aug.astype(np.int32)

    print(f"\n[SMOTE] 완료")
    print(f"  X_aug shape: {X_aug.shape}")
    print(f"  Bot(1): {y_aug.sum():,}  "
          f"Normal(0): {(y_aug == 0).sum():,}  "
          f"Bot 비율: {y_aug.mean():.4f}")

    # ── flat 저장 (RF/XGB) ────────────────────────────────
    np.save(SAVE_FLAT / "X_trainval.npy", X_aug)
    np.save(SAVE_FLAT / "y_trainval.npy", y_aug)
    print(f"\n[SAVE] flat/X_trainval.npy  shape={X_aug.shape}")

    # ── seq 저장 (CNN-LSTM/GRU): (n, 77, 1) ──────────────
    X_aug_seq = X_aug.reshape(-1, n_feat, 1)
    np.save(SAVE_SEQ / "X_trainval.npy", X_aug_seq)
    np.save(SAVE_SEQ / "y_trainval.npy", y_aug)
    print(f"[SAVE] seq/X_trainval.npy   shape={X_aug_seq.shape}")

    # ── test 원본에서 복사 ────────────────────────────────
    for d_src, d_dst in [(SRC_FLAT, SAVE_FLAT), (SRC_SEQ, SAVE_SEQ)]:
        for fname in ["X_test.npy", "y_test.npy"]:
            src = d_src / fname
            if src.exists():
                shutil.copy2(src, d_dst / fname)

    scaler_src = SRC_SEQ / "scaler_flow.pkl"
    if scaler_src.exists():
        shutil.copy2(scaler_src, SAVE_SEQ / "scaler_flow.pkl")

    print(f"\n[COPY] test 원본 복사 완료")

    # ── meta 저장 ─────────────────────────────────────────
    meta = {
        "method":            "SMOTE",
        "target_ratio":      TARGET_RATIO,
        "original_bot":      int(n_current),
        "augmented_bot":     int(y_aug.sum()),
        "generated_samples": int(n_target - n_current),
        "normal_count":      int((y_aug == 0).sum()),
        "total_trainval":    int(len(y_aug)),
        "bot_ratio_after":   float(y_aug.mean()),
        "n_features":        n_feat,
        "k_neighbors":       5,
        "random_state":      RANDOM_STATE,
    }
    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"\n[DONE] {SAVE_ROOT}")
    print(f"\n[다음 단계]")
    print(f"  python train_rf.py --dataset {DATASET} --augment smote")


if __name__ == "__main__":
    main()
