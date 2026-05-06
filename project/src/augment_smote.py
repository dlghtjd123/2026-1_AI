"""
augment_smote.py

CIC-IDS2017 Bot 클래스 SMOTE 증강

증강 대상: train 데이터만 (val/test 불변)
저장 경로: data/processed/cicids2017_smote/
  flat/ X_train.npy  y_train.npy  (RF/XGB용)
  seq/  X_train.npy  y_train.npy  (CNN-LSTM/GRU용)
  val, test는 원본에서 복사

이후: train_*.py의 DATA_DIR를
      cicids2017 → cicids2017_smote 로 변경 후 재학습
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent

SRC_ROOT  = _PROJECT / "data" / "processed" / "cicids2017"
SAVE_ROOT = _PROJECT / "data" / "processed" / "cicids2017_smote"

SRC_FLAT  = SRC_ROOT / "flat"
SRC_SEQ   = SRC_ROOT / "seq"

SAVE_FLAT = SAVE_ROOT / "flat"
SAVE_SEQ  = SAVE_ROOT / "seq"


# =========================================================
# 설정
# =========================================================
RANDOM_STATE   = 42

# SMOTE 후 목표 비율
# 0.5 = Bot : Normal = 1 : 1
# 0.3 = Bot : Normal = 3 : 7
TARGET_RATIO   = 0.5


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

    # ── 원본 train 로드 ───────────────────────────────────
    X_train = np.load(SRC_FLAT / "X_train.npy")
    y_train = np.load(SRC_FLAT / "y_train.npy").astype(int)

    n_feat = X_train.shape[1]

    print(f"\n[LOAD] X_train: {X_train.shape}")
    print(f"[LOAD] Bot(1): {y_train.sum():,}  "
          f"Normal(0): {(y_train == 0).sum():,}  "
          f"Bot 비율: {y_train.mean():.4f}")

    # ── SMOTE 적용 ────────────────────────────────────────
    # sampling_strategy: 소수 클래스 / 다수 클래스 비율
    n_majority  = (y_train == 0).sum()
    n_target    = int(n_majority * TARGET_RATIO / (1 - TARGET_RATIO))
    n_current   = y_train.sum()

    if n_target <= n_current:
        print(f"\n[INFO] 이미 목표 비율 달성 — SMOTE 불필요")
        print(f"       현재 Bot: {n_current:,} / 목표: {n_target:,}")
        return

    sampling_strategy = {1: n_target}

    print(f"\n[SMOTE] 생성할 Bot 샘플: {n_target - n_current:,}개")
    print(f"[SMOTE] 증강 후 Bot: {n_target:,} / Normal: {n_majority:,}")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    X_aug, y_aug = smote.fit_resample(X_train, y_train)

    X_aug = X_aug.astype(np.float32)
    y_aug = y_aug.astype(np.int32)

    print(f"\n[SMOTE] 완료")
    print(f"  X_aug shape:   {X_aug.shape}")
    print(f"  Bot(1): {y_aug.sum():,}  "
          f"Normal(0): {(y_aug == 0).sum():,}  "
          f"Bot 비율: {y_aug.mean():.4f}")

    # ── flat 저장 (RF/XGB) ────────────────────────────────
    np.save(SAVE_FLAT / "X_train.npy", X_aug)
    np.save(SAVE_FLAT / "y_train.npy", y_aug)
    print(f"\n[SAVE] flat/X_train.npy  shape={X_aug.shape}")

    # ── seq 저장 (CNN-LSTM/GRU): (n, 77, 1) ──────────────
    X_aug_seq = X_aug.reshape(-1, n_feat, 1)
    np.save(SAVE_SEQ / "X_train.npy", X_aug_seq)
    np.save(SAVE_SEQ / "y_train.npy", y_aug)
    print(f"[SAVE] seq/X_train.npy   shape={X_aug_seq.shape}")

    # ── val / test 원본에서 복사 ──────────────────────────
    for split in ["val", "test"]:
        for d_src, d_dst in [(SRC_FLAT, SAVE_FLAT), (SRC_SEQ, SAVE_SEQ)]:
            for fname in [f"X_{split}.npy", f"y_{split}.npy"]:
                src = d_src / fname
                dst = d_dst / fname
                if src.exists():
                    shutil.copy2(src, dst)

    # scaler 복사
    scaler_src = SRC_SEQ / "scaler_flow.pkl"
    if scaler_src.exists():
        shutil.copy2(scaler_src, SAVE_SEQ / "scaler_flow.pkl")

    print(f"\n[COPY] val/test 원본 복사 완료")

    # ── meta 저장 ─────────────────────────────────────────
    meta = {
        "method":            "SMOTE",
        "target_ratio":      TARGET_RATIO,
        "original_bot":      int(n_current),
        "augmented_bot":     int(y_aug.sum()),
        "generated_samples": int(n_target - n_current),
        "normal_count":      int((y_aug == 0).sum()),
        "total_train":       int(len(y_aug)),
        "bot_ratio_after":   float(y_aug.mean()),
        "n_features":        n_feat,
        "k_neighbors":       5,
        "random_state":      RANDOM_STATE,
    }
    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"\n[DONE] {SAVE_ROOT}")
    print(f"\n[다음 단계]")
    print(f"  train_*.py 에서 DATA_DIR를 아래로 변경 후 재학습:")
    print(f"  cicids2017 → cicids2017_smote")


if __name__ == "__main__":
    main()