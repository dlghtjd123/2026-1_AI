"""
debug_utils.py

K-fold 내부 원인 분석용 디버그 유틸리티.
train_rf.py / train_cnn_lstm.py 에서 --debug 또는 증강 시 자동 호출.
"""

from __future__ import annotations

import numpy as np


def debug_fold(
    fold: int,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    y_train_orig: np.ndarray,
    y_train_aug: np.ndarray | None = None,
    augment: str = "none",
    class_weight: str | None = None,
    pos_weight: float | None = None,
) -> None:
    """
    Fold 단위 진단 정보 출력.

    Args:
        fold:         현재 fold 번호
        y_val:        val fold 원본 라벨
        y_pred:       val fold 예측 라벨
        y_prob:       val fold 봇넷 예측 확률
        y_train_orig: 증강 전 train fold 라벨 (subsample 후)
        y_train_aug:  증강 후 train fold 라벨 (None이면 증강 없음)
        augment:      증강 방식 문자열
        class_weight: RF class_weight 설정값 (str, 선택)
        pos_weight:   DL Focal Loss 등가 pos_weight (float, 선택)
    """
    sep = f"  {'─'*54}"
    print(f"\n{sep}")
    print(f"  [DEBUG Fold {fold}]  augment={augment}")

    # ── 1. Val 클래스 분포 ──────────────────────────────────
    n_val     = len(y_val)
    n_bot_val = int(y_val.sum())
    n_nor_val = n_val - n_bot_val
    print(f"\n  [VAL 분포]  총={n_val:,}  "
          f"Bot={n_bot_val:,} ({n_bot_val/n_val*100:.3f}%)  "
          f"Normal={n_nor_val:,}")

    # ── 2. 예측 분포 ────────────────────────────────────────
    n_pred_bot = int(y_pred.sum())
    n_pred_nor = len(y_pred) - n_pred_bot
    print(f"  [PRED 분포]  Bot_pred={n_pred_bot:,}  Normal_pred={n_pred_nor:,}")

    # ── 3. 혼동 행렬 요약 ───────────────────────────────────
    tp = int(((y_val == 1) & (y_pred == 1)).sum())
    fp = int(((y_val == 0) & (y_pred == 1)).sum())
    tn = int(((y_val == 0) & (y_pred == 0)).sum())
    fn = int(((y_val == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    print(f"  [CM]  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
    print(f"        Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    # ── 4. 확률 분포 ────────────────────────────────────────
    if len(y_prob) > 0:
        bot_probs = y_prob[y_val == 1]
        nor_probs = y_prob[y_val == 0]
        print(f"\n  [PROB 분포 — 봇넷 샘플]  "
              f"min={bot_probs.min():.4f}  "
              f"mean={bot_probs.mean():.4f}  "
              f"median={np.median(bot_probs):.4f}  "
              f"max={bot_probs.max():.4f}" if len(bot_probs) > 0
              else "\n  [PROB 분포 — 봇넷 샘플]  (없음)")
        if len(nor_probs) > 0:
            print(f"  [PROB 분포 — 정상 샘플]  "
                  f"min={nor_probs.min():.4f}  "
                  f"mean={nor_probs.mean():.4f}  "
                  f"median={np.median(nor_probs):.4f}  "
                  f"max={nor_probs.max():.4f}")

        # 임계값별 Recall 참고
        for thr in (0.3, 0.2, 0.1, 0.05):
            preds_t = (y_prob >= thr).astype(int)
            tp_t    = int(((y_val == 1) & (preds_t == 1)).sum())
            fp_t    = int(((y_val == 0) & (preds_t == 1)).sum())
            fn_t    = int(((y_val == 1) & (preds_t == 0)).sum())
            rec_t   = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
            pre_t   = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
            f1_t    = (2 * pre_t * rec_t / (pre_t + rec_t)
                       if (pre_t + rec_t) > 0 else 0.0)
            print(f"  [THR {thr:.2f}]  "
                  f"TP={tp_t:,}  FP={fp_t:,}  FN={fn_t:,}  "
                  f"Recall={rec_t:.4f}  Precision={pre_t:.4f}  F1={f1_t:.4f}")

    # ── 5. 학습 데이터 분포 (증강 전후) ─────────────────────
    n_orig     = len(y_train_orig)
    n_bot_orig = int(y_train_orig.sum())
    print(f"\n  [TRAIN 증강 전]  총={n_orig:,}  "
          f"Bot={n_bot_orig:,} ({n_bot_orig/n_orig*100:.3f}%)")

    if y_train_aug is not None:
        n_aug     = len(y_train_aug)
        n_bot_aug = int(y_train_aug.sum())
        added     = n_bot_aug - n_bot_orig
        print(f"  [TRAIN 증강 후]  총={n_aug:,}  "
              f"Bot={n_bot_aug:,} ({n_bot_aug/n_aug*100:.3f}%)  "
              f"추가={added:,}")

    # ── 6. 가중치 설정 안내 ─────────────────────────────────
    if class_weight is not None:
        print(f"\n  [WEIGHT] class_weight={class_weight!r}")
    if pos_weight is not None:
        print(f"  [WEIGHT] Focal Loss 등가 pos_weight={pos_weight:.2f}x")

    # ── 7. 과보정 경고 ──────────────────────────────────────
    if n_bot_val > 0:
        fp_rate = fp / n_nor_val if n_nor_val > 0 else 0.0
        fn_rate = fn / n_bot_val if n_bot_val > 0 else 0.0
        if fp_rate > 0.05:
            print(f"\n  [WARN] FP율={fp_rate:.3f} > 5% — 과보정(over-correction) 의심")
        if fn_rate > 0.5:
            print(f"  [WARN] FN율={fn_rate:.3f} > 50% — 봇넷 탐지율 저조")

    print(sep)
