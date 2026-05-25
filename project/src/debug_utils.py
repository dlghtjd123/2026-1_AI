"""
debug_utils.py

원인 분석용 디버그 유틸리티

사용법 (train_rf.py / train_xgb.py 등 fold 평가 직후):
  from debug_utils import debug_fold

  debug_fold(
      fold=fold,
      y_val=y_val,
      y_pred=y_pred,
      y_prob=y_prob,
      y_train_orig=y_train_orig,   # 증강 전 라벨
      y_train_aug=y_train_aug,     # 증강 후 라벨
      class_weight=class_weight,   # RF용 (없으면 None)
      scale_pos_weight=spw,        # XGB용 (없으면 None)
  )
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix


def debug_fold(
    fold: int,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    y_train_orig: np.ndarray | None = None,
    y_train_aug:  np.ndarray | None = None,
    class_weight=None,
    scale_pos_weight: float | None = None,
    pos_weight: float | None = None,
    augment: str = "none",   # 증강 방식 ("none" 이면 이중 보정 경고 미표시)
) -> None:
    """
    fold 평가 직후 호출. 이중 보정 및 확률 분포 분석.

    확인 포인트:
      1. 증강 전/후 클래스 비율
      2. 모델 내부 클래스 보정값 (class_weight / scale_pos_weight / pos_weight)
      3. Confusion Matrix (TN/FP/FN/TP) 및 FP율
      4. 예측 확률 분포 (Botnet 쪽 편향 여부)
    """
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  [DEBUG] Fold {fold}")
    print(sep)

    # ── 1. 클래스 비율 ─────────────────────────────────────
    print("\n  [DEBUG-1] 클래스 비율")
    if y_train_orig is not None:
        n_bot_orig = int(y_train_orig.sum())
        print(f"    원본 train  : Bot={n_bot_orig:,}  비율={y_train_orig.mean():.6f}")
    if y_train_aug is not None:
        n_bot_aug = int(y_train_aug.sum())
        print(f"    증강 후     : Bot={n_bot_aug:,}  비율={y_train_aug.mean():.4f}")
    print(f"    val (원본)  : Bot={(y_val==1).sum():,}  비율={y_val.mean():.6f}")

    if y_train_orig is not None and y_train_aug is not None:
        aug_ratio = y_train_aug.mean() / max(y_train_orig.mean(), 1e-9)
        val_ratio = y_val.mean()
        mismatch  = y_train_aug.mean() / max(val_ratio, 1e-9)
        print(f"    증폭 배수   : {aug_ratio:.0f}x")
        print(f"    train/val 비율 불일치: {mismatch:.0f}x  "
              f"← {'⚠️ 심각' if mismatch > 50 else '양호'}")

    # ── 2. 모델 내부 클래스 보정 ───────────────────────────
    print("\n  [DEBUG-2] 모델 내부 클래스 보정값")
    if class_weight is not None:
        print(f"    class_weight     : {class_weight}  "
              f"{'⚠️ 이중 보정 의심' if class_weight not in (None, 1, 'none') else 'OK'}")
    if scale_pos_weight is not None:
        print(f"    scale_pos_weight : {scale_pos_weight:.4f}  "
              f"{'⚠️ 이중 보정 의심' if scale_pos_weight > 1.5 else 'OK'}")
    if pos_weight is not None:
        print(f"    pos_weight       : {pos_weight:.4f}  "
              f"{'⚠️ 이중 보정 의심' if pos_weight > 1.5 else 'OK'}")
    if all(v is None for v in [class_weight, scale_pos_weight, pos_weight]):
        print(f"    보정값 없음 (증강만 적용)")

    # ── 3. Confusion Matrix ────────────────────────────────
    print("\n  [DEBUG-3] Confusion Matrix")
    cm           = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fp_rate      = fp / max(fp + tn, 1)
    fn_rate      = fn / max(fn + tp, 1)
    print(f"    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"    FP율 (오탐률): {fp_rate:.6f}  "
          f"← {'⚠️ 높음' if fp_rate > 0.001 else '정상'}")
    print(f"    FN율 (미탐률): {fn_rate:.6f}")
    print(f"    FP/TP 비율   : {fp/max(tp,1):.2f}  "
          f"(1 이상이면 오탐이 실제 탐지보다 많음)")

    # ── 4. 예측 확률 분포 ──────────────────────────────────
    print("\n  [DEBUG-4] 예측 확률 분포")
    quantiles = np.quantile(y_prob, [0.5, 0.9, 0.99, 0.999, 0.9999])
    print(f"    전체   min={y_prob.min():.4f}  "
          f"mean={y_prob.mean():.6f}  max={y_prob.max():.4f}")
    print(f"    분위수  50%={quantiles[0]:.4f}  90%={quantiles[1]:.4f}  "
          f"99%={quantiles[2]:.4f}  99.9%={quantiles[3]:.4f}  99.99%={quantiles[4]:.4f}")

    prob_normal = y_prob[y_val == 0]
    prob_bot    = y_prob[y_val == 1]
    print(f"    Normal  mean={prob_normal.mean():.6f}  "
          f"max={prob_normal.max():.4f}  >0.5인 비율={( prob_normal > 0.5).mean():.6f}")
    print(f"    Botnet  mean={prob_bot.mean():.4f}   "
          f"min={prob_bot.min():.4f}   >0.5인 비율={(prob_bot > 0.5).mean():.4f}")

    # ── 이중 보정 진단 ─────────────────────────────────────
    print(f"\n  [DEBUG-진단]")
    issues = []

    if y_train_aug is not None and y_train_orig is not None:
        ratio = y_train_aug.mean() / max(y_train_orig.mean(), 1e-9)
        if ratio > 50:
            issues.append(f"증폭 배수 과다 ({ratio:.0f}x) → prior mismatch")

    # 이중 보정은 증강이 실제로 적용됐을 때만 체크
    if augment != "none":
        if class_weight not in (None, "none"):
            issues.append(f"이중 보정: class_weight={class_weight} + 증강 동시 적용")
        if scale_pos_weight is not None and scale_pos_weight > 1.5:
            issues.append(f"이중 보정: scale_pos_weight={scale_pos_weight:.2f} + 증강 동시 적용")
        if pos_weight is not None and pos_weight > 1.5:
            issues.append(f"이중 보정: pos_weight={pos_weight:.2f} + 증강 동시 적용")
    else:
        if class_weight not in (None, "none"):
            print(f"    class_weight={class_weight} (baseline — 정상)")

    if fp / max(tp, 1) > 1:
        issues.append(f"FP > TP ({fp}/{tp}) → 오탐이 실제 탐지보다 많음")

    prob_normal = y_prob[y_val == 0]
    if len(prob_normal) > 0 and prob_normal.mean() > 0.005:
        issues.append(f"정상 샘플 확률 평균 높음 ({prob_normal.mean():.6f}) → 모델 편향")

    if issues:
        print(f"    ⚠️  의심 원인:")
        for iss in issues:
            print(f"       - {iss}")
    else:
        print(f"    ✅  이상 없음")

    print(sep)