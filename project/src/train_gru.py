import copy
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_utils import augment_train_fold
from debug_utils import debug_fold


# =========================================================
# 인자 파싱
# =========================================================
_parser = argparse.ArgumentParser()
_parser.add_argument("--augment", type=str, default="none",
                     choices=["none", "smote", "gan", "wgan_gp", "wcgan_gp"])
_parser.add_argument("--dataset", type=str, default="cicids2017",
                     choices=["cicids2017", "cicids2018", "ctu13"])
_parser.add_argument("--n_folds", type=int, default=5)
_parser.add_argument("--debug",   action="store_true",
                     help="디버그 모드: 원인 분석 로그 출력")
_parser.add_argument("--max_normal", type=int, default=500_000,
                     help="fold당 최대 정상 샘플 수 상한 (기본값: 500000, 0=제한 없음)")
_parser.add_argument("--max_mismatch", type=float, default=10.0,
                     help="train/val 봇넷 비율 최대 배수 (기본값: 10)")
AUGMENT      = _parser.parse_args().augment
DATASET      = _parser.parse_args().dataset
N_FOLDS      = _parser.parse_args().n_folds
DEBUG        = _parser.parse_args().debug
MAX_NORMAL   = _parser.parse_args().max_normal
MAX_MISMATCH = _parser.parse_args().max_mismatch

# Focal Loss alpha → pos_weight 등가값 (이중 보정 진단용)
# alpha=0.75 → 봇넷 클래스가 정상 대비 0.75/0.25 = 3.0배 가중
FOCAL_ALPHA     = 0.75
FOCAL_POS_EQUIV = FOCAL_ALPHA / (1.0 - FOCAL_ALPHA)   # 3.0


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent
_ROOT     = _PROJECT.parent

DATA_DIR   = _PROJECT / "data" / "processed" / DATASET / "seq"
DATA_ROOT  = _PROJECT / "data" / "processed"
_MODEL_SUFFIX = f"_{AUGMENT}" if AUGMENT != "none" else ""
MODEL_DIR  = _ROOT / "artifacts" / f"models_{DATASET}{_MODEL_SUFFIX}" / "gru"
RESULT_DIR = _ROOT / "artifacts" / f"results_{DATASET}{_MODEL_SUFFIX}"


# =========================================================
# 유틸
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss (Normal / Botnet)
    alpha=0.75 → 봇넷 클래스 3배 가중 (이중 보정 주의)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight  = torch.tensor([1.0 - self.alpha, self.alpha], device=logits.device)
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        p_t     = torch.exp(-ce_loss)
        loss    = (1.0 - p_t) ** self.gamma * ce_loss
        return loss.mean()


class GRUModel(nn.Module):
    def __init__(self, n_features, gru_hidden=64, dropout=0.3):
        super().__init__()
        self.gru     = nn.GRU(n_features, gru_hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(gru_hidden, 2)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1]))


def load_data(data_dir: Path):
    X = np.load(data_dir / "X_trainval.npy")
    y = np.load(data_dir / "y_trainval.npy").astype(int)
    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0, output_dict=True
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def collect_probs_and_loss(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            total_loss += criterion(logits, y_batch).item() * X_batch.size(0)
            y_true_all.extend(y_batch.cpu().numpy().tolist())
            y_prob_all.extend(probs.cpu().numpy().tolist())
    return (
        total_loss / len(loader.dataset),
        np.array(y_true_all).astype(int),
        np.array(y_prob_all),
    )


def train_one_fold(X_train, y_train, X_val, y_val, device, fold, criterion=None):
    """
    Returns:
        model, threshold, best_val_metrics, n_features, best_val_prob
        ↑ best_val_prob 추가 — debug_fold에서 확률 분포 분석용
    """
    n_features   = X_train.shape[2]
    train_loader = DataLoader(SequenceDataset(X_train, y_train),
                              batch_size=128, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(SequenceDataset(X_val, y_val),
                              batch_size=256, shuffle=False, num_workers=0)

    model = GRUModel(n_features=n_features).to(device)
    if criterion is None:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs       = 30
    patience         = 6
    min_epochs       = 20
    best_score       = None
    best_state       = None
    best_threshold   = "argmax"
    best_epoch       = 0
    best_val_metrics = None
    best_val_prob    = None       # debug용 저장
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, y_val_true, y_val_prob = collect_probs_and_loss(
            model, val_loader, device, criterion
        )
        y_pred              = (y_val_prob >= 0.5).astype(int)
        current_val_metrics = compute_metrics(y_val_true, y_pred, y_val_prob)
        current_val_metrics["selected_threshold"] = "argmax"

        current_score = (
            current_val_metrics["f1"],
            current_val_metrics["recall"],
            current_val_metrics["precision"],
            current_val_metrics["roc_auc"] if current_val_metrics["roc_auc"] is not None else -1.0,
            -val_loss,
        )

        print(
            f"  [Fold {fold} Epoch {epoch:02d}] "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"argmax | f1={current_val_metrics['f1']:.4f} | "
            f"recall={current_val_metrics['recall']:.4f}"
        )

        if best_score is None or current_score > best_score:
            best_score       = current_score
            best_state       = copy.deepcopy(model.state_dict())
            best_threshold   = "argmax"
            best_epoch       = epoch
            best_val_metrics = current_val_metrics
            best_val_prob    = y_val_prob.copy()   # debug용
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= min_epochs and patience_counter >= patience:
            print(f"  [Fold {fold}] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    best_val_metrics["selected_threshold"] = best_threshold
    print(f"  [Fold {fold}] Best epoch: {best_epoch}")
    return model, best_threshold, best_val_metrics, n_features, best_val_prob


def print_fold_summary(fold_results: list[dict]) -> dict:
    keys = ["f1", "recall", "precision", "roc_auc", "accuracy"]
    summary = {}
    print(f"\n{'='*60}")
    print(f"  GRU  {N_FOLDS}-Fold CV 결과 요약  [{DATASET} / {AUGMENT}]")
    print(f"  ★ 주 지표: F1, Recall  /  보조: ROC-AUC")
    print(f"{'='*60}")
    print(f"{'지표':<14} {'평균':>8} {'표준편차':>10} {'최소':>8} {'최대':>8}")
    print("-" * 52)
    for key in keys:
        vals = [r[key] for r in fold_results if r.get(key) is not None]
        if not vals:
            continue
        mean, std = float(np.mean(vals)), float(np.std(vals))
        summary[key] = {"mean": mean, "std": std,
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
        marker = " ★" if key in ("f1", "recall") else ""
        print(f"{key:<14} {mean:>8.4f} {std:>10.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}{marker}")
    print("=" * 60)
    return summary


# =========================================================
# main
# =========================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[CONFIG] dataset : {DATASET}")
    print(f"[CONFIG] augment : {AUGMENT}")
    print(f"[CONFIG] n_folds : {N_FOLDS}")
    print(f"[CONFIG] debug           : {DEBUG}")
    print(f"[CONFIG] max_normal       : {MAX_NORMAL:,} (0=제한 없음)")
    print(f"[CONFIG] max_mismatch     : {MAX_MISMATCH:.0f}x  (train/val 봇넷 비율 최대 배수)")
    print(f"[CONFIG] data    : {DATA_DIR}")
    print(f"[INFO]   device  : {device}")
    print(f"[INFO]   Focal Loss alpha={FOCAL_ALPHA}  "
          f"(등가 pos_weight={FOCAL_POS_EQUIV:.1f}x)")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X_all, y_all = load_data(DATA_DIR)
    print(f"[INFO] X_trainval shape : {X_all.shape}")
    print(f"[INFO] Botnet ratio     : {y_all.mean():.4f}")

    print(f"[INFO] K-fold 전 shape: {X_all.shape}  Bot={y_all.sum():,}  "
          f"(subsample은 각 fold train에만 적용)")

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results    = []
    best_fold_score = None
    best_fold_model = None
    best_fold_thr   = "argmax"
    best_fold_idx   = -1
    best_n_features = X_all.shape[2]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), 1):
        print(f"\n[Fold {fold}/{N_FOLDS}] train={len(train_idx):,}  val={len(val_idx):,}")
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        # 증강 전 라벨 저장 (debug용)
        y_train_orig = y_train.copy()

        # train fold에만 subsample 적용 — val은 원본 분포 유지
        if MAX_NORMAL > 0:
            from augment_utils import subsample_benign
            _nf = X_train.shape[1]
            X_train_f, y_train = subsample_benign(
                X_train.reshape(len(X_train), -1), y_train,
                max_normal=MAX_NORMAL, max_mismatch=MAX_MISMATCH
            )
            X_train = X_train_f.reshape(-1, _nf, 1)
            y_train_orig = y_train.copy()  # subsample 후 orig 갱신

        # train fold에만 증강 — val fold는 항상 원본 유지
        X_train, y_train = augment_train_fold(
            X_train, y_train, AUGMENT, DATASET, DATA_ROOT
        )
        if AUGMENT != "none":
            print(f"  [AUG] train: {len(y_train):,}  val: {len(y_val):,} (원본)")

        # 증강 시 FocalLoss 대신 CrossEntropyLoss (이중 보정 방지)
        criterion = (nn.CrossEntropyLoss() if AUGMENT != "none"
                     else FocalLoss(alpha=FOCAL_ALPHA, gamma=2.0))
        model, thr, metrics, n_feat, val_prob = train_one_fold(
            X_train, y_train, X_val, y_val, device, fold, criterion=criterion
        )
        metrics["fold"] = fold
        fold_results.append(metrics)

        print(
            f"  → F1={metrics['f1']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"ROC-AUC={metrics.get('roc_auc', 0):.4f}"
        )

        # ── 디버그 출력 (--debug 또는 증강 시 자동) ──────────
        if DEBUG or AUGMENT != "none":
            y_pred = (val_prob >= 0.5).astype(int)
            debug_fold(
                fold=fold,
                y_val=y_val,
                y_pred=y_pred,
                y_prob=val_prob,
                y_train_orig=y_train_orig,
                y_train_aug=y_train if AUGMENT != "none" else None,
                # CrossEntropyLoss 사용 시 pos_weight=None (이중 보정 아님)
                pos_weight=None if AUGMENT != "none" else FOCAL_POS_EQUIV,
                augment=AUGMENT,
            )

        score = (metrics["f1"], metrics["recall"])
        if best_fold_score is None or score > best_fold_score:
            best_fold_score = score
            best_fold_model = model
            best_fold_thr   = thr
            best_fold_idx   = fold
            best_n_features = n_feat

    summary = print_fold_summary(fold_results)
    print(f"\n[INFO] Best fold: {best_fold_idx}  (F1={best_fold_score[0]:.4f})")

    torch.save(
        {
            "model_state_dict": best_fold_model.state_dict(),
            "n_features":    best_n_features,
            "n_classes":     2,
            "window_size":   X_all.shape[1],
            "gru_hidden":    64,
            "dropout":       0.3,
        },
        MODEL_DIR / "gru_flow.pt",
    )

    with open(MODEL_DIR / "gru_flow_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_fold_thr, "best_fold": best_fold_idx}, f, indent=4)

    output = {
        "dataset":        DATASET,
        "augment":        AUGMENT,
        "n_folds":        N_FOLDS,
        "primary_metric": ["f1", "recall"],
        "summary":        summary,
        "fold_results":   fold_results,
        "best_fold":      best_fold_idx,
    }
    with open(RESULT_DIR / "gru_flow_kfold_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"\n[SAVED] {MODEL_DIR}/gru_flow.pt")
    print(f"[SAVED] {MODEL_DIR}/gru_flow_threshold.json")
    print(f"[SAVED] {RESULT_DIR}/gru_flow_kfold_results.json")


if __name__ == "__main__":
    main()