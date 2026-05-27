"""
augment_gan.py

Bot 클래스 GAN Generator 학습 — K-fold 대응
서브그룹 분리 전략 (Zhao et al. 2024 방식)

수정 사항:
  - Generator 출력: Tanh[-1,1] → Sigmoid[0,1]  (MinMaxScaler 범위 일치)
  - TARGET_RATIO: 0.5 → 0.1  (SMOTE와 동일 비율로 공정 비교)
  - 전체 augmented 데이터 저장 제거  (K-fold fold-level 증강과 충돌)
  - Generator만 저장 → augment_utils.py가 fold마다 생성

저장 경로: data/processed/{dataset}_gan/
  generator.pt  ← 세그먼트별 Generator 전체 (augment_utils.py에서 로드)
  meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# 인자 파싱
# =========================================================
_parser = argparse.ArgumentParser()
_parser.add_argument("--dataset", type=str, default="cicids2017",
                     choices=["cicids2017", "cicids2018", "ctu13"])
_parser.add_argument("--n_segments", type=int, default=4)
ARGS       = _parser.parse_args()
DATASET    = ARGS.dataset
N_SEGMENTS = ARGS.n_segments


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent

SRC_FLAT  = _PROJECT / "data" / "processed" / DATASET / "flat"
SAVE_ROOT = _PROJECT / "data" / "processed" / f"{DATASET}_gan"


# =========================================================
# 하이퍼파라미터
# =========================================================
NOISE_DIM    = 100
N_EPOCHS     = 500
BATCH_SIZE   = 64
LR_G         = 2e-4
LR_D         = 2e-4
TARGET_RATIO = 0.1
RANDOM_STATE = 42
MIN_SEG_SIZE = 30


# =========================================================
# 서브그룹 분리
# =========================================================
def segment_botnet(X_bot: np.ndarray, n_segments: int) -> List[np.ndarray]:
    """
    Zhao et al. (2024): 저분산 컬럼 기준 분리 → K-Means fallback
    """
    col_uniq      = np.array([len(np.unique(X_bot[:, c])) for c in range(X_bot.shape[1])])
    low_card_cols = np.where((col_uniq >= 2) & (col_uniq <= 5))[0]

    if len(low_card_cols) > 0:
        sorted_cols = low_card_cols[np.argsort(X_bot[:, low_card_cols].var(axis=0))]
        segments    = [X_bot]
        for col in sorted_cols:
            if len(segments) >= n_segments:
                break
            new_segs = []
            for seg in segments:
                mid   = np.median(seg[:, col])
                left  = seg[seg[:, col] <= mid]
                right = seg[seg[:, col] >  mid]
                if len(left) >= MIN_SEG_SIZE and len(right) >= MIN_SEG_SIZE:
                    new_segs.extend([left, right])
                else:
                    new_segs.append(seg)
            segments = new_segs

        valid   = [s for s in segments if len(s) >= MIN_SEG_SIZE]
        invalid = [s for s in segments if len(s) <  MIN_SEG_SIZE]
        if invalid and valid:
            valid[0] = np.vstack([valid[0]] + invalid)
        elif invalid:
            valid = [np.vstack(invalid)]

        if len(valid) >= 2:
            print(f"[SEG] 저분산 컬럼 기반 → {len(valid)}개 세그먼트")
            for i, s in enumerate(valid):
                print(f"      seg[{i}]: {len(s):,}개")
            return valid

    try:
        from sklearn.cluster import KMeans
        k  = max(2, min(n_segments, len(X_bot) // MIN_SEG_SIZE))
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels   = km.fit_predict(X_bot)
        segments = [X_bot[labels == i] for i in range(k) if (labels == i).sum() >= MIN_SEG_SIZE]
        if len(segments) >= 2:
            print(f"[SEG] K-Means(k={k}) → {len(segments)}개 세그먼트")
            for i, s in enumerate(segments):
                print(f"      seg[{i}]: {len(s):,}개")
            return segments
    except ImportError:
        pass

    print(f"[SEG] 분리 불가 → 전체 Bot {len(X_bot):,}개 단일 사용")
    return [X_bot]


def per_segment_counts(segments: List[np.ndarray], n_total: int) -> List[int]:
    total  = sum(len(s) for s in segments)
    counts = [int(n_total * len(s) / total) for s in segments]
    counts[0] += n_total - sum(counts)
    return counts


# =========================================================
# 모델
# =========================================================
class Generator(nn.Module):
    def __init__(self, noise_dim: int, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512),       nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, n_features),
            nn.Sigmoid(),   # [0,1] — MinMaxScaler 범위와 일치
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# =========================================================
# 학습
# =========================================================
def train_gan(X_seg: np.ndarray, noise_dim: int, n_epochs: int,
              batch_size: int, lr_g: float, lr_d: float,
              device: torch.device, seg_idx: int = 0) -> Generator:
    n_features = X_seg.shape[1]
    G = Generator(noise_dim, n_features).to(device)
    D = Discriminator(n_features).to(device)

    opt_G     = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D     = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    eff_batch = max(8, min(batch_size, len(X_seg) // 2))
    loader    = DataLoader(TensorDataset(torch.tensor(X_seg, dtype=torch.float32)),
                           batch_size=eff_batch, shuffle=True, drop_last=True)

    print(f"\n  [GAN seg={seg_idx}] 샘플={len(X_seg):,}  batch={eff_batch}  device={device}")

    for epoch in range(1, n_epochs + 1):
        g_losses, d_losses = [], []
        for (X_real,) in loader:
            X_real = X_real.to(device)
            bs     = X_real.size(0)

            z      = torch.randn(bs, noise_dim, device=device)
            X_fake = G(z).detach()
            loss_D = (criterion(D(X_real), torch.ones(bs, device=device)) +
                      criterion(D(X_fake), torch.zeros(bs, device=device))) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            z      = torch.randn(bs, noise_dim, device=device)
            loss_G = criterion(D(G(z)), torch.ones(bs, device=device))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        if epoch % 100 == 0 or epoch == 1:
            print(f"    [Epoch {epoch:4d}/{n_epochs}] "
                  f"G={np.mean(g_losses):.4f}  D={np.mean(d_losses):.4f}")

    return G


# =========================================================
# main
# =========================================================
def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"  GAN Augmentation  [dataset={DATASET}]")
    print("=" * 65)
    print(f"  TARGET_RATIO={TARGET_RATIO}  N_SEGMENTS={N_SEGMENTS}  device={device}")
    print(f"  → Generator만 저장 (fold-level 증강은 augment_utils.py가 담당)")
    print("=" * 65)

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    X_trainval = np.load(SRC_FLAT / "X_trainval.npy")
    y_trainval = np.load(SRC_FLAT / "y_trainval.npy").astype(int)
    n_feat     = X_trainval.shape[1]
    n_majority = (y_trainval == 0).sum()
    n_current  = y_trainval.sum()

    print(f"\n[LOAD] X_trainval: {X_trainval.shape}")
    print(f"[LOAD] Bot(1)={n_current:,}  Normal(0)={n_majority:,}  "
          f"비율={y_trainval.mean():.4f}")

    # 서브그룹 분리
    X_bot    = X_trainval[y_trainval == 1]
    segments = segment_botnet(X_bot, N_SEGMENTS)

    # 세그먼트별 GAN 학습
    generators = []
    for i, seg in enumerate(segments):
        print(f"\n{'─'*50}")
        print(f"[GAN] 세그먼트 {i+1}/{len(segments)} 학습...")
        G_i = train_gan(seg, NOISE_DIM, N_EPOCHS, BATCH_SIZE, LR_G, LR_D, device, seg_idx=i)
        generators.append(G_i)

    # Generator 저장 (단일 파일, augment_utils.py에서 로드)
    gen_path = SAVE_ROOT / "generator.pt"
    torch.save({
        "generators":    [G_i.state_dict() for G_i in generators],
        "segment_sizes": [len(s) for s in segments],
        "noise_dim":     NOISE_DIM,
        "n_features":    n_feat,
        "target_ratio":  TARGET_RATIO,
    }, gen_path)
    print(f"\n[SAVE] Generator 저장: {gen_path}")
    print(f"       세그먼트 {len(generators)}개  각 크기: {[len(s) for s in segments]}")

    meta = {
        "method": "GAN_subgroup", "target_ratio": TARGET_RATIO,
        "n_segments": len(segments),
        "segment_sizes": [int(len(s)) for s in segments],
        "original_bot": int(n_current),
        "n_features": n_feat, "noise_dim": NOISE_DIM,
        "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
        "lr_g": LR_G, "lr_d": LR_D,
        "note": "Generator만 저장. 실제 증강은 K-fold train fold 내부에서 augment_utils.py가 수행.",
    }
    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"\n[DONE] {SAVE_ROOT}")
    print(f"  다음 단계: python train_rf.py --dataset {DATASET} --augment gan")


if __name__ == "__main__":
    main()
