"""
augment_gan.py

CIC-IDS2017 Bot 클래스 GAN 증강

구조:
  Generator:     noise(100) → 77피처 Bot 샘플
  Discriminator: 77피처 → real/fake 판별

증강 대상: train 데이터 Bot 클래스만
저장 경로: data/processed/cicids2017_gan/
  flat/ X_train.npy  y_train.npy
  seq/  X_train.npy  y_train.npy
  val, test는 원본에서 복사

이후: train_*.py의 DATA_DIR를
      cicids2017 → cicids2017_gan 로 변경 후 재학습
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# 경로 설정
# =========================================================
_SRC_DIR  = Path(__file__).resolve().parent
_PROJECT  = _SRC_DIR.parent

SRC_ROOT  = _PROJECT / "data" / "processed" / "cicids2017"
SAVE_ROOT = _PROJECT / "data" / "processed" / "cicids2017_gan"

SRC_FLAT  = SRC_ROOT / "flat"
SRC_SEQ   = SRC_ROOT / "seq"

SAVE_FLAT = SAVE_ROOT / "flat"
SAVE_SEQ  = SAVE_ROOT / "seq"


# =========================================================
# 설정
# =========================================================
NOISE_DIM    = 100      # Generator 입력 노이즈 차원
N_EPOCHS     = 500      # 학습 에포크
BATCH_SIZE   = 256      # 배치 크기
LR_G         = 2e-4     # Generator 학습률
LR_D         = 2e-4     # Discriminator 학습률
TARGET_RATIO = 0.5      # 증강 후 목표 Bot 비율
RANDOM_STATE = 42


# =========================================================
# 모델 정의
# =========================================================
class Generator(nn.Module):
    """
    노이즈 → Bot 플로우 생성
    입력: (batch, NOISE_DIM)
    출력: (batch, n_features)
    """
    def __init__(self, noise_dim: int, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, n_features),
            nn.Tanh(),          # 출력 범위 [-1, 1]
                                # 입력 데이터도 StandardScaler → 비슷한 범위
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """
    Bot 플로우 → real/fake 판별
    입력: (batch, n_features)
    출력: (batch,) → sigmoid 확률
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            # BCEWithLogitsLoss 사용 → sigmoid 없음
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# =========================================================
# 학습
# =========================================================
def train_gan(
    X_bot:      np.ndarray,
    noise_dim:  int,
    n_epochs:   int,
    batch_size: int,
    lr_g:       float,
    lr_d:       float,
    device:     torch.device,
) -> Generator:
    """
    Bot 샘플로만 GAN 학습
    """
    n_features = X_bot.shape[1]

    G = Generator(noise_dim, n_features).to(device)
    D = Discriminator(n_features).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(torch.tensor(X_bot, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"\n[GAN] 학습 시작")
    print(f"  Bot 샘플: {len(X_bot):,}  /  n_features: {n_features}")
    print(f"  Epochs: {n_epochs}  /  Batch: {batch_size}")
    print(f"  Device: {device}")

    for epoch in range(1, n_epochs + 1):
        g_losses, d_losses = [], []

        for (X_real,) in loader:
            X_real = X_real.to(device)
            bs     = X_real.size(0)

            real_label = torch.ones(bs,  device=device)
            fake_label = torch.zeros(bs, device=device)

            # ── Discriminator 학습 ──────────────────────
            z      = torch.randn(bs, noise_dim, device=device)
            X_fake = G(z).detach()

            loss_D = (
                criterion(D(X_real), real_label)
                + criterion(D(X_fake), fake_label)
            ) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── Generator 학습 ──────────────────────────
            z      = torch.randn(bs, noise_dim, device=device)
            X_fake = G(z)

            loss_G = criterion(D(X_fake), real_label)  # D를 속이려 함

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        if epoch % 50 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:4d}/{n_epochs}] "
                  f"G_loss={np.mean(g_losses):.4f}  "
                  f"D_loss={np.mean(d_losses):.4f}")

    print("[GAN] 학습 완료")
    return G


# =========================================================
# 샘플 생성
# =========================================================
def generate_samples(
    G:          Generator,
    n_samples:  int,
    noise_dim:  int,
    device:     torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    G.eval()
    samples = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            z   = torch.randn(end - start, noise_dim, device=device)
            samples.append(G(z).cpu().numpy())
    return np.vstack(samples).astype(np.float32)


# =========================================================
# main
# =========================================================
def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  GAN Augmentation — CIC-IDS2017 Bot 클래스")
    print("=" * 60)
    print(f"  SRC_ROOT     = {SRC_ROOT}")
    print(f"  SAVE_ROOT    = {SAVE_ROOT}")
    print(f"  TARGET_RATIO = {TARGET_RATIO}")
    print(f"  Device       = {device}")
    print("=" * 60)

    for d in [SAVE_FLAT, SAVE_SEQ]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 원본 train 로드 ───────────────────────────────────
    X_train = np.load(SRC_FLAT / "X_train.npy")
    y_train = np.load(SRC_FLAT / "y_train.npy").astype(int)

    n_feat     = X_train.shape[1]
    n_majority = (y_train == 0).sum()
    n_current  = y_train.sum()
    n_target   = int(n_majority * TARGET_RATIO / (1 - TARGET_RATIO))

    print(f"\n[LOAD] X_train: {X_train.shape}")
    print(f"[LOAD] Bot(1): {n_current:,}  "
          f"Normal(0): {n_majority:,}  "
          f"Bot 비율: {y_train.mean():.4f}")

    if n_target <= n_current:
        print(f"\n[INFO] 이미 목표 비율 달성 — GAN 불필요")
        return

    n_generate = n_target - n_current
    print(f"\n[GAN] 생성할 Bot 샘플: {n_generate:,}개")
    print(f"[GAN] 증강 후 Bot: {n_target:,} / Normal: {n_majority:,}")

    # ── Bot 샘플만 추출 ───────────────────────────────────
    X_bot = X_train[y_train == 1]
    print(f"\n[GAN] Bot 학습 데이터: {X_bot.shape}")

    # ── GAN 학습 ──────────────────────────────────────────
    G = train_gan(
        X_bot      = X_bot,
        noise_dim  = NOISE_DIM,
        n_epochs   = N_EPOCHS,
        batch_size = BATCH_SIZE,
        lr_g       = LR_G,
        lr_d       = LR_D,
        device     = device,
    )

    # ── 샘플 생성 ─────────────────────────────────────────
    X_fake = generate_samples(G, n_generate, NOISE_DIM, device)
    print(f"\n[GEN] 생성된 샘플: {X_fake.shape}")

    # ── 원본 + 생성 데이터 병합 ───────────────────────────
    X_aug = np.vstack([X_train, X_fake]).astype(np.float32)
    y_aug = np.concatenate([
        y_train,
        np.ones(n_generate, dtype=np.int32),
    ])

    # shuffle
    idx   = np.random.permutation(len(X_aug))
    X_aug = X_aug[idx]
    y_aug = y_aug[idx]

    print(f"\n[AUG] X_aug shape: {X_aug.shape}")
    print(f"[AUG] Bot(1): {y_aug.sum():,}  "
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

    scaler_src = SRC_SEQ / "scaler_flow.pkl"
    if scaler_src.exists():
        shutil.copy2(scaler_src, SAVE_SEQ / "scaler_flow.pkl")

    print(f"\n[COPY] val/test 원본 복사 완료")

    # ── Generator 저장 ────────────────────────────────────
    torch.save(
        {
            "model_state_dict": G.state_dict(),
            "noise_dim":        NOISE_DIM,
            "n_features":       n_feat,
        },
        SAVE_ROOT / "generator.pt",
    )

    # ── meta 저장 ─────────────────────────────────────────
    meta = {
        "method":            "GAN",
        "target_ratio":      TARGET_RATIO,
        "original_bot":      int(n_current),
        "augmented_bot":     int(y_aug.sum()),
        "generated_samples": int(n_generate),
        "normal_count":      int((y_aug == 0).sum()),
        "total_train":       int(len(y_aug)),
        "bot_ratio_after":   float(y_aug.mean()),
        "n_features":        n_feat,
        "noise_dim":         NOISE_DIM,
        "n_epochs":          N_EPOCHS,
        "batch_size":        BATCH_SIZE,
        "lr_g":              LR_G,
        "lr_d":              LR_D,
    }
    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"\n[DONE] {SAVE_ROOT}")
    print(f"\n[다음 단계]")
    print(f"  train_*.py 에서 DATA_DIR를 아래로 변경 후 재학습:")
    print(f"  cicids2017 → cicids2017_gan")


if __name__ == "__main__":
    main()