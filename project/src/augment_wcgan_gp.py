"""
augment_wcgan_gp.py

CIC-IDS2017 Bot 클래스 WCGAN-GP 증강

WGAN-GP와의 차이:
  Generator와 Critic 모두 클래스 레이블을 조건으로 받음
  → 레이블 정보를 활용해 더 정확한 클래스별 샘플 생성
  → 봇넷/정상 패턴 구분이 명확한 샘플 생성 가능

GAN과의 차이:
  Wasserstein loss   → 학습 안정성 향상
  Gradient Penalty   → Weight Clipping 대신 사용 (WGAN-GP 방식)
  Conditional        → 레이블 조건부 생성

학습 방식:
  Critic을 N_CRITIC번 학습 후 Generator 1번 학습
  Gradient Penalty: 실제/가짜 샘플 보간점에서 gradient 제약

저장 경로: data/processed/cicids2017_wcgan_gp/
  flat/ X_train.npy  y_train.npy
  seq/  X_train.npy  y_train.npy
  val, test는 원본에서 복사

사용법:
  python augment_wcgan_gp.py
  python train_rf.py       --augment wcgan_gp
  python train_xgb.py      --augment wcgan_gp
  python train_cnn_lstm.py --augment wcgan_gp
  python train_gru.py      --augment wcgan_gp
  python train_cnn_gru.py  --augment wcgan_gp
  python evaluate.py       --augment wcgan_gp
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
SAVE_ROOT = _PROJECT / "data" / "processed" / "cicids2017_wcgan_gp"

SRC_FLAT  = SRC_ROOT / "flat"
SRC_SEQ   = SRC_ROOT / "seq"

SAVE_FLAT = SAVE_ROOT / "flat"
SAVE_SEQ  = SAVE_ROOT / "seq"


# =========================================================
# 하이퍼파라미터
# =========================================================
NOISE_DIM    = 100      # Generator 입력 노이즈 차원
LABEL_DIM    = 16       # 레이블 임베딩 차원
N_EPOCHS     = 1000     # 학습 에포크 (WGAN 계열은 더 많이 필요)
BATCH_SIZE   = 256      # 배치 크기
LR_G         = 1e-4     # Generator 학습률
LR_D         = 1e-4     # Critic 학습률
N_CRITIC     = 5        # Generator 1회당 Critic 학습 횟수
LAMBDA_GP    = 10       # Gradient Penalty 가중치
TARGET_RATIO = 0.5      # 증강 후 목표 Bot 비율
RANDOM_STATE = 42


# =========================================================
# 모델 정의
# =========================================================
class ConditionalGenerator(nn.Module):
    """
    노이즈 + 클래스 레이블 → Bot 플로우 생성

    입력: noise (batch, NOISE_DIM) + label (batch,)
    출력: (batch, n_features)
    """
    def __init__(self, noise_dim: int, label_dim: int, n_features: int):
        super().__init__()
        self.label_emb = nn.Embedding(2, label_dim)  # 0=Normal, 1=Bot

        self.net = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, n_features),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_emb(labels)          # (batch, label_dim)
        x = torch.cat([z, label_emb], dim=1)        # (batch, noise_dim + label_dim)
        return self.net(x)


class ConditionalCritic(nn.Module):
    """
    플로우 + 클래스 레이블 → Wasserstein 거리 추정 (스칼라)

    입력: sample (batch, n_features) + label (batch,)
    출력: (batch,) → sigmoid 없음 (Wasserstein)
    """
    def __init__(self, n_features: int, label_dim: int):
        super().__init__()
        self.label_emb = nn.Embedding(2, label_dim)

        self.net = nn.Sequential(
            nn.Linear(n_features + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1),
            # Wasserstein → sigmoid 없음
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_emb(labels)          # (batch, label_dim)
        x = torch.cat([x, label_emb], dim=1)        # (batch, n_features + label_dim)
        return self.net(x).squeeze(1)


# =========================================================
# Gradient Penalty
# =========================================================
def compute_gradient_penalty(
    critic:    ConditionalCritic,
    real:      torch.Tensor,
    fake:      torch.Tensor,
    labels:    torch.Tensor,
    device:    torch.device,
    lambda_gp: float,
) -> torch.Tensor:
    """
    실제/가짜 샘플 사이 보간점에서 gradient norm 제약
    ||∇D(x_hat)||_2 ≈ 1 이 되도록 패널티 부과

    WGAN-GP 핵심 아이디어:
      Weight Clipping 대신 gradient penalty 사용
      → 학습 안정성 ↑, 모드 붕괴 방지
    """
    batch_size = real.size(0)

    # 보간 계수 α ~ Uniform(0, 1)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real)

    # 보간 샘플
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    # Critic 출력
    d_interpolated = critic(interpolated, labels)

    # Gradient 계산
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradient Penalty = λ * E[(||∇D(x_hat)||_2 - 1)^2]
    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    gp = lambda_gp * ((grad_norm - 1) ** 2).mean()

    return gp


# =========================================================
# 학습
# =========================================================
def train_wcgan_gp(
    X_bot:      np.ndarray,
    noise_dim:  int,
    label_dim:  int,
    n_epochs:   int,
    batch_size: int,
    lr_g:       float,
    lr_d:       float,
    n_critic:   int,
    lambda_gp:  float,
    device:     torch.device,
) -> ConditionalGenerator:
    n_features = X_bot.shape[1]

    G = ConditionalGenerator(noise_dim, label_dim, n_features).to(device)
    D = ConditionalCritic(n_features, label_dim).to(device)

    # WGAN-GP: betas=(0.0, 0.9) 권장
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.9))

    # Bot 레이블만 사용 (label=1)
    X_tensor = torch.tensor(X_bot, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"\n[WCGAN-GP] 학습 시작")
    print(f"  Bot 샘플: {len(X_bot):,}  /  n_features: {n_features}")
    print(f"  Epochs: {n_epochs}  /  Batch: {batch_size}  /  N_Critic: {n_critic}")
    print(f"  Lambda_GP: {lambda_gp}  /  Device: {device}")

    for epoch in range(1, n_epochs + 1):
        g_losses, d_losses, gp_losses = [], [], []

        for (X_real,) in loader:
            X_real  = X_real.to(device)
            bs      = X_real.size(0)

            # 레이블: 전부 Bot(1)
            real_labels = torch.ones(bs, dtype=torch.long, device=device)

            # ── Critic N_CRITIC회 학습 ──────────────────────
            for _ in range(n_critic):
                z      = torch.randn(bs, noise_dim, device=device)
                X_fake = G(z, real_labels).detach()

                # Wasserstein loss
                loss_real = -D(X_real, real_labels).mean()
                loss_fake =  D(X_fake, real_labels).mean()

                # Gradient Penalty
                gp = compute_gradient_penalty(
                    D, X_real, X_fake, real_labels, device, lambda_gp
                )

                loss_D = loss_real + loss_fake + gp

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                d_losses.append((loss_real + loss_fake).item())
                gp_losses.append(gp.item())

            # ── Generator 1회 학습 ──────────────────────────
            z      = torch.randn(bs, noise_dim, device=device)
            X_fake = G(z, real_labels)

            # Generator는 Critic을 속이려 함 (Wasserstein: -E[D(fake)])
            loss_G = -D(X_fake, real_labels).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_losses.append(loss_G.item())

        if epoch % 100 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:4d}/{n_epochs}] "
                  f"G={np.mean(g_losses):+.4f}  "
                  f"D={np.mean(d_losses):+.4f}  "
                  f"GP={np.mean(gp_losses):.4f}")

    print("[WCGAN-GP] 학습 완료")
    return G


# =========================================================
# 샘플 생성
# =========================================================
def generate_samples(
    G:          ConditionalGenerator,
    n_samples:  int,
    noise_dim:  int,
    device:     torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    G.eval()
    samples = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end    = min(start + batch_size, n_samples)
            bs     = end - start
            z      = torch.randn(bs, noise_dim, device=device)
            labels = torch.ones(bs, dtype=torch.long, device=device)  # Bot=1
            samples.append(G(z, labels).cpu().numpy())
    return np.vstack(samples).astype(np.float32)


# =========================================================
# main
# =========================================================
def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  WCGAN-GP Augmentation — CIC-IDS2017 Bot 클래스")
    print("=" * 65)
    print(f"  SRC_ROOT     = {SRC_ROOT}")
    print(f"  SAVE_ROOT    = {SAVE_ROOT}")
    print(f"  TARGET_RATIO = {TARGET_RATIO}")
    print(f"  N_EPOCHS     = {N_EPOCHS}")
    print(f"  N_CRITIC     = {N_CRITIC}")
    print(f"  LAMBDA_GP    = {LAMBDA_GP}")
    print(f"  Device       = {device}")
    print("=" * 65)

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
        print(f"\n[INFO] 이미 목표 비율 달성 — 증강 불필요")
        return

    n_generate = n_target - n_current
    print(f"\n[WCGAN-GP] 생성할 Bot 샘플: {n_generate:,}개")
    print(f"[WCGAN-GP] 증강 후 Bot: {n_target:,} / Normal: {n_majority:,}")

    # ── Bot 샘플만 추출 ───────────────────────────────────
    X_bot = X_train[y_train == 1]
    print(f"\n[WCGAN-GP] Bot 학습 데이터: {X_bot.shape}")

    # ── WCGAN-GP 학습 ─────────────────────────────────────
    G = train_wcgan_gp(
        X_bot     = X_bot,
        noise_dim = NOISE_DIM,
        label_dim = LABEL_DIM,
        n_epochs  = N_EPOCHS,
        batch_size= BATCH_SIZE,
        lr_g      = LR_G,
        lr_d      = LR_D,
        n_critic  = N_CRITIC,
        lambda_gp = LAMBDA_GP,
        device    = device,
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
                if src.exists():
                    shutil.copy2(src, d_dst / fname)

    scaler_src = SRC_SEQ / "scaler_flow.pkl"
    if scaler_src.exists():
        shutil.copy2(scaler_src, SAVE_SEQ / "scaler_flow.pkl")

    print(f"\n[COPY] val/test 원본 복사 완료")

    # ── Generator 저장 ────────────────────────────────────
    torch.save(
        {
            "model_state_dict": G.state_dict(),
            "noise_dim":        NOISE_DIM,
            "label_dim":        LABEL_DIM,
            "n_features":       n_feat,
        },
        SAVE_ROOT / "generator_wcgan_gp.pt",
    )

    # ── meta 저장 ─────────────────────────────────────────
    meta = {
        "method":            "WCGAN-GP",
        "target_ratio":      TARGET_RATIO,
        "original_bot":      int(n_current),
        "augmented_bot":     int(y_aug.sum()),
        "generated_samples": int(n_generate),
        "normal_count":      int((y_aug == 0).sum()),
        "total_train":       int(len(y_aug)),
        "bot_ratio_after":   float(y_aug.mean()),
        "n_features":        n_feat,
        "noise_dim":         NOISE_DIM,
        "label_dim":         LABEL_DIM,
        "n_epochs":          N_EPOCHS,
        "batch_size":        BATCH_SIZE,
        "lr_g":              LR_G,
        "lr_d":              LR_D,
        "n_critic":          N_CRITIC,
        "lambda_gp":         LAMBDA_GP,
    }
    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    print(f"\n[DONE] {SAVE_ROOT}")


if __name__ == "__main__":
    main()