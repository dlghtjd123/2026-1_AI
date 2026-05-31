"""
augment_utils.py

K-fold 내부에서 train fold에만 증강 적용하는 유틸리티

수정 사항:
  - Generator 출력: Tanh → Sigmoid  (augment_gan/wcgan_gp와 일치)
  - TARGET_RATIO: 0.0025 → 0.1  (SMOTE와 동일, 공정 비교)
  - 다중 세그먼트 Generator 지원  (새 저장 형식: generators 리스트)
  - 구버전 단일 model_state_dict 형식 하위 호환 유지

사용법:
  from augment_utils import augment_train_fold

  for fold, (train_idx, val_idx) in enumerate(kf.split(...)):
      X_train, y_train = augment_train_fold(
          X_all[train_idx], y_all[train_idx],
          augment=AUGMENT, dataset=DATASET, data_root=DATA_ROOT,
      )
      X_val, y_val = X_all[val_idx], y_all[val_idx]  # 원본 유지
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

TARGET_RATIO = 0.1   # 하위 호환용 (내부에서 직접 사용 안 함)
DEFAULT_AUGMENT_MULTIPLIER = 2.0  # 증강 후 봇넷 수 = 원본 봇넷 × multiplier
GAN_EPOCHS = int(os.environ.get("FOLD_GAN_EPOCHS", "500"))
WCGAN_EPOCHS = int(os.environ.get("FOLD_WCGAN_EPOCHS", "500"))
GAN_BATCH_SIZE = 64
GAN_NOISE_DIM = 100
WCGAN_LABEL_DIM = 16
WCGAN_N_CRITIC = 5
WCGAN_LAMBDA_GP = 10.0
MIN_SEG_SIZE = 30


# =========================================================
# Benign 서브샘플링 (학습 속도 향상)
# =========================================================
def subsample_benign(
    X: np.ndarray,
    y: np.ndarray,
    max_normal: int = 0,
    max_mismatch: float = 10.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    봇넷 샘플 전부 유지, Benign만 줄여서 학습 속도 향상.
    test set은 절대 건드리지 말 것 — trainval에만 적용.

    Args:
        max_normal:    정상 샘플 최대 개수 (0=제한 없음, 상한 cap 역할)
        max_mismatch:  train 봇넷 비율이 원본 비율의 최대 N배까지 허용
                       (기본값: 10 → 모든 데이터셋에 일관 적용)

    계산 방식:
        원본 봇넷 비율 × max_mismatch = 허용 최대 train 봇넷 비율
        → 해당 비율을 넘지 않는 정상 샘플 수 계산
        → max_normal이 설정된 경우 둘 중 작은 값 적용
    """
    bot_idx   = np.where(y == 1)[0]
    nor_idx   = np.where(y == 0)[0]
    n_bot     = len(bot_idx)
    orig_ratio = n_bot / len(y)   # 원본 봇넷 비율 (val 비율 근사)

    # mismatch cap 기준 최대 정상 수 계산
    max_ratio = min(orig_ratio * max_mismatch, 0.5)
    n_normal_by_mismatch = int(n_bot * (1 - max_ratio) / max_ratio)

    # max_normal이 설정된 경우 둘 중 작은 값 (더 엄격한 제한)
    if max_normal > 0:
        n_normal_target = min(max_normal, n_normal_by_mismatch)
    else:
        n_normal_target = n_normal_by_mismatch

    if len(nor_idx) <= n_normal_target:
        print(f"  [SUBSAMPLE] Benign {len(nor_idx):,}개 ≤ 목표 {n_normal_target:,} → 전체 사용")
        return X, y

    rng            = np.random.RandomState(random_state)
    nor_idx_subset = rng.choice(nor_idx, n_normal_target, replace=False)
    keep           = np.sort(np.concatenate([bot_idx, nor_idx_subset]))

    X_sub, y_sub = X[keep], y[keep]
    print(f"  [SUBSAMPLE] Benign {len(nor_idx):,} → {n_normal_target:,}개  "
          f"(Bot {n_bot:,} 전부 유지)  "
          f"비율={y_sub.mean():.4f}  mismatch≤{max_mismatch:.0f}x  총={len(y_sub):,}개")
    return X_sub, y_sub


# =========================================================
# SMOTE
# =========================================================
def _smote(
    X: np.ndarray,
    y: np.ndarray,
    augment_multiplier: float = DEFAULT_AUGMENT_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE

    n_current  = int(y.sum())
    n_target   = int(n_current * augment_multiplier)

    if n_target <= n_current:
        print(f"    [AUG] SMOTE 불필요 (현재 Bot={n_current:,} >= 목표 {n_target:,})")
        return X, y

    smote = SMOTE(
        sampling_strategy={1: n_target},
        random_state=42,
        k_neighbors=min(5, n_current - 1),
    )
    X_aug, y_aug = smote.fit_resample(X, y)
    print(f"    [AUG] SMOTE: {n_current:,} → {y_aug.sum():,} Bot  "
          f"(+{int(y_aug.sum()) - n_current:,})  비율={y_aug.mean():.4f}")
    return X_aug.astype(np.float32), y_aug.astype(np.int32)


# =========================================================
# Generator 클래스 정의
# =========================================================
def _make_gan_generator(noise_dim: int, n_features: int):
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
                nn.Linear(256, 512),       nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
                nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
                nn.Linear(256, n_features),
                nn.Sigmoid(),   # [0,1] — MinMaxScaler 범위 일치
            )
        def forward(self, z):
            return self.net(z)

    return Generator()


def _make_wcgan_generator(noise_dim: int, label_dim: int, n_features: int):
    import torch
    import torch.nn as nn

    class ConditionalGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_emb = nn.Embedding(2, label_dim)
            self.net = nn.Sequential(
                nn.Linear(noise_dim + label_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
                nn.Linear(256, 512),                   nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512),                   nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
                nn.Linear(512, 256),                   nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
                nn.Linear(256, n_features),
                nn.Sigmoid(),   # [0,1] — MinMaxScaler 범위 일치
            )
        def forward(self, z, labels):
            return self.net(torch.cat([z, self.label_emb(labels)], dim=1))

    return ConditionalGenerator()


def _segment_botnet(X_bot: np.ndarray, n_segments: int = 4) -> list[np.ndarray]:
    if len(X_bot) < MIN_SEG_SIZE * 2:
        return [X_bot]

    try:
        from sklearn.cluster import KMeans

        k = max(2, min(n_segments, len(X_bot) // MIN_SEG_SIZE))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_bot)
        segments = [X_bot[labels == i] for i in range(k) if np.sum(labels == i) >= MIN_SEG_SIZE]
        return segments if segments else [X_bot]
    except Exception:
        return [X_bot]


def _make_discriminator(n_features: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(n_features, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
        nn.Linear(256, 128),        nn.LeakyReLU(0.2), nn.Dropout(0.3),
        nn.Linear(128, 1),
    )


def _make_wcgan_critic(label_dim: int, n_features: int):
    import torch
    import torch.nn as nn

    class ConditionalCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_emb = nn.Embedding(2, label_dim)
            self.net = nn.Sequential(
                nn.Linear(n_features + label_dim, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Linear(512, 256),                    nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Linear(256, 128),                    nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x, labels):
            return self.net(torch.cat([x, self.label_emb(labels)], dim=1)).squeeze(1)

    return ConditionalCritic()


def _train_fold_gan_generator(X_seg: np.ndarray, device, seg_idx: int):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    n_features = X_seg.shape[1]
    generator = _make_gan_generator(GAN_NOISE_DIM, n_features).to(device)
    discriminator = _make_discriminator(n_features).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    batch_size = min(GAN_BATCH_SIZE, max(2, len(X_seg)))
    loader = DataLoader(
        TensorDataset(torch.tensor(X_seg, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(X_seg) >= batch_size * 2,
    )

    print(f"    [GAN train seg={seg_idx}] real={len(X_seg):,} epochs={GAN_EPOCHS}")
    for _ in range(GAN_EPOCHS):
        for (x_real,) in loader:
            x_real = x_real.to(device)
            bs = x_real.size(0)

            z = torch.randn(bs, GAN_NOISE_DIM, device=device)
            x_fake = generator(z).detach()
            loss_d = (
                criterion(discriminator(x_real).squeeze(1), torch.ones(bs, device=device)) +
                criterion(discriminator(x_fake).squeeze(1), torch.zeros(bs, device=device))
            ) / 2
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            z = torch.randn(bs, GAN_NOISE_DIM, device=device)
            loss_g = criterion(discriminator(generator(z)).squeeze(1), torch.ones(bs, device=device))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

    return generator.eval()


def _compute_gradient_penalty(critic, real, fake, labels, device):
    import torch

    bs = real.size(0)
    alpha = torch.rand(bs, 1, device=device).expand_as(real)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp, labels)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    return WCGAN_LAMBDA_GP * ((grads.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()


def _train_fold_wcgan_generator(X_seg: np.ndarray, device, seg_idx: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    n_features = X_seg.shape[1]
    generator = _make_wcgan_generator(GAN_NOISE_DIM, WCGAN_LABEL_DIM, n_features).to(device)
    critic = _make_wcgan_critic(WCGAN_LABEL_DIM, n_features).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

    batch_size = min(GAN_BATCH_SIZE, max(2, len(X_seg)))
    loader = DataLoader(
        TensorDataset(torch.tensor(X_seg, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(X_seg) >= batch_size * 2,
    )

    print(f"    [WCGAN-GP train seg={seg_idx}] real={len(X_seg):,} epochs={WCGAN_EPOCHS}")
    for _ in range(WCGAN_EPOCHS):
        for (x_real,) in loader:
            x_real = x_real.to(device)
            bs = x_real.size(0)
            labels = torch.ones(bs, dtype=torch.long, device=device)

            for _critic_step in range(WCGAN_N_CRITIC):
                z = torch.randn(bs, GAN_NOISE_DIM, device=device)
                x_fake = generator(z, labels).detach()
                gp = _compute_gradient_penalty(critic, x_real, x_fake, labels, device)
                loss_d = -critic(x_real, labels).mean() + critic(x_fake, labels).mean() + gp
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

            z = torch.randn(bs, GAN_NOISE_DIM, device=device)
            loss_g = -critic(generator(z, labels), labels).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

    return generator.eval()


def _cache_path(data_root: Path, dataset: str, augment: str, fold_id: str,
                n_current: int, n_generate: int, n_features: int,
                augment_multiplier: float) -> Path:
    cache_dir = data_root / f"{dataset}_{augment}_fold_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_tag = (
        f"e{GAN_EPOCHS}" if augment == "gan"
        else f"e{WCGAN_EPOCHS}_c{WCGAN_N_CRITIC}_gp{WCGAN_LAMBDA_GP:g}"
    )
    return cache_dir / (
        f"{fold_id}_bot{n_current}_gen{n_generate}_feat{n_features}"
        f"_mul{augment_multiplier:g}_{train_tag}.npy"
    )


def _generate_from_fold_generators(generators, segment_sizes, n_generate, conditional, device):
    import torch

    total = sum(segment_sizes)
    generated = []
    allocated = 0
    with torch.no_grad():
        for i, (generator, seg_size) in enumerate(zip(generators, segment_sizes)):
            n_gen_i = int(n_generate * seg_size / total)
            if i == len(generators) - 1:
                n_gen_i = n_generate - allocated
            allocated += n_gen_i
            if n_gen_i <= 0:
                continue

            chunks = []
            for start in range(0, n_gen_i, 1024):
                bs = min(1024, n_gen_i - start)
                z = torch.randn(bs, GAN_NOISE_DIM, device=device)
                if conditional:
                    labels = torch.ones(bs, dtype=torch.long, device=device)
                    chunks.append(generator(z, labels).cpu().numpy())
                else:
                    chunks.append(generator(z).cpu().numpy())
            generated.append(np.vstack(chunks))

    return np.vstack(generated).astype(np.float32)


def _fold_gan_augment(
    X: np.ndarray,
    y: np.ndarray,
    augment: str,
    dataset: str,
    data_root: Path,
    fold_id: str,
    augment_multiplier: float = DEFAULT_AUGMENT_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    n_current = int(y.sum())
    n_target = int(n_current * augment_multiplier)
    n_generate = n_target - n_current
    if n_generate <= 0:
        print(f"    [AUG] {augment.upper()} 불필요 (현재 Bot={n_current:,})")
        return X, y

    cache_file = _cache_path(
        data_root, dataset, augment, fold_id, n_current, n_generate, X.shape[1],
        augment_multiplier,
    )
    if cache_file.exists():
        X_fake = np.load(cache_file)
        print(f"    [AUG] {augment.upper()} fold-local cache 사용: {cache_file.name}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = 42 if fold_id == "final" else 42 + sum(ord(ch) for ch in fold_id)
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_bot = X[y == 1].astype(np.float32)
        segments = _segment_botnet(X_bot)
        conditional = augment in ("wgan_gp", "wcgan_gp")
        print(f"    [AUG] {augment.upper()}: fold-local Generator 학습 "
              f"(fold={fold_id}, segments={len(segments)}, fake={n_generate:,})")

        if conditional:
            generators = [
                _train_fold_wcgan_generator(seg, device, i)
                for i, seg in enumerate(segments)
            ]
        else:
            generators = [
                _train_fold_gan_generator(seg, device, i)
                for i, seg in enumerate(segments)
            ]

        X_fake = _generate_from_fold_generators(
            generators,
            [len(seg) for seg in segments],
            n_generate,
            conditional,
            device,
        )
        np.save(cache_file, X_fake)
        print(f"    [AUG] fold-local fake 저장: {cache_file}")

    X_aug = np.vstack([X, X_fake]).astype(np.float32)
    y_aug = np.concatenate([y, np.ones(len(X_fake), dtype=np.int32)])
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]
    print(f"    [AUG] {augment.upper()}: {n_current:,} → {y_aug.sum():,} Bot "
          f"(+{len(X_fake):,})  비율={y_aug.mean():.4f}")
    return X_aug, y_aug


# =========================================================
# GAN / WCGAN-GP 증강
# 새 형식: {"generators": [state_dict, ...], "segment_sizes": [...], ...}
# 구 형식: {"model_state_dict": state_dict, ...}  (하위 호환)
# =========================================================
def _gan_augment(
    X: np.ndarray,
    y: np.ndarray,
    gen_path: Path,
    conditional: bool = False,
    augment_multiplier: float = DEFAULT_AUGMENT_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if not gen_path.exists():
        print(f"    [AUG] Generator 없음: {gen_path} → 증강 스킵")
        return X, y
    print("    [AUG] 주의: GAN 계열 Generator는 trainval 사전학습본을 사용합니다. "
          "holdout test는 원본 유지, CV는 보조 검증으로 해석하세요.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(gen_path, map_location=device)

    noise_dim  = ckpt["noise_dim"]
    n_features = ckpt["n_features"]
    label_dim  = ckpt.get("label_dim", 16)

    n_current  = int(y.sum())
    n_target   = int(n_current * augment_multiplier)
    n_generate = n_target - n_current

    if n_generate <= 0:
        print(f"    [AUG] GAN 불필요 (현재 Bot={n_current:,} >= 목표 {n_target:,})")
        return X, y

    # ── 새 형식: 다중 세그먼트 Generator ────────────────────
    if "generators" in ckpt:
        state_dicts   = ckpt["generators"]
        segment_sizes = ckpt["segment_sizes"]
        total_seg     = sum(segment_sizes)

        all_fake = []
        for i, (state_dict, seg_size) in enumerate(zip(state_dicts, segment_sizes)):
            n_gen_i = int(n_generate * seg_size / total_seg)
            if i == len(state_dicts) - 1:
                n_gen_i = n_generate - sum(
                    int(n_generate * s / total_seg) for s in segment_sizes[:-1]
                )
            if n_gen_i <= 0:
                continue

            if conditional:
                G = _make_wcgan_generator(noise_dim, label_dim, n_features).to(device)
            else:
                G = _make_gan_generator(noise_dim, n_features).to(device)
            G.load_state_dict(state_dict)
            G.eval()

            seg_samples = []
            with torch.no_grad():
                for start in range(0, n_gen_i, 1024):
                    bs = min(1024, n_gen_i - start)
                    z  = torch.randn(bs, noise_dim, device=device)
                    if conditional:
                        labels = torch.ones(bs, dtype=torch.long, device=device)
                        seg_samples.append(G(z, labels).cpu().numpy())
                    else:
                        seg_samples.append(G(z).cpu().numpy())
            all_fake.append(np.vstack(seg_samples))

        X_fake = np.vstack(all_fake).astype(np.float32)

    # ── 구 형식: 단일 Generator (하위 호환) ─────────────────
    else:
        if conditional:
            G = _make_wcgan_generator(noise_dim, label_dim, n_features).to(device)
        else:
            G = _make_gan_generator(noise_dim, n_features).to(device)
        G.load_state_dict(ckpt["model_state_dict"])
        G.eval()

        samples = []
        with torch.no_grad():
            for start in range(0, n_generate, 1024):
                bs = min(1024, n_generate - start)
                z  = torch.randn(bs, noise_dim, device=device)
                if conditional:
                    labels = torch.ones(bs, dtype=torch.long, device=device)
                    samples.append(G(z, labels).cpu().numpy())
                else:
                    samples.append(G(z).cpu().numpy())
        X_fake = np.vstack(samples).astype(np.float32)

    # 합치기
    X_aug = np.vstack([X, X_fake]).astype(np.float32)
    y_aug = np.concatenate([y, np.ones(len(X_fake), dtype=np.int32)])
    idx   = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]

    gan_type = "WCGAN-GP" if conditional else "GAN"
    print(f"    [AUG] {gan_type}: {n_current:,} → {y_aug.sum():,} Bot  "
          f"(+{len(X_fake):,})  비율={y_aug.mean():.4f}")
    return X_aug, y_aug


# =========================================================
# 공개 인터페이스
# =========================================================
def augment_train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    augment: str,
    dataset: str,
    data_root: Path,
    fold_id: str | int | None = None,
    augment_multiplier: float = DEFAULT_AUGMENT_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-fold 내부에서 train fold에만 증강 적용.
    val fold는 호출하지 않아서 항상 원본 유지.

    Args:
        X_train:   train fold 피처 (n, 32) flat 또는 (n, 32, 1) seq
        y_train:   train fold 라벨
        augment:   증강 방식 ('none'/'smote'/'gan'/'wgan_gp'/'wcgan_gp')
        dataset:   데이터셋 이름 ('cicids2017'/'cicids2018'/'ctu13')
        data_root: data/processed 경로
        fold_id:   fold-local GAN/WCGAN 학습 및 캐시 구분자
        augment_multiplier: 증강 후 목표 Bot 수 배수 (예: 2, 5, 10)

    Returns:
        증강된 X_train, y_train
        (seq 입력이면 증강 후 다시 seq shape으로 반환)
    """
    if augment == "none":
        return X_train, y_train

    # seq (n, 32, 1) → flat (n, 32)
    is_seq = X_train.ndim == 3
    if is_seq:
        n_feat = X_train.shape[1]
        X_flat = X_train.reshape(-1, n_feat)
    else:
        X_flat = X_train
        n_feat = X_flat.shape[1]

    print(f"  [AUG] train: {len(y_train):,}  val: -  "
          f"Bot 비율(원본)={y_train.mean():.6f}  "
          f"증강 목표={augment_multiplier:g}x")

    if augment == "smote":
        X_aug, y_aug = _smote(X_flat, y_train, augment_multiplier=augment_multiplier)

    elif augment == "gan":
        X_aug, y_aug = _fold_gan_augment(
            X_flat, y_train, augment, dataset, data_root,
            fold_id=str(fold_id or "unknown"),
            augment_multiplier=augment_multiplier,
        )

    elif augment in ("wgan_gp", "wcgan_gp"):
        X_aug, y_aug = _fold_gan_augment(
            X_flat, y_train, "wcgan_gp", dataset, data_root,
            fold_id=str(fold_id or "unknown"),
            augment_multiplier=augment_multiplier,
        )

    else:
        return X_train, y_train

    # flat → seq 복원
    if is_seq:
        X_aug = X_aug.reshape(-1, n_feat, 1)

    return X_aug, y_aug
