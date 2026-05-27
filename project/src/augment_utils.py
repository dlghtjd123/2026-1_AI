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

from pathlib import Path

import numpy as np

TARGET_RATIO = 0.1   # 하위 호환용 (내부에서 직접 사용 안 함)
AUGMENT_MULTIPLIER = 2.0  # 증강 후 봇넷 수 = 원본 봇넷 × AUGMENT_MULTIPLIER


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
def _smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE

    n_current  = int(y.sum())
    n_target   = int(n_current * AUGMENT_MULTIPLIER)

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
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if not gen_path.exists():
        print(f"    [AUG] Generator 없음: {gen_path} → 증강 스킵")
        return X, y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(gen_path, map_location=device)

    noise_dim  = ckpt["noise_dim"]
    n_features = ckpt["n_features"]
    label_dim  = ckpt.get("label_dim", 16)

    n_current  = int(y.sum())
    n_target   = int(n_current * AUGMENT_MULTIPLIER)
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
          f"Bot 비율(원본)={y_train.mean():.6f}")

    if augment == "smote":
        X_aug, y_aug = _smote(X_flat, y_train)

    elif augment == "gan":
        gen_path     = data_root / f"{dataset}_gan" / "generator.pt"
        X_aug, y_aug = _gan_augment(X_flat, y_train, gen_path, conditional=False)

    elif augment in ("wgan_gp", "wcgan_gp"):
        gen_path     = data_root / f"{dataset}_wcgan_gp" / "generator_wcgan_gp.pt"
        X_aug, y_aug = _gan_augment(X_flat, y_train, gen_path, conditional=True)

    else:
        return X_train, y_train

    # flat → seq 복원
    if is_seq:
        X_aug = X_aug.reshape(-1, n_feat, 1)

    return X_aug, y_aug