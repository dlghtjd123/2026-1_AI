"""
GAN으로 Bot 클래스를 생성 증강하는 파일.

학습 세트의 실제 Bot 샘플만 사용해 생성기와 판별기를 학습하고,
목표 Bot 개수에 맞춰 생성 Bot 샘플을 만든 뒤 학습 세트에 결합한다.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BOT_CLASS_ID = 3
RANDOM_STATE = 42


class Generator(nn.Module):
    """
    무작위 잡음 벡터를 입력받아 Bot feature와 같은 차원의 생성 샘플을 만든다.
    """

    def __init__(self, noise_dim: int, latent_dim: int):
        """noise_dim 크기의 잡음 벡터를 latent_dim 크기의 feature 벡터로 변환한다."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        """생성기 신경망을 통과시켜 생성 Bot feature를 반환한다."""
        return self.net(noise)


class Discriminator(nn.Module):
    """
    입력 feature가 실제 Bot인지 생성기가 만든 Bot인지 판별한다.
    """

    def __init__(self, latent_dim: int):
        """latent_dim 크기의 feature 벡터를 받아 실제/생성 판별 점수를 출력하도록 구성한다."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """입력 feature에 대한 실제/생성 판별 점수를 반환한다."""
        return self.net(x).squeeze(1)


def combine_generated_bot(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_fake: np.ndarray,
    class_id: int = BOT_CLASS_ID,
) -> tuple[np.ndarray, np.ndarray]:
    """
    생성된 Bot 샘플을 기존 학습 세트에 추가하고 전체 학습 세트 순서를 섞는다.

    순서를 섞는 이유는 뒤쪽에 생성 샘플이 몰려 있는 상태로 학습되는 것을 피하기 위해서이다.
    """
    y_fake = np.full(len(X_fake), class_id, dtype=np.int32)
    X_aug = np.vstack([X_train, X_fake]).astype(np.float32)
    y_aug = np.concatenate([y_train, y_fake]).astype(np.int32)
    rng = np.random.RandomState(RANDOM_STATE)
    order = rng.permutation(len(y_aug))
    return X_aug[order], y_aug[order]


def train_gan(
    X_bot: np.ndarray,
    latent_dim: int,
    noise_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Generator:
    """
    Bot 샘플만 사용해 일반 GAN을 학습한다.

    판별기는 실제 Bot과 생성 Bot을 구분하도록 학습하고,
    생성기는 판별기를 속이는 방향으로 학습한다.
    """
    generator = Generator(noise_dim, latent_dim).to(device)
    discriminator = Discriminator(latent_dim).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    effective_batch = min(batch_size, max(2, len(X_bot)))
    loader = DataLoader(
        TensorDataset(torch.tensor(X_bot, dtype=torch.float32)),
        batch_size=effective_batch,
        shuffle=True,
        drop_last=True,
    )
    if len(loader) == 0:
        loader = DataLoader(
            TensorDataset(torch.tensor(X_bot, dtype=torch.float32)),
            batch_size=len(X_bot),
            shuffle=True,
            drop_last=False,
        )

    print(f"[GAN] Bot samples={len(X_bot):,} epochs={epochs} batch={effective_batch}")
    for epoch in range(1, epochs + 1):
        g_losses: list[float] = []
        d_losses: list[float] = []
        for (real_x,) in loader:
            real_x = real_x.to(device)
            bs = real_x.size(0)

            noise = torch.randn(bs, noise_dim, device=device)
            fake_x = generator(noise).detach()
            loss_d = criterion(discriminator(real_x), torch.ones(bs, device=device)) + criterion(
                discriminator(fake_x), torch.zeros(bs, device=device)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            noise = torch.randn(bs, noise_dim, device=device)
            fake_x = generator(noise)
            loss_g = criterion(discriminator(fake_x), torch.ones(bs, device=device))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
        print(
            f"[GAN] epoch={epoch:03d}/{epochs} "
            f"G={np.mean(g_losses):.5f} D={np.mean(d_losses):.5f}"
        )
    return generator.eval()


def augment_gan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_count: int,
    noise_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    class_id: int = BOT_CLASS_ID,
) -> tuple[np.ndarray, np.ndarray, Generator | None, np.ndarray | None]:
    """
    GAN으로 생성 Bot 샘플을 만들어 Bot 개수를 target_count까지 늘린다.

    생성된 샘플은 성능 평가뿐 아니라 생성 데이터 품질 진단 파일에도 사용된다.
    """
    bot_mask = y_train == class_id
    current = int(bot_mask.sum())
    n_generate = target_count - current
    if n_generate <= 0:
        return X_train, y_train, None, None
    if current < 2:
        raise ValueError("GAN은 Bot 샘플이 최소 2개 이상 필요하다.")

    generator = train_gan(
        X_train[bot_mask],
        latent_dim=X_train.shape[1],
        noise_dim=noise_dim,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )

    generated = []
    with torch.no_grad():
        for start in range(0, n_generate, 2048):
            bs = min(2048, n_generate - start)
            noise = torch.randn(bs, noise_dim, device=device)
            generated.append(generator(noise).cpu().numpy())
    X_fake = np.vstack(generated).astype(np.float32)
    X_aug, y_aug = combine_generated_bot(X_train, y_train, X_fake, class_id=class_id)
    return X_aug, y_aug, generator, X_fake


def main() -> None:
    """이 파일을 직접 실행했을 때 GAN 증강만 수행하도록 전체 실험 루프를 호출한다."""
    from bot_augmentation_experiment import parse_args, run_training

    args = parse_args()
    if "--augments" not in sys.argv:
        args.augments = ["gan"]
    run_training(args)


if __name__ == "__main__":
    main()
