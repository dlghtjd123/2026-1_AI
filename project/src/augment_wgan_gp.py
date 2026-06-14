"""
WGAN-GP로 Bot 클래스를 생성 증강하는 파일.

일반 GAN보다 안정적인 학습을 위해 Wasserstein 손실과 gradient penalty를 사용한다.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from augment_gan import BOT_CLASS_ID, Generator, combine_generated_bot


class Critic(nn.Module):
    """
    WGAN-GP에서 샘플의 실제/생성 정도를 점수로 출력하는 critic 신경망.

    일반 GAN의 판별기처럼 확률을 직접 출력하지 않고 Wasserstein 점수 계산에 사용된다.
    """

    def __init__(self, latent_dim: int):
        """latent_dim 크기의 feature 벡터를 받아 critic 점수 하나를 출력하도록 구성한다."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """입력 feature에 대한 critic 점수를 반환한다."""
        return self.net(x).squeeze(1)


def compute_wgan_gradient_penalty(
    critic: Critic,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    lambda_gp: float,
    device: torch.device,
) -> torch.Tensor:
    """
    WGAN-GP의 gradient penalty 항을 계산한다.

    실제 샘플과 생성 샘플 사이를 보간한 지점에서 gradient norm이 1에 가깝도록
    제약하여 critic 학습을 안정화한다.
    """
    bs = real_x.size(0)
    alpha = torch.rand(bs, 1, device=device).expand_as(real_x)
    interpolated = (alpha * real_x + (1.0 - alpha) * fake_x).requires_grad_(True)
    score = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    return lambda_gp * ((gradients.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(
    X_bot: np.ndarray,
    latent_dim: int,
    noise_dim: int,
    epochs: int,
    batch_size: int,
    n_critic: int,
    lambda_gp: float,
    device: torch.device,
) -> Generator:
    """
    Bot 샘플만 사용해 WGAN-GP 생성기를 학습한다.

    critic을 n_critic번 더 자주 업데이트한 뒤 생성기를 업데이트하는 방식으로
    Wasserstein GAN 학습 절차를 따른다.
    """
    generator = Generator(noise_dim, latent_dim).to(device)
    critic = Critic(latent_dim).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9))

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

    print(f"[WGAN-GP] Bot samples={len(X_bot):,} epochs={epochs} batch={effective_batch}")
    for epoch in range(1, epochs + 1):
        g_losses: list[float] = []
        c_losses: list[float] = []
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)
            for _ in range(n_critic):
                noise = torch.randn(bs, noise_dim, device=device)
                fake = generator(noise).detach()
                gp = compute_wgan_gradient_penalty(critic, real_batch, fake, lambda_gp, device)
                loss_c = critic(fake).mean() - critic(real_batch).mean() + gp
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()

            noise = torch.randn(bs, noise_dim, device=device)
            fake = generator(noise)
            loss_g = -critic(fake).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            g_losses.append(loss_g.item())
            c_losses.append(loss_c.item())
        print(
            f"[WGAN-GP] epoch={epoch:03d}/{epochs} "
            f"G={np.mean(g_losses):.5f} C={np.mean(c_losses):.5f}"
        )
    return generator.eval()


def augment_wgan_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_count: int,
    noise_dim: int,
    epochs: int,
    batch_size: int,
    n_critic: int,
    lambda_gp: float,
    device: torch.device,
    class_id: int = BOT_CLASS_ID,
) -> tuple[np.ndarray, np.ndarray, Generator | None, np.ndarray | None]:
    """
    WGAN-GP로 생성 Bot 샘플을 만들어 Bot 개수를 target_count까지 늘린다.

    반환되는 생성 Bot 배열은 생성 데이터 품질 진단에 사용된다.
    """
    bot_mask = y_train == class_id
    current = int(bot_mask.sum())
    n_generate = target_count - current
    if n_generate <= 0:
        return X_train, y_train, None, None
    if current < 2:
        raise ValueError("WGAN-GP는 Bot 샘플이 최소 2개 이상 필요하다.")

    X_bot = X_train[bot_mask]
    generator = train_wgan_gp(
        X_bot,
        latent_dim=X_train.shape[1],
        noise_dim=noise_dim,
        epochs=epochs,
        batch_size=batch_size,
        n_critic=n_critic,
        lambda_gp=lambda_gp,
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
    """이 파일을 직접 실행했을 때 WGAN-GP 증강만 수행하도록 전체 실험 루프를 호출한다."""
    from bot_augmentation_experiment import parse_args, run_training

    args = parse_args()
    if "--augments" not in sys.argv:
        args.augments = ["wgan_gp"]
    run_training(args)


if __name__ == "__main__":
    main()
