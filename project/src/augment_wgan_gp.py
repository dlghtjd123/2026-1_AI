from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from augment_gan import BOT_CLASS_ID, Generator, combine_generated_bot


class Critic(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def compute_wgan_gradient_penalty(
    critic: Critic,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    lambda_gp: float,
    device: torch.device,
) -> torch.Tensor:
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
    bot_mask = y_train == class_id
    current = int(bot_mask.sum())
    n_generate = target_count - current
    if n_generate <= 0:
        return X_train, y_train, None, None
    if current < 2:
        raise ValueError("WGAN-GP requires at least 2 Bot samples.")

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
    from bot_augmentation_experiment import parse_args, run_training

    args = parse_args()
    if "--augments" not in sys.argv:
        args.augments = ["wgan_gp"]
    run_training(args)


if __name__ == "__main__":
    main()
