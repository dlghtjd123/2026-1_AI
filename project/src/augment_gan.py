from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BOT_CLASS_ID = 3
RANDOM_STATE = 42


class Generator(nn.Module):
    def __init__(self, noise_dim: int, latent_dim: int):
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
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self, latent_dim: int):
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
        return self.net(x).squeeze(1)


def combine_generated_bot(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_fake: np.ndarray,
    class_id: int = BOT_CLASS_ID,
) -> tuple[np.ndarray, np.ndarray]:
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
    bot_mask = y_train == class_id
    current = int(bot_mask.sum())
    n_generate = target_count - current
    if n_generate <= 0:
        return X_train, y_train, None, None
    if current < 2:
        raise ValueError("GAN requires at least 2 Bot samples.")

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
    from bot_augmentation_experiment import parse_args, run_training

    args = parse_args()
    if "--augments" not in sys.argv:
        args.augments = ["gan"]
    run_training(args)


if __name__ == "__main__":
    main()
