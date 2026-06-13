from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        hidden = max(64, min(256, n_features * 2))
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(
    X_train: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Autoencoder:
    model = Autoencoder(X_train.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"[AE] train samples={len(X_train):,} latent_dim={latent_dim} epochs={epochs}")
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[AE] epoch={epoch:03d}/{epochs} loss={np.mean(losses):.6f}")
    return model.eval()


def encode_features(model: Autoencoder, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    encoded = []
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (batch,) in loader:
            encoded.append(model.encoder(batch.to(device)).cpu().numpy())
    return np.vstack(encoded).astype(np.float32)
