"""
Autoencoder 기반 feature 압축을 담당하는 파일.

스케일링된 원본 feature를 latent_dim 차원으로 압축하여, 논문식 AE latent feature
공간에서도 증강 기법을 비교할 수 있게 한다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """
    입력 feature를 latent 벡터로 압축한 뒤 다시 복원하는 간단한 Autoencoder.

    encoder 출력은 증강 및 Random Forest 학습에 사용할 latent feature로 활용된다.
    """

    def __init__(self, n_features: int, latent_dim: int):
        """입력 feature 수와 latent 차원을 받아 인코더/디코더 구조를 만든다."""
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
        """입력 x를 latent 벡터로 압축한 뒤 복원 결과를 반환한다."""
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(
    X_train: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Autoencoder:
    """
    학습 세트 feature만 사용해 Autoencoder를 학습한다.

    테스트 세트는 학습에 사용하지 않으므로 feature 압축 과정에서도 데이터 누수를 막는다.
    """
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
    """
    학습된 Autoencoder의 인코더만 사용해 입력 feature를 latent feature로 변환한다.

    학습/테스트 데이터 모두 같은 인코더를 사용하지만, 인코더 학습은 학습 세트에서만 수행된다.
    """
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
