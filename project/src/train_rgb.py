# train_cnn_lstm.py
import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=64,
            kernel_size=3
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq, feature)
        x = x.permute(0, 2, 1)   # → (batch, feature, seq)

        x = self.conv1(x)        # (batch, 64, seq-2)
        x = x.permute(0, 2, 1)   # (batch, seq, 64)

        _, (h, _) = self.lstm(x)

        out = self.fc(h[-1])
        return self.sigmoid(out)