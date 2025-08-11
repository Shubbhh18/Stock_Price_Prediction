"""
LSTM model for forecasting numeric sequences (OHLCV sequences).
Inputs: (batch, seq_len, features)
Output: next-step regression (single value) by default
"""
import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.1, out_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_size)
        )

    def forward(self, x):
        # x shape: (B, T, input_size)
        out, (h, c) = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)
