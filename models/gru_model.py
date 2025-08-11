"""
GRU variant for time-series forecasting.
"""
import torch
import torch.nn as nn

class GRUForecast(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.1, out_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, out_size)
        )

    def forward(self, x):
        out, h = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)
