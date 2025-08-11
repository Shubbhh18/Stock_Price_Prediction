"""
Combined CNN->LSTM model.
Assumes you have an image per time-step (sequence of images), or uses a single image + numeric sequence.
Here we provide an example that accepts sequences of images (B, T, C, H, W).
"""
import torch
import torch.nn as nn
from torchvision import models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, out_dim=512):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)
        num_f = backbone.fc.in_features
        backbone.fc = nn.Linear(num_f, out_dim)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

class CNNLSTMClassifier(nn.Module):
    def __init__(self, cnn_out_dim=512, lstm_hidden=128, lstm_layers=2, num_classes=2):
        super().__init__()
        self.cnn = CNNFeatureExtractor(out_dim=cnn_out_dim)
        self.lstm = nn.LSTM(cnn_out_dim, lstm_hidden, lstm_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, imgs_seq):
        # imgs_seq: (B, T, C, H, W)
        B, T, C, H, W = imgs_seq.shape
        x = imgs_seq.view(B*T, C, H, W)
        feats = self.cnn(x)               # (B*T, feat)
        feats = feats.view(B, T, -1)      # (B, T, feat)
        out, _ = self.lstm(feats)         # (B, T, hidden)
        last = out[:, -1, :]
        return self.classifier(last)
