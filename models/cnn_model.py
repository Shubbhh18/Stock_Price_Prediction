"""
Simple CNN-based classifier using ResNet18 backbone (PyTorch).
Provides helper to build, save, load and a basic training loop.
"""
import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes=2, pretrained=True, dropout=0.3):
    # Backwards compatible: choose weights object if available
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )
    return model

def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_model(path: str, num_classes=2, device='cpu'):
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Simple training loop (example)
def train_model(model, train_loader, val_loader=None, epochs=5, lr=1e-4, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device); labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            _, top = torch.max(preds, 1)
            total += labels.size(0)
            correct += (top == labels).sum().item()
        print(f"Epoch {ep}/{epochs} - train_loss: {running/total:.4f}, acc: {correct/total:.4f}")
        # optional: add validation loop
