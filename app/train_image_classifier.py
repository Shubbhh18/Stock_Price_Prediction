"""
Simple training script for the image classifier using an index CSV/XLSX.
CSV/XLSX schema: columns 'image' (relative filename) and 'label' (0/1).
Usage:
  .venv\Scripts\python -m app.train_image_classifier --images_dir data/raw/Patterns --index data/Patterns.csv --epochs 3 --out models/cnn_latest.pth
"""
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn

from models.cnn_model import build_resnet18, save_model
from utils.data_processing import ImageLabelDataset


def train(images_dir: Path, index_path: Path, out_path: Path, epochs: int = 3, batch_size: int = 32, lr: float = 1e-4, device: str = 'cpu', num_classes: int = 2, image_size: int = 224, image_col: str = 'image', label_col: str = 'label'):
    ds = ImageLabelDataset(str(images_dir), str(index_path), image_col=image_col, label_col=label_col, image_size=image_size, train=True)
    n_total = len(ds)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    # Make val transforms deterministic
    val_ds.dataset.transform = ImageLabelDataset(str(images_dir), str(index_path), image_col=image_col, label_col=label_col, image_size=image_size, train=False).transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_resnet18(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train(); running=0.0; total=0; correct=0
        for images, labels in train_loader:
            images = images.to(device); labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * images.size(0)
            _, top = torch.max(preds, 1)
            total += labels.size(0)
            correct += (top == labels).sum().item()
        train_loss = running/total; train_acc = correct/total

        # Val
        model.eval(); v_loss=0.0; v_total=0; v_correct=0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device); labels = labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                v_loss += loss.item() * images.size(0)
                _, top = torch.max(preds, 1)
                v_total += labels.size(0)
                v_correct += (top == labels).sum().item()
        val_loss = v_loss/v_total; val_acc = v_correct/v_total
        print(f"Epoch {ep}/{epochs} - train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out_path))
    print(f"Saved model to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', required=True)
    p.add_argument('--index', required=True, help='CSV/XLSX containing columns image,label')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='models/cnn_latest.pth')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--image_col', default='image')
    p.add_argument('--label_col', default='label')
    args = p.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(Path(args.images_dir), Path(args.index), Path(args.out), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, image_size=args.image_size, image_col=args.image_col, label_col=args.label_col)

if __name__ == '__main__':
    main()


