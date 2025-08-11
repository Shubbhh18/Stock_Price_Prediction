"""
Numeric preprocessing: build sequences from OHLCV data,
and compute simple indicators (SMA, RSI).
Also includes a simple image dataset for training an image classifier from a CSV/XLSX index.
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def build_sequences_from_ohlc(df: pd.DataFrame, seq_len=50, feature_cols=None, target_col='close'):
    """
    df: must contain columns ['open','high','low','close','volume'] (or subset)
    returns X (N, seq_len, features), y (N,) where y is next close price
    """
    if feature_cols is None:
        feature_cols = ['open','high','low','close','volume']
    arr = df[feature_cols].values
    X, y = [], []
    for i in range(len(arr) - seq_len - 1):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len][feature_cols.index('close')])
    return np.array(X), np.array(y)

def SMA(series: pd.Series, n=14):
    return series.rolling(n).mean()

def RSI(series: pd.Series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / (loss + 1e-8)
    return 100 - 100/(1+rs)


class ImageLabelDataset(Dataset):
    """
    Torch Dataset that reads image paths and integer labels from a CSV/XLSX file.
    Columns expected by default: 'image' (relative path or filename), 'label' (0/1).
    """
    def __init__(self, images_root: str, index_path: str, image_col: str = 'image', label_col: str = 'label', image_size: int = 224, train: bool = True):
        self.images_root = images_root
        self.df = pd.read_csv(index_path) if index_path.lower().endswith('.csv') else pd.read_excel(index_path)
        self.image_col = image_col
        self.label_col = label_col
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row[self.image_col])
        if not img_path.lower().startswith(str(self.images_root).lower()):
            import os
            img_path = os.path.join(self.images_root, img_path)
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        y = int(row[self.label_col])
        return x, y
