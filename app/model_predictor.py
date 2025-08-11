"""
Lightweight inference helpers.
If you don't have trained models yet, this module falls back to 'visual rule' or simple heuristics.
"""
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from utils.image_preprocessing import get_transforms

# Placeholder mapping for labels
CLASS_MAP = {0: "bearish", 1: "bullish"}

def _resolve_device(device: str):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def load_cnn_model(path: Path, device='cpu', num_classes=2):
    from models.cnn_model import load_model
    resolved = _resolve_device(device)
    return load_model(str(path), num_classes=num_classes, device=resolved)

def predict_image(model, image_path: Path, device='cpu', image_size=224):
    resolved = _resolve_device(device)
    model.to(resolved)
    model.eval()
    transform = get_transforms(image_size=image_size, train=False)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(resolved)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return {"label": CLASS_MAP.get(pred, str(pred)), "probabilities": probs.tolist()}

def naive_price_forecast(ohlc_df, horizon=1):
    """
    Very simple baseline: next price = last_close * (1 + mean_return)
    """
    if 'close' not in ohlc_df.columns:
        raise ValueError("ohlc_df must contain 'close'")
    returns = ohlc_df['close'].pct_change().dropna()
    mean_r = returns.tail(50).mean() if len(returns) >= 50 else returns.mean()
    last = ohlc_df['close'].iloc[-1]
    pred = last * (1 + mean_r * horizon)
    return float(pred)

# --- Image-only derived return estimation (heuristic) ---
def estimate_return_from_probs(probabilities: np.ndarray, scale: float = 0.01) -> float:
    """
    Map class probabilities to a signed expected return.
    Positive if bullish > bearish, negative otherwise. Scale ~1% by default.
    """
    if probabilities is None or len(probabilities) < 2:
        return 0.0
    return float((probabilities[1] - probabilities[0]) * scale)

def analyze_image(model, image_path: Path, device='cpu', image_size=224, last_price: float | None = None, horizon: int = 1, ohlc_df=None):
    """
    Combined analysis: classify behavior from image and estimate a future price.
    If ohlc_df is provided, use naive_price_forecast. Otherwise, estimate return from class probs.
    If last_price is provided, convert return to absolute future_price.
    """
    result = predict_image(model, image_path, device=device, image_size=image_size)
    probs = np.array(result["probabilities"], dtype=float)
    predicted_return = estimate_return_from_probs(probs)

    future_price = None
    method = "classification_return"
    if ohlc_df is not None:
        future_price = naive_price_forecast(ohlc_df, horizon=horizon)
        method = "naive_from_ohlc"
    elif last_price is not None:
        future_price = float(last_price) * (1.0 + predicted_return * horizon)

    return {
        "behavior": result["label"],
        "probabilities": result["probabilities"],
        "predicted_return": predicted_return,
        "horizon": int(horizon),
        "future_price": future_price,
        "method": method,
    }
