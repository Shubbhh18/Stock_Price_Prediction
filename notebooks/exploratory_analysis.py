# exploratory analysis starter
# 1) load data
from data.sample_data_loader import load_csv, DATA_ROOT
import pandas as pd
meta = load_csv("Meta.csv")
patterns = load_csv("Patterns.csv")
seg = load_csv("Segmentation.csv")
print(meta.head())
# 2) quick image display
from PIL import Image
img_path = DATA_ROOT / seg['Path'].iloc[0]
Image.open(img_path).resize((300,200))
# 3) build sequence examples, compute indicators
from utils.data_processing import SMA, RSI
# ...
