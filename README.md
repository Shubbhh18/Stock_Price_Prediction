# Stock Price / Chart Prediction Project

## Structure
- `data/` : raw and processed datasets
- `models/` : model code and saved weights
- `utils/` : helper functions, preprocessing
- `app/` : inference CLI and Flask API
- `notebooks/` : analysis & experiments

##  **Quickstart**

### **1. Setup Environment**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Run Basic CLI**
```bash
# Simple image prediction
.venv\Scripts\python -m app.main --image "chart.jpg" --model models\trained_model.pth

# Analysis mode with price prediction
.venv\Scripts\python -m app.main --image "chart.jpg" --model models\trained_model.pth --analyze --last_price 100 --horizon 5
```

### **3. Run Flask API**
```bash
.venv\Scripts\python -m app.flask_app
```
Then visit `http://127.0.0.1:5000` for web interface.

## 🇮🇳 **Indian Stock Market Analysis**

### **Comprehensive Stock Analyzer**
The system now recognizes all chart patterns from your dataset as Indian stocks and provides detailed market analysis:

```bash
# Full Indian stock analysis
.venv\Scripts\python -m app.indian_stock_analyzer --image "chart.jpg" --stock ADANIGREEN --model models\trained_model.pth --last_price 933 --horizon 5
```

**Features:**
- ✅ **50+ Indian Stocks** across all major sectors
- ✅ **Real Company Data** (names, sectors, market caps)
- ✅ **Pattern Recognition** (Double Top/Bottom from your dataset)
- ✅ **Indian Market Context** (RBI policy, crude prices, global trends)
- ✅ **Sector Analysis** (Banking, IT, Oil & Gas, FMCG, Auto, Pharma)
- ✅ **Peer Comparison** (compare with similar stocks)
- ✅ **Trading Recommendations** with risk assessment

### **Available Indian Stocks**
- **Adani Group**: ADANIGREEN, ADANIENT, ADANIPORTS
- **Banking**: HDFCBANK, ICICIBANK, SBIN, AXISBANK
- **IT**: TCS, INFY, WIPRO, HCLTECH
- **Oil & Gas**: RELIANCE, ONGC, IOC, BPCL
- **FMCG**: ITC, HINDUNILVR, NESTLEIND
- **Auto**: MARUTI, TATAMOTORS, M&M
- **Pharma**: SUNPHARMA, DRREDDY, CIPLA
- **And many more...**

See `INDIAN_STOCKS_REFERENCE.md` for complete list.

## Train the image classifier
```powershell
# If your CSV uses our default schema (image,label):
.venv\Scripts\python -m app.train_image_classifier --images_dir data\raw\Patterns --index data\Patterns.csv --epochs 3 --out models\cnn_latest.pth --device auto

# For your archive schema (Path, ClassId) pointing to 'Patterns/<file>.jpg':
.venv\Scripts\python -m app.train_image_classifier --images_dir data\raw --index archive\Patterns.csv --image_col Path --label_col ClassId --epochs 3 --out models\cnn_latest.pth --device auto
```

## Batch Processing
Process entire folders of images and output CSV results:

```powershell
# Simple predictions (behavior only):
.venv\Scripts\python -m app.batch_predict --images_dir archive\Patterns --model models\cnn_latest.pth --output predictions.csv --device auto

# Full analysis (behavior + future price):
.venv\Scripts\python -m app.batch_predict --images_dir archive\Patterns --model models\cnn_latest.pth --output analysis.csv --device auto --analyze --last_price 100 --horizon 5
```

The batch CLI will:
- Process all images in the specified directory
- Show progress every 10 images
- Save results to CSV with columns:
  - Simple mode: `image`, `label`, `probabilities`
  - Analyze mode: `image`, `behavior`, `predicted_return`, `future_price`, `method`
- Display sample results and summary statistics

## Stock-Specific Analysis 
Analyze candlestick chart images with stock context and recommendations:

### Single Stock Analysis
```powershell
# Basic behavior analysis:
.venv\Scripts\python -m app.stock_predictor --image chart.jpg --stock AAPL --model models\trained_model.pth --device auto

# Full analysis with price prediction:
.venv\Scripts\python -m app.stock_predictor --image chart.jpg --stock AAPL --model models\trained_model.pth --device auto --last_price 150.50 --horizon 5
```

**Output includes:**
-  Stock symbol, sector, market cap, volatility
-  Chart pattern behavior (bullish/bearish)
-  Trading recommendations (Buy/Sell/Strong Buy/Strong Sell)
-  Confidence level and risk assessment
-  Predicted return and future price (if last_price provided)

### Batch Stock Analysis
Process multiple stock charts with their symbols:

```powershell
# Create sample input CSV:
.venv\Scripts\python -m app.batch_stock_analyzer --create-sample

# Run batch analysis:
.venv\Scripts\python -m app.batch_stock_analyzer --input stocks.csv --model models\trained_model.pth --output results.csv --device auto --horizon 3
```

**Input CSV format:**
```csv
image,stock,last_price
chart1.jpg,AAPL,150.0
chart2.jpg,MSFT,300.0
chart3.jpg,GOOGL,2500.0
```

**Output includes:**
- Stock analysis for each chart
- Trading recommendations
- Risk assessment
- Summary statistics (buy/sell counts, confidence levels)
