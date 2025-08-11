# 🇮🇳 Indian Stock Market Chart Pattern Analyzer - Reference Guide

## 🎯 **What This System Does**
This system recognizes **Double Top** and **Double Bottom** chart patterns from your dataset and provides comprehensive Indian stock market analysis including:
- Real company information and sector details
- Pattern-specific trading strategies
- Indian market context and timing
- Sector peer comparisons
- Risk assessment and recommendations

## 📊 **Available Indian Stocks by Sector**

### 🏭 **Adani Group**
- **ADANIGREEN** - Adani Green Energy Ltd (Renewable Energy)
- **ADANIENT** - Adani Enterprises Ltd (Conglomerate)
- **ADANIPORTS** - Adani Ports & SEZ Ltd (Infrastructure)
- **ADANIPOWER** - Adani Power Ltd (Power)
- **ADANITRANS** - Adani Transmission Ltd (Power Transmission)

### 🏦 **Banking & Financial**
- **HDFCBANK** - HDFC Bank Ltd (Private Banking)
- **ICICIBANK** - ICICI Bank Ltd (Private Banking)
- **SBIN** - State Bank of India (PSU Banking)
- **AXISBANK** - Axis Bank Ltd (Private Banking)
- **KOTAKBANK** - Kotak Mahindra Bank Ltd (Private Banking)

### 💻 **IT & Technology**
- **TCS** - Tata Consultancy Services Ltd (IT Services)
- **INFY** - Infosys Ltd (IT Services)
- **WIPRO** - Wipro Ltd (IT Services)
- **HCLTECH** - HCL Technologies Ltd (IT Services)
- **TECHM** - Tech Mahindra Ltd (IT Services)

### ⛽ **Oil & Gas**
- **RELIANCE** - Reliance Industries Ltd (Oil & Gas)
- **ONGC** - Oil & Natural Gas Corp Ltd (Oil Exploration)
- **IOC** - Indian Oil Corporation Ltd (Oil Marketing)
- **BPCL** - Bharat Petroleum Corp Ltd (Oil Marketing)
- **HPCL** - Hindustan Petroleum Corp Ltd (Oil Marketing)

### 🛒 **FMCG (Fast Moving Consumer Goods)**
- **ITC** - ITC Ltd (Cigarettes, Hotels, FMCG)
- **HINDUNILVR** - Hindustan Unilever Ltd (Consumer Goods)
- **NESTLEIND** - Nestle India Ltd (Food Products)
- **BRITANNIA** - Britannia Industries Ltd (Biscuits)
- **MARICO** - Marico Ltd (Personal Care)

### 🚗 **Automobile**
- **MARUTI** - Maruti Suzuki India Ltd (Passenger Cars)
- **TATAMOTORS** - Tata Motors Ltd (Cars & Commercial Vehicles)
- **M&M** - Mahindra & Mahindra Ltd (SUVs & Tractors)
- **BAJAJ-AUTO** - Bajaj Auto Ltd (Two-wheelers)
- **HEROMOTOCO** - Hero MotoCorp Ltd (Two-wheelers)

### 💊 **Pharmaceuticals**
- **SUNPHARMA** - Sun Pharmaceutical Industries Ltd (Generic Drugs)
- **DRREDDY** - Dr Reddy's Laboratories Ltd (Generic Drugs)
- **CIPLA** - Cipla Ltd (Generic Drugs)
- **DIVISLAB** - Divi's Laboratories Ltd (API Manufacturing)
- **APOLLOHOSP** - Apollo Hospitals Enterprise Ltd (Healthcare)

### 🏗️ **Infrastructure**
- **LT** - Larsen & Toubro Ltd (Engineering & Construction)
- **BHEL** - Bharat Heavy Electricals Ltd (Power Equipment)
- **NTPC** - NTPC Ltd (Power Generation)
- **POWERGRID** - Power Grid Corporation Ltd (Power Transmission)

### 🏭 **Metals & Mining**
- **TATASTEEL** - Tata Steel Ltd (Steel)
- **JSWSTEEL** - JSW Steel Ltd (Steel)
- **VEDL** - Vedanta Ltd (Mining & Metals)
- **HINDALCO** - Hindalco Industries Ltd (Aluminium)
- **COALINDIA** - Coal India Ltd (Coal Mining)

### 🏠 **Real Estate**
- **DLF** - DLF Ltd (Real Estate Development)
- **GODREJPROP** - Godrej Properties Ltd (Real Estate)
- **SUNTV** - Sun TV Network Ltd (Media & Entertainment)
- **INDIAINFO** - India Infoline Ltd (Financial Services)

## 🚀 **How to Use**

### **Basic Analysis (Pattern Recognition Only)**
```bash
.venv\Scripts\python -m app.indian_stock_analyzer --image "chart.jpg" --stock ADANIGREEN --model models\trained_model.pth
```

### **Full Analysis with Price Prediction**
```bash
.venv\Scripts\python -m app.indian_stock_analyzer --image "chart.jpg" --stock RELIANCE --model models\trained_model.pth --last_price 2500 --horizon 5
```

### **Batch Analysis (Multiple Stocks)**
```bash
.venv\Scripts\python -m app.batch_stock_analyzer --input sample_stocks.csv --model models\trained_model.pth --output results.csv
```

## 📈 **Chart Patterns Recognized**

### **Double Top (Bearish)**
- **Description**: Bearish reversal pattern indicating potential downtrend
- **Indian Market Context**: Common during earnings season or policy announcements
- **Trading Strategy**: Consider selling or shorting, set stop-loss above resistance
- **Success Rate**: 70-75% in trending markets

### **Double Bottom (Bullish)**
- **Description**: Bullish reversal pattern indicating potential uptrend
- **Indian Market Context**: Often seen during market corrections or sector rotation
- **Trading Strategy**: Consider buying, set stop-loss below support
- **Success Rate**: 75-80% in trending markets

## 💡 **Key Features**

✅ **Real Indian Stock Data** - Company names, sectors, market caps
✅ **Sector-Specific Analysis** - Banking, IT, Oil & Gas, FMCG, Auto, Pharma
✅ **Pattern Recognition** - Double Top/Bottom from your dataset
✅ **Indian Market Context** - RBI policy, crude prices, global trends
✅ **Sector Peer Comparison** - Compare with similar stocks
✅ **Risk Assessment** - Confidence levels and stop-loss strategies
✅ **Currency Support** - ₹ (INR) for Indian stocks
✅ **Exchange Information** - NSE (National Stock Exchange)

## 🔍 **Example Output**
The system provides comprehensive analysis including:
- Stock information and sector details
- Chart pattern identification
- Technical analysis insights
- Indian market context
- Trading recommendations
- Risk assessment
- Sector peer comparisons

## 📝 **Notes**
- All stocks are from NSE (National Stock Exchange of India)
- Analysis is based on your trained model's pattern recognition
- Recommendations include Indian market-specific factors
- Currency displayed in ₹ (Indian Rupees)
- Sector analysis considers Indian market conditions

---
*This system transforms your chart pattern dataset into a comprehensive Indian stock market analysis tool! 🇮🇳📈*
