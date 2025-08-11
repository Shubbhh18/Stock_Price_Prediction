"""
Stock-specific prediction CLI that analyzes candlestick chart images with stock context.
Usage:
  .venv\Scripts\python -m app.stock_predictor --image chart.jpg --stock AAPL --model models\trained_model.pth --device auto
"""
import argparse
from pathlib import Path
import pandas as pd
from app.model_predictor import load_cnn_model, predict_image, analyze_image


def get_stock_context(stock_symbol: str) -> dict:
    """Get basic stock context information with real Indian stock data."""
    stock_symbol = stock_symbol.upper()
    
    # Real Indian stock information - covering major sectors
    indian_stocks = {
        # Adani Group
        'ADANIGREEN': {
            'name': 'Adani Green Energy Ltd',
            'sector': 'Renewable Energy',
            'market_cap': 'Large Cap',
            'volatility': 'High',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Renewable energy stocks often show double tops/bottoms due to policy changes'
        },
        'ADANIENT': {
            'name': 'Adani Enterprises Ltd',
            'sector': 'Conglomerate',
            'market_cap': 'Large Cap',
            'volatility': 'High',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Conglomerate stocks show complex patterns due to diverse business exposure'
        },
        'ADANIPORTS': {
            'name': 'Adani Ports & SEZ Ltd',
            'sector': 'Infrastructure',
            'market_cap': 'Large Cap',
            'volatility': 'High',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Infrastructure stocks show cyclical patterns based on economic cycles'
        },
        
        # Banking & Financial
        'HDFCBANK': {
            'name': 'HDFC Bank Ltd',
            'sector': 'Banking',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Banking stocks show double tops/bottoms during rate change cycles'
        },
        'ICICIBANK': {
            'name': 'ICICI Bank Ltd',
            'sector': 'Banking',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Private banks show technical patterns based on credit growth cycles'
        },
        'SBIN': {
            'name': 'State Bank of India',
            'sector': 'Banking',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'PSU banks show patterns influenced by government policies'
        },
        
        # IT & Technology
        'TCS': {
            'name': 'Tata Consultancy Services Ltd',
            'sector': 'Information Technology',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'IT stocks show patterns based on global tech trends and currency movements'
        },
        'INFY': {
            'name': 'Infosys Ltd',
            'sector': 'Information Technology',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Large IT companies show double tops during earnings season'
        },
        'WIPRO': {
            'name': 'Wipro Ltd',
            'sector': 'Information Technology',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'IT stocks show reversal patterns during quarterly results'
        },
        
        # Oil & Gas
        'RELIANCE': {
            'name': 'Reliance Industries Ltd',
            'sector': 'Oil & Gas',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Oil stocks show double tops/bottoms based on crude price cycles'
        },
        'ONGC': {
            'name': 'Oil & Natural Gas Corp Ltd',
            'sector': 'Oil & Gas',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'PSU oil companies show patterns influenced by government pricing policies'
        },
        
        # FMCG
        'ITC': {
            'name': 'ITC Ltd',
            'sector': 'FMCG',
            'market_cap': 'Large Cap',
            'volatility': 'Low',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'FMCG stocks show stable patterns with occasional double tops during high valuations'
        },
        'HINDUNILVR': {
            'name': 'Hindustan Unilever Ltd',
            'sector': 'FMCG',
            'market_cap': 'Large Cap',
            'volatility': 'Low',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Consumer goods show defensive patterns during market volatility'
        },
        
        # Auto
        'MARUTI': {
            'name': 'Maruti Suzuki India Ltd',
            'sector': 'Automobile',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Auto stocks show cyclical patterns based on demand and input costs'
        },
        'TATAMOTORS': {
            'name': 'Tata Motors Ltd',
            'sector': 'Automobile',
            'market_cap': 'Large Cap',
            'volatility': 'High',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Auto manufacturers show volatile patterns during industry transitions'
        },
        
        # Pharma
        'SUNPHARMA': {
            'name': 'Sun Pharmaceutical Industries Ltd',
            'sector': 'Pharmaceuticals',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Pharma stocks show patterns based on regulatory approvals and patent cliffs'
        },
        'DRREDDY': {
            'name': 'Dr Reddy\'s Laboratories Ltd',
            'sector': 'Pharmaceuticals',
            'market_cap': 'Large Cap',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Generic pharma shows patterns during FDA inspections and approvals'
        },
        
        # Stock Exchanges & Financial Services
        'BSE': {
            'name': 'BSE Ltd',
            'sector': 'Financial Services',
            'market_cap': 'Mid Cap',
            'volatility': 'High',
            'currency': 'INR',
            'exchange': 'BSE',
            'pattern_analysis': 'Exchange stocks show patterns based on trading volumes and market sentiment'
        },
        'NSE': {
            'name': 'National Stock Exchange of India Ltd',
            'sector': 'Financial Services',
            'market_cap': 'Unlisted',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'pattern_analysis': 'Exchange operator shows patterns based on market activity and regulatory changes'
        }
    }
    
    # Check if we have real data for this stock
    if stock_symbol in indian_stocks:
        stock_info = indian_stocks[stock_symbol]
        return {
            'symbol': stock_symbol,
            'name': stock_info['name'],
            'sector': stock_info['sector'],
            'market_cap': stock_info['market_cap'],
            'volatility': stock_info['volatility'],
            'currency': stock_info['currency'],
            'exchange': stock_info['exchange'],
            'type': 'Indian Stock',
            'pattern_analysis': stock_info['pattern_analysis']
        }
    else:
        # Fallback for unknown stocks
        return {
            'symbol': stock_symbol,
            'name': f'{stock_symbol} Stock',
            'sector': 'Unknown',
            'market_cap': 'Unknown',
            'volatility': 'Medium',
            'currency': 'INR',
            'exchange': 'NSE',
            'type': 'Stock',
            'pattern_analysis': 'Pattern analysis available after stock identification'
        }


def get_pattern_insights(pattern_label: str, stock_context: dict) -> dict:
    """Get detailed insights about the chart pattern for Indian stocks."""
    
    pattern_insights = {
        'Double top': {
            'description': 'Bearish reversal pattern indicating potential downtrend',
            'indian_market_context': 'Common in Indian markets during earnings season or policy announcements',
            'trading_strategy': 'Consider selling or shorting, set stop-loss above resistance',
            'risk_factors': 'High - pattern failure can lead to continued uptrend',
            'timeframe': 'Short to medium term (1-4 weeks)',
            'success_rate': '70-75% in trending markets'
        },
        'Double bottom': {
            'description': 'Bullish reversal pattern indicating potential uptrend',
            'indian_market_context': 'Often seen during market corrections or sector rotation',
            'trading_strategy': 'Consider buying, set stop-loss below support',
            'risk_factors': 'Medium - pattern failure can lead to continued downtrend',
            'timeframe': 'Short to medium term (1-4 weeks)',
            'success_rate': '75-80% in trending markets'
        }
    }
    
    # Get base pattern insights
    base_insights = pattern_insights.get(pattern_label, {
        'description': 'Technical pattern analysis',
        'indian_market_context': 'Pattern behavior in Indian market conditions',
        'trading_strategy': 'General trading approach',
        'risk_factors': 'Medium',
        'timeframe': 'Short to medium term',
        'success_rate': '70-80%'
    })
    
    # Enhance with stock-specific insights
    enhanced_insights = base_insights.copy()
    enhanced_insights['stock_specific'] = stock_context.get('pattern_analysis', 'Stock-specific pattern analysis available')
    
    return enhanced_insights


def analyze_stock_chart(image_path: Path, stock_symbol: str, model_path: Path, 
                       device: str = 'cpu', last_price: float = None, horizon: int = 1):
    """Analyze a candlestick chart image with stock-specific context."""
    
    # Load model
    model = load_cnn_model(model_path, device=device)
    
    # Get stock context
    stock_context = get_stock_context(stock_symbol)
    
    # Get basic prediction
    if last_price:
        result = analyze_image(model, image_path, device=device, last_price=last_price, horizon=horizon)
        prediction_type = "full_analysis"
    else:
        result = predict_image(model, image_path, device=device)
        prediction_type = "behavior_only"
    
    # Get pattern insights
    if prediction_type == "full_analysis":
        pattern_label = result['behavior']
    else:
        pattern_label = result['label']
    
    pattern_insights = get_pattern_insights(pattern_label, stock_context)
    
    # Enhance with stock context
    enhanced_result = {
        'stock_symbol': stock_symbol.upper(),
        'stock_context': stock_context,
        'prediction_type': prediction_type,
        'chart_analysis': result,
        'pattern_insights': pattern_insights,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Add stock-specific insights
    if prediction_type == "full_analysis":
        behavior = result['behavior']
        predicted_return = result['predicted_return']
        
        # Stock-specific recommendations
        if behavior == 'bullish':
            if predicted_return > 0.01:  # >1% return
                enhanced_result['recommendation'] = 'Strong Buy'
                enhanced_result['confidence'] = 'High'
            else:
                enhanced_result['recommendation'] = 'Buy'
                enhanced_result['confidence'] = 'Medium'
        else:  # bearish
            if abs(predicted_return) > 0.01:  # >1% decline
                enhanced_result['recommendation'] = 'Strong Sell'
                enhanced_result['confidence'] = 'High'
            else:
                enhanced_result['recommendation'] = 'Sell'
                enhanced_result['confidence'] = 'Medium'
        
        enhanced_result['risk_level'] = 'Medium' if abs(predicted_return) < 0.02 else 'High'
        
    else:
        # Behavior only mode
        behavior = result['label']
        enhanced_result['recommendation'] = 'Buy' if behavior == 'bullish' else 'Sell'
        enhanced_result['confidence'] = 'Medium'
        enhanced_result['risk_level'] = 'Medium'
    
    return enhanced_result


def print_stock_analysis(result: dict):
    """Pretty print the stock analysis results."""
    print("\n" + "="*70)
    print(f"📈 INDIAN STOCK MARKET ANALYSIS: {result['stock_symbol']}")
    print("="*70)
    
    # Stock Info
    print(f"📊 Stock Symbol: {result['stock_symbol']}")
    print(f"🏢 Company: {result['stock_context']['name']}")
    print(f"🏭 Sector: {result['stock_context']['sector']}")
    print(f"📏 Market Cap: {result['stock_context']['market_cap']}")
    print(f"⚡ Volatility: {result['stock_context']['volatility']}")
    print(f"💱 Exchange: {result['stock_context']['exchange']}")
    
    # Chart Analysis
    print(f"\n🔍 CHART PATTERN ANALYSIS:")
    if result['prediction_type'] == "full_analysis":
        chart = result['chart_analysis']
        currency_symbol = "₹" if result['stock_context']['currency'] == 'INR' else "$"
        print(f"   • Pattern Type: {chart['behavior'].upper()}")
        print(f"   • Predicted Return: {chart['predicted_return']:.4f} ({chart['predicted_return']*100:.2f}%)")
        print(f"   • Future Price: {currency_symbol}{chart['future_price']:.2f}")
        print(f"   • Time Horizon: {chart['horizon']} periods")
        print(f"   • Method: {chart['method']}")
    else:
        chart = result['chart_analysis']
        print(f"   • Pattern Type: {chart['label'].upper()}")
        print(f"   • Probabilities: {chart['probabilities']}")
    
    # Pattern Insights
    pattern = result['pattern_insights']
    print(f"\n📊 PATTERN INSIGHTS:")
    print(f"   • Description: {pattern['description']}")
    print(f"   • Indian Market Context: {pattern['indian_market_context']}")
    print(f"   • Trading Strategy: {pattern['trading_strategy']}")
    print(f"   • Risk Factors: {pattern['risk_factors']}")
    print(f"   • Timeframe: {pattern['timeframe']}")
    print(f"   • Success Rate: {pattern['success_rate']}")
    
    # Stock-Specific Analysis
    print(f"\n🏢 STOCK-SPECIFIC ANALYSIS:")
    print(f"   • Sector Pattern: {result['stock_context']['pattern_analysis']}")
    
    # Recommendations
    print(f"\n💡 TRADING RECOMMENDATIONS:")
    print(f"   • Action: {result['recommendation']}")
    print(f"   • Confidence: {result['confidence']}")
    print(f"   • Risk Level: {result['risk_level']}")
    
    # Timestamp
    print(f"\n⏰ Analysis Time: {result['timestamp']}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Stock-specific candlestick chart analysis")
    parser.add_argument("--image", required=True, help="Path to candlestick chart image")
    parser.add_argument("--stock", required=True, help="Stock symbol (e.g., AAPL, MSFT)")
    parser.add_argument("--model", required=True, help="Path to trained model .pth")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--last_price", type=float, help="Current stock price for future estimation")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--output", help="Optional CSV output path for batch processing")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"❌ Error: Image file not found: {args.image}")
        return
    
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    try:
        # Analyze the stock chart
        result = analyze_stock_chart(
            Path(args.image), 
            args.stock, 
            Path(args.model), 
            device=args.device,
            last_price=args.last_price,
            horizon=args.horizon
        )
        
        # Print results
        print_stock_analysis(result)
        
        # Save to CSV if requested
        if args.output:
            df = pd.DataFrame([result])
            df.to_csv(args.output, index=False)
            print(f"\n💾 Results saved to: {args.output}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
