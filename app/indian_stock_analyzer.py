"""
Comprehensive Indian Stock Market Chart Pattern Analyzer
Recognizes all chart patterns from the dataset and provides Indian market context.
Usage:
  .venv\Scripts\python -m app.indian_stock_analyzer --image chart.jpg --stock ADANIGREEN --model models\trained_model.pth
"""
import argparse
from pathlib import Path
import pandas as pd
from app.model_predictor import load_cnn_model, predict_image, analyze_image
from app.stock_predictor import get_stock_context, get_pattern_insights


def get_indian_market_sectors():
    """Get comprehensive list of Indian market sectors with representative stocks."""
    return {
        'Adani Group': ['ADANIGREEN', 'ADANIENT', 'ADANIPORTS', 'ADANIPOWER', 'ADANITRANS'],
        'Banking & Financial': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK'],
        'IT & Technology': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
        'Oil & Gas': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HPCL'],
        'FMCG': ['ITC', 'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'MARICO'],
        'Automobile': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
        'Pharmaceuticals': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP'],
        'Infrastructure': ['LT', 'BHEL', 'NTPC', 'POWERGRID', 'ADANIPORTS'],
        'Metals & Mining': ['TATASTEEL', 'JSWSTEEL', 'VEDL', 'HINDALCO', 'COALINDIA'],
        'Real Estate': ['DLF', 'GODREJPROP', 'SUNTV', 'INDIAINFO', 'UNITECH'],
        'Stock Exchanges': ['BSE', 'NSE']
    }


def analyze_indian_chart_pattern(image_path: Path, stock_symbol: str, model_path: Path, 
                               device: str = 'cpu', last_price: float = None, horizon: int = 1):
    """Comprehensive analysis of Indian stock chart patterns."""
    
    # Load model
    model = load_cnn_model(model_path, device=device)
    
    # Get stock context
    stock_context = get_stock_context(stock_symbol)
    
    # Get basic prediction
    if last_price:
        result = analyze_image(model, image_path, device=device, last_price=last_price, horizon=horizon)
        prediction_type = "full_analysis"
        pattern_label = result['behavior']
    else:
        result = predict_image(model, image_path, device=device)
        prediction_type = "behavior_only"
        pattern_label = result['label']
    
    # Map pattern labels to our expected format
    if 'bullish' in pattern_label.lower():
        mapped_pattern = 'Double bottom'
    elif 'bearish' in pattern_label.lower():
        mapped_pattern = 'Double top'
    else:
        mapped_pattern = pattern_label
    
    # Get pattern insights
    pattern_insights = get_pattern_insights(mapped_pattern, stock_context)
    
    # Get sector peers for comparison
    sector_peers = get_sector_peers(stock_symbol)
    
    # Enhance with comprehensive analysis
    enhanced_result = {
        'stock_symbol': stock_symbol.upper(),
        'stock_context': stock_context,
        'prediction_type': prediction_type,
        'chart_analysis': result,
        'pattern_insights': pattern_insights,
        'sector_peers': sector_peers,
        'indian_market_analysis': get_indian_market_context(pattern_label, stock_context),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Add comprehensive recommendations
    enhanced_result.update(get_comprehensive_recommendations(result, stock_context, pattern_insights))
    
    return enhanced_result


def get_sector_peers(stock_symbol: str) -> list:
    """Get sector peers for comparison."""
    sectors = get_indian_market_sectors()
    
    for sector, stocks in sectors.items():
        if stock_symbol in stocks:
            # Return other stocks in the same sector
            return [s for s in stocks if s != stock_symbol]
    
    return []


def get_indian_market_context(pattern_label: str, stock_context: dict) -> dict:
    """Get Indian market specific context for the pattern."""
    
    # Map the pattern labels to our expected format
    if 'bullish' in pattern_label.lower():
        pattern_type = 'Double bottom'
    elif 'bearish' in pattern_label.lower():
        pattern_type = 'Double top'
    else:
        pattern_type = pattern_label
    
    market_context = {
        'Double top': {
            'market_condition': 'Bearish reversal expected',
            'sector_impact': f"{stock_context['sector']} sector may face selling pressure",
            'market_timing': 'Often occurs during earnings season or policy announcements',
            'volume_expectation': 'High volume confirms pattern validity',
            'key_levels': 'Key support levels to watch for breakdown confirmation'
        },
        'Double bottom': {
            'market_condition': 'Bullish reversal expected',
            'sector_impact': f"{stock_context['sector']} sector may see buying interest",
            'market_timing': 'Common during market corrections or sector rotation',
            'volume_expectation': 'Increasing volume supports reversal',
            'key_levels': 'Key resistance levels to watch for breakout confirmation'
        }
    }
    
    return market_context.get(pattern_type, {
        'market_condition': 'Pattern analysis in progress',
        'sector_impact': 'Sector-specific analysis available',
        'market_timing': 'General market timing guidance',
        'volume_expectation': 'Volume analysis recommended',
        'key_levels': 'Technical levels to monitor'
    })


def get_comprehensive_recommendations(result: dict, stock_context: dict, pattern_insights: dict) -> dict:
    """Get comprehensive trading recommendations based on Indian market context."""
    
    if 'behavior' in result:  # Full analysis mode
        behavior = result['behavior']
        predicted_return = result['predicted_return']
        
        # Enhanced recommendations based on Indian market patterns
        if behavior == 'bullish':
            if predicted_return > 0.02:  # >2% return
                recommendation = 'Strong Buy'
                confidence = 'High'
                risk_level = 'Low'
                stop_loss_strategy = 'Set stop-loss at recent support levels'
            elif predicted_return > 0.01:  # >1% return
                recommendation = 'Buy'
                confidence = 'Medium'
                risk_level = 'Medium'
                stop_loss_strategy = 'Set stop-loss below pattern support'
            else:
                recommendation = 'Buy on Dips'
                confidence = 'Low'
                risk_level = 'Medium'
                stop_loss_strategy = 'Wait for better entry points'
        else:  # bearish
            if abs(predicted_return) > 0.02:  # >2% decline
                recommendation = 'Strong Sell'
                confidence = 'High'
                risk_level = 'Low'
                stop_loss_strategy = 'Set stop-loss at recent resistance levels'
            elif abs(predicted_return) > 0.01:  # >1% decline
                recommendation = 'Sell'
                confidence = 'Medium'
                risk_level = 'Medium'
                stop_loss_strategy = 'Set stop-loss above pattern resistance'
            else:
                recommendation = 'Sell on Rallies'
                confidence = 'Low'
                risk_level = 'Medium'
                stop_loss_strategy = 'Wait for better exit points'
        
        # Sector-specific adjustments
        if stock_context['sector'] in ['Banking', 'Financial']:
            recommendation += ' (Monitor RBI policy)'
        elif stock_context['sector'] in ['Oil & Gas']:
            recommendation += ' (Watch crude prices)'
        elif stock_context['sector'] in ['IT', 'Technology']:
            recommendation += ' (Check global tech trends)'
        
    else:  # Behavior only mode
        behavior = result['label']
        recommendation = 'Buy' if behavior == 'bullish' else 'Sell'
        confidence = 'Medium'
        risk_level = 'Medium'
        stop_loss_strategy = 'Set appropriate stop-loss levels'
    
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'risk_level': risk_level,
        'stop_loss_strategy': stop_loss_strategy,
        'position_sizing': 'Consider position sizing based on risk tolerance',
        'exit_strategy': 'Plan exit strategy based on pattern completion'
    }


def print_comprehensive_analysis(result: dict):
    """Print comprehensive Indian stock market analysis."""
    print("\n" + "="*80)
    print(f"🇮🇳 COMPREHENSIVE INDIAN STOCK MARKET ANALYSIS: {result['stock_symbol']}")
    print("="*80)
    
    # Stock Information
    print(f"📊 STOCK INFORMATION:")
    print(f"   • Symbol: {result['stock_symbol']}")
    print(f"   • Company: {result['stock_context']['name']}")
    print(f"   • Sector: {result['stock_context']['sector']}")
    print(f"   • Market Cap: {result['stock_context']['market_cap']}")
    print(f"   • Volatility: {result['stock_context']['volatility']}")
    print(f"   • Exchange: {result['stock_context']['exchange']}")
    
    # Chart Pattern Analysis
    print(f"\n🔍 CHART PATTERN ANALYSIS:")
    if result['prediction_type'] == "full_analysis":
        chart = result['chart_analysis']
        currency_symbol = "₹" if result['stock_context']['currency'] == 'INR' else "$"
        print(f"   • Pattern Type: {chart['behavior'].upper()}")
        print(f"   • Predicted Return: {chart['predicted_return']:.4f} ({chart['predicted_return']*100:.2f}%)")
        print(f"   • Future Price: {currency_symbol}{chart['future_price']:.2f}")
        print(f"   • Time Horizon: {chart['horizon']} periods")
        print(f"   • Analysis Method: {chart['method']}")
    else:
        chart = result['chart_analysis']
        print(f"   • Pattern Type: {chart['label'].upper()}")
        print(f"   • Confidence: {chart['probabilities']}")
    
    # Pattern Insights
    pattern = result['pattern_insights']
    print(f"\n📊 TECHNICAL PATTERN INSIGHTS:")
    print(f"   • Description: {pattern['description']}")
    print(f"   • Indian Market Context: {pattern['indian_market_context']}")
    print(f"   • Trading Strategy: {pattern['trading_strategy']}")
    print(f"   • Risk Factors: {pattern['risk_factors']}")
    print(f"   • Timeframe: {pattern['timeframe']}")
    print(f"   • Success Rate: {pattern['success_rate']}")
    
    # Indian Market Context
    market = result['indian_market_analysis']
    print(f"\n🇮🇳 INDIAN MARKET CONTEXT:")
    print(f"   • Market Condition: {market['market_condition']}")
    print(f"   • Sector Impact: {market['sector_impact']}")
    print(f"   • Market Timing: {market['market_timing']}")
    print(f"   • Volume Expectation: {market['volume_expectation']}")
    print(f"   • Key Levels: {market['key_levels']}")
    
    # Stock-Specific Analysis
    print(f"\n🏢 STOCK-SPECIFIC ANALYSIS:")
    print(f"   • Sector Pattern: {result['stock_context']['pattern_analysis']}")
    
    # Sector Peers
    if result['sector_peers']:
        print(f"\n📈 SECTOR PEERS FOR COMPARISON:")
        print(f"   • {', '.join(result['sector_peers'])}")
    
    # Trading Recommendations
    print(f"\n💡 COMPREHENSIVE TRADING RECOMMENDATIONS:")
    print(f"   • Action: {result['recommendation']}")
    print(f"   • Confidence: {result['confidence']}")
    print(f"   • Risk Level: {result['risk_level']}")
    print(f"   • Stop-Loss Strategy: {result['stop_loss_strategy']}")
    print(f"   • Position Sizing: {result['position_sizing']}")
    print(f"   • Exit Strategy: {result['exit_strategy']}")
    
    # Timestamp
    print(f"\n⏰ Analysis Time: {result['timestamp']}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Indian Stock Market Chart Pattern Analyzer")
    parser.add_argument("--image", required=True, help="Path to candlestick chart image")
    parser.add_argument("--stock", required=True, help="Stock symbol (e.g., ADANIGREEN, RELIANCE, TCS)")
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
        # Analyze the Indian stock chart
        result = analyze_indian_chart_pattern(
            Path(args.image), 
            args.stock, 
            Path(args.model), 
            device=args.device,
            last_price=args.last_price,
            horizon=args.horizon
        )
        
        # Print comprehensive results
        print_comprehensive_analysis(result)
        
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
