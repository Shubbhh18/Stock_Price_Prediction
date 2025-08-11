"""
Batch stock analysis CLI that processes multiple candlestick chart images with stock symbols.
Usage:
  .venv\Scripts\python -m app.batch_stock_analyzer --input stocks.csv --model models\trained_model.pth --output stock_analysis.csv --device auto
"""
import argparse
from pathlib import Path
import pandas as pd
from app.stock_predictor import analyze_stock_chart


def batch_stock_analysis(input_csv: Path, model_path: Path, output_path: Path, 
                        device: str = 'cpu', horizon: int = 1):
    """Process multiple stock chart images from a CSV input."""
    
    # Read input CSV
    try:
        df = pd.read_csv(input_csv)
        required_cols = ['image', 'stock']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ Error: CSV must contain columns: {required_cols}")
            return
        
        print(f"📊 Processing {len(df)} stock charts...")
        
    except Exception as e:
        print(f"❌ Error reading input CSV: {e}")
        return
    
    results = []
    for idx, row in df.iterrows():
        try:
            image_path = Path(row['image'])
            stock_symbol = row['stock']
            last_price = row.get('last_price', None)
            
            if not image_path.exists():
                print(f"⚠️  Warning: Image not found: {image_path}")
                continue
            
            print(f"🔍 Analyzing {stock_symbol} ({idx+1}/{len(df)})...")
            
            # Analyze the stock chart
            result = analyze_stock_chart(
                image_path, 
                stock_symbol, 
                model_path, 
                device=device,
                last_price=last_price,
                horizon=horizon
            )
            
            # Flatten the result for CSV output
            flat_result = {
                'stock_symbol': result['stock_symbol'],
                'sector': result['stock_context']['sector'],
                'market_cap': result['stock_context']['market_cap'],
                'volatility': result['stock_context']['volatility'],
                'prediction_type': result['prediction_type'],
                'recommendation': result['recommendation'],
                'confidence': result['confidence'],
                'risk_level': result['risk_level'],
                'timestamp': result['timestamp']
            }
            
            # Add chart analysis details
            if result['prediction_type'] == "full_analysis":
                chart = result['chart_analysis']
                flat_result.update({
                    'behavior': chart['behavior'],
                    'predicted_return': chart['predicted_return'],
                    'future_price': chart['future_price'],
                    'horizon': chart['horizon'],
                    'method': chart['method']
                })
            else:
                chart = result['chart_analysis']
                flat_result.update({
                    'behavior': chart['label'],
                    'probabilities': str(chart['probabilities'])
                })
            
            results.append(flat_result)
            
        except Exception as e:
            print(f"❌ Error processing {row.get('stock', 'unknown')}: {e}")
            results.append({
                'stock_symbol': row.get('stock', 'unknown'),
                'error': str(e)
            })
    
    # Save results
    if results:
        output_df = pd.DataFrame(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Successfully processed: {len([r for r in results if 'error' not in r])}/{len(df)} stocks")
        print(f"💾 Results saved to: {output_path}")
        
        # Show summary
        if len(results) > 0:
            print(f"\n📈 Summary:")
            buy_count = len([r for r in results if 'recommendation' in r and 'Buy' in r['recommendation']])
            sell_count = len([r for r in results if 'recommendation' in r and 'Sell' in r['recommendation']])
            print(f"   • Buy Recommendations: {buy_count}")
            print(f"   • Sell Recommendations: {sell_count}")
            
            high_confidence = len([r for r in results if 'confidence' in r and r['confidence'] == 'High'])
            print(f"   • High Confidence: {high_confidence}")
    else:
        print("❌ No results to save")


def create_sample_input_csv(output_path: str = "sample_stocks.csv"):
    """Create a sample input CSV for batch stock analysis."""
    sample_data = {
        'image': [
            'archive/Patterns/0_0000_00001.jpg',
            'archive/Patterns/0_0000_00002.jpg',
            'archive/Patterns/1_0000_00001.jpg'
        ],
        'stock': ['AAPL', 'MSFT', 'GOOGL'],
        'last_price': [150.0, 300.0, 2500.0]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"📝 Sample input CSV created: {output_path}")
    print("💡 Edit this file with your stock chart images and symbols")


def main():
    parser = argparse.ArgumentParser(description="Batch stock chart analysis")
    parser.add_argument("--input", help="Input CSV with columns: image, stock, [last_price]")
    parser.add_argument("--model", help="Path to trained model .pth")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--create-sample", action="store_true", help="Create a sample input CSV")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_input_csv()
        return
    
    # Check required arguments for analysis
    if not args.input or not args.model or not args.output:
        print("❌ Error: --input, --model, and --output are required for analysis")
        print("💡 Use --create-sample to create a sample input CSV")
        return
    
    if not Path(args.input).exists():
        print(f"❌ Error: Input CSV not found: {args.input}")
        return
    
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    batch_stock_analysis(
        Path(args.input),
        Path(args.model),
        Path(args.output),
        device=args.device,
        horizon=args.horizon
    )


if __name__ == "__main__":
    main()
