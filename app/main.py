"""
Simple CLI to predict from an image:
python app/main.py --image path/to/image.png --model path/to/model.pth
"""
import argparse
from pathlib import Path
import pandas as pd
from app.model_predictor import predict_image, load_cnn_model, analyze_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to chart image")
    parser.add_argument("--model", required=True, help="Path to trained model .pth")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--last_price", type=float, default=None, help="Optional last known price for future price estimate")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast steps (for naive/heuristic)")
    parser.add_argument("--analyze", action="store_true", help="Return combined analysis (behavior + future price)")
    parser.add_argument("--ohlc", type=str, default=None, help="Optional path to OHLC CSV/XLSX with a 'close' column. If provided, used for price forecast.")
    args = parser.parse_args()

    model = load_cnn_model(Path(args.model), device=args.device)
    if args.analyze:
        ohlc_df = None
        if args.ohlc:
            ohlc_path = Path(args.ohlc)
            if ohlc_path.suffix.lower() in [".xlsx", ".xls"]:
                ohlc_df = pd.read_excel(ohlc_path)
            else:
                ohlc_df = pd.read_csv(ohlc_path)
            # if last_price not provided, infer from last close
            if args.last_price is None and 'close' in ohlc_df.columns and len(ohlc_df) > 0:
                args.last_price = float(ohlc_df['close'].iloc[-1])
        out = analyze_image(model, Path(args.image), device=args.device, last_price=args.last_price, horizon=args.horizon, ohlc_df=ohlc_df)
        print("Analysis:", out)
    else:
        out = predict_image(model, Path(args.image), device=args.device)
        print("Prediction:", out)

if __name__ == "__main__":
    main()
