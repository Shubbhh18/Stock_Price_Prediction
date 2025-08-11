"""
Batch prediction CLI that processes a folder of images and outputs CSV results.
Usage:
  .venv\Scripts\python -m app.batch_predict --images_dir archive\Patterns --model models\cnn_latest.pth --output predictions.csv --device auto
"""
import argparse
from pathlib import Path
import pandas as pd
from app.model_predictor import load_cnn_model, predict_image, analyze_image


def batch_predict(images_dir: Path, model_path: Path, output_path: Path, device: str = 'cpu', 
                  analyze_mode: bool = False, last_price: float = None, horizon: int = 1):
    """Process all images in a directory and save results to CSV."""
    model = load_cnn_model(model_path, device=device)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    results = []
    for i, img_path in enumerate(image_files):
        try:
            if analyze_mode:
                result = analyze_image(model, img_path, device=device, last_price=last_price, horizon=horizon)
                results.append({
                    'image': img_path.name,
                    'behavior': result['behavior'],
                    'predicted_return': result['predicted_return'],
                    'future_price': result['future_price'],
                    'method': result['method']
                })
            else:
                result = predict_image(model, img_path, device=device)
                results.append({
                    'image': img_path.name,
                    'label': result['label'],
                    'probabilities': str(result['probabilities'])
                })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            results.append({
                'image': img_path.name,
                'error': str(e)
            })
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Successfully processed {len([r for r in results if 'error' not in r])}/{len(image_files)} images")
    
    # Show sample results
    if results:
        print("\nSample results:")
        print(df.head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Batch predict on a folder of images")
    parser.add_argument("--images_dir", required=True, help="Directory containing images")
    parser.add_argument("--model", required=True, help="Path to trained model .pth")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--analyze", action="store_true", help="Use analyze mode (behavior + future price)")
    parser.add_argument("--last_price", type=float, help="Last known price for future price estimation")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    
    args = parser.parse_args()
    
    if args.analyze and args.last_price is None:
        print("Warning: --analyze mode requires --last_price for future price estimation")
        print("Using default last_price=100")
        args.last_price = 100.0
    
    batch_predict(
        Path(args.images_dir), 
        Path(args.model), 
        Path(args.output), 
        device=args.device,
        analyze_mode=args.analyze,
        last_price=args.last_price,
        horizon=args.horizon
    )


if __name__ == "__main__":
    main()
