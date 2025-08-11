"""
Minimal Flask API to upload an image and get prediction.
POST /predict with form-data file field name 'file'
Returns JSON: {label, probabilities}
"""
from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
from werkzeug.utils import secure_filename
import os
from app.model_predictor import load_cnn_model, predict_image, analyze_image

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_latest.pth"  # adjust as needed
DEVICE = "cpu"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload cap

# lazy load model
_model = None
def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found, train and save to models/cnn_latest.pth")
        _model = load_cnn_model(MODEL_PATH, device=DEVICE)
    return _model

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        """
        <html>
          <head><title>Stock Pattern Predictor</title></head>
          <body>
            <h2>Upload an image to get a prediction</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
              <input type="file" name="file" accept="image/*" required />
              <button type="submit">Predict</button>
            </form>
            <p>Or POST multipart/form-data with field name <code>file</code> to <code>/predict</code>.</p>
          </body>
        </html>
        """
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "usage": "POST an image with multipart/form-data field 'file' to this endpoint.",
            "example": "curl -s -X POST -F file=@path/to/image.jpg http://127.0.0.1:5000/predict"
        }), 200
    if 'file' not in request.files:
        return jsonify({"error":"No file part"}), 400
    f = request.files['file']
    if f.filename == "":
        return jsonify({"error":"no selected file"}), 400
    filename = secure_filename(f.filename)
    dest = UPLOAD_DIR / filename
    f.save(dest)
    model = get_model()
    result = predict_image(model, dest, device=DEVICE)
    return jsonify(result)

@app.route("/analyze", methods=["POST", "GET"])
def analyze():
    if request.method == "GET":
        return jsonify({
            "usage": "POST multipart/form-data with 'file' image, and optional 'last_price' (float) and 'horizon' (int).",
        }), 200
    if 'file' not in request.files:
        return jsonify({"error":"No file part"}), 400
    f = request.files['file']
    if f.filename == "":
        return jsonify({"error":"no selected file"}), 400
    filename = secure_filename(f.filename)
    dest = UPLOAD_DIR / filename
    f.save(dest)
    last_price = request.form.get('last_price', type=float, default=None)
    horizon = request.form.get('horizon', type=int, default=1)
    model = get_model()
    result = analyze_image(model, dest, device=DEVICE, last_price=last_price, horizon=horizon)
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    try:
        m = get_model()
        return jsonify({"status": "ok", "model_loaded": m is not None}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
