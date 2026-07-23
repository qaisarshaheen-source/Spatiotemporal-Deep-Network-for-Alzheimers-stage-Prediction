"""
Flask app for deploying the Alzheimer's detection model.
Run: python app.py  (then open http://127.0.0.1:5000)
"""
import os
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

from config import (
    BASE_DIR,
    MODELS_DIR,
    SAVED_MODEL_PATH,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    CLASS_NAMES,
    IMG_EXTENSIONS,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and class names once at startup
_model = None
_class_names = None


def load_model_and_classes():
    global _model, _class_names
    if _model is not None:
        return
    if not os.path.isfile(SAVED_MODEL_PATH):
        return  # Allow app to start; predict will return an error
    _model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    if os.path.isfile(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            _class_names = [line.strip() for line in f if line.strip()]
    else:
        _class_names = CLASS_NAMES


def preprocess_image(image_path_or_pil):
    """Resize and normalize image to match training pipeline."""
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(image_array):
    """Return class index, label, and per-class probabilities."""
    load_model_and_classes()
    if _model is None:
        raise FileNotFoundError(
            f"Model not found at {SAVED_MODEL_PATH}. Run train_models.py first."
        )
    logits = _model.predict(image_array, verbose=0)
    probs = logits[0].tolist()
    idx = int(np.argmax(logits[0]))
    label = _class_names[idx] if idx < len(_class_names) else f"Class_{idx}"
    return {
        "class_index": idx,
        "class_label": label,
        "probabilities": dict(zip(_class_names, probs)),
        "confidence": float(probs[idx]),
    }


def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files.get("file") or request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, or JPEG."}), 400
    try:
        img = Image.open(file.stream).convert("RGB")
        arr = preprocess_image(img)
        result = predict(arr)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if request.method == "GET":
        return render_template("upload.html", class_names=CLASS_NAMES)
    if "file" not in request.files:
        return render_template("upload.html", error="No file selected", class_names=CLASS_NAMES)
    file = request.files["file"]
    if not file or file.filename == "":
        return render_template("upload.html", error="No file selected", class_names=CLASS_NAMES)
    if not allowed_file(file.filename):
        return render_template("upload.html", error="Invalid file type.", class_names=CLASS_NAMES)
    try:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        arr = preprocess_image(path)
        result = predict(arr)
        return render_template(
            "upload.html",
            result=result,
            image_url=url_for("uploaded_file", filename=filename),
            class_names=CLASS_NAMES,
        )
    except Exception as e:
        return render_template("upload.html", error=str(e), class_names=CLASS_NAMES)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    load_model_and_classes()
    if _model is None:
        print(f"Warning: No model at {SAVED_MODEL_PATH}. Run train_models.py first.")
    app.run(host="0.0.0.0", port=5000, debug=True)
