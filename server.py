import setup_tf  # Setup TensorFlow path for Windows
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import uuid
import json
from datetime import datetime

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    TF_IMPORT_ERROR = None
except Exception as e:
    load_model = None
    TENSORFLOW_AVAILABLE = False
    TF_IMPORT_ERROR = str(e)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
METRICS_PATH = os.path.join(RESULTS_FOLDER, 'metrics.json')
HISTORY_PATH = os.path.join(RESULTS_FOLDER, 'prediction_history.json')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
DEFAULT_IMG_SIZE = (128, 128)

MODEL_CANDIDATES = [
    os.environ.get('MODEL_PATH'),
    os.path.join(BASE_DIR, 'pancreas_model.h5'),
    os.path.join(BASE_DIR, 'pancreas_custom_cnn.h5'),
    os.path.join(BASE_DIR, 'pancreas_vgg16.h5'),
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p and os.path.exists(p)), os.path.join(BASE_DIR, 'pancreas_model.h5'))

IMG_SIZE = DEFAULT_IMG_SIZE
USE_RGB_MODEL = False  # Auto-inferred from model input shape when possible
MODEL_ERROR = None


def apply_clahe(img):
    """Apply CLAHE for contrast enhancement (matches training preprocessing)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def infer_model_settings(model):
    """Infer input size and channels from the loaded model."""
    global IMG_SIZE, USE_RGB_MODEL
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if input_shape and len(input_shape) == 4:
            _, h, w, c = input_shape
            if h and w:
                IMG_SIZE = (int(w), int(h))
            if c == 3:
                USE_RGB_MODEL = True
            elif c == 1:
                USE_RGB_MODEL = False
        print(f"Model input shape: {input_shape}; IMG_SIZE={IMG_SIZE}; USE_RGB_MODEL={USE_RGB_MODEL}")
    except Exception as e:
        print(f"Warning: could not infer model input shape: {e}. Using defaults.")


def preprocess_image(filepath):
    """
    CT scan preprocessing matching the training pipeline.
    Supports both grayscale CNN and RGB transfer learning models.
    """
    try:
        # Read as Grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None

        # Apply CLAHE for contrast enhancement
        img = apply_clahe(img)

        # Resize
        processed_img = cv2.resize(img, IMG_SIZE)

        # Light Gaussian blur to reduce noise
        processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)

        # Normalize (0-1)
        normalized_img = processed_img / 255.0

        # Create display image (RGB) for the frontend
        display_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)

        if USE_RGB_MODEL:
            # Convert to RGB for transfer learning model (3 channels)
            normalized_img = np.stack([normalized_img] * 3, axis=-1)

        return normalized_img, display_img
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None


app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'frontend'))
CORS(app)  # Enable CORS for all routes

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load Model
model = None
if not TENSORFLOW_AVAILABLE:
    MODEL_ERROR = f"TensorFlow import failed: {TF_IMPORT_ERROR}"
    print(MODEL_ERROR)
else:
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        infer_model_settings(model)
        print("Model loaded successfully.")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"Error loading model: {MODEL_ERROR}")


# Functions to load/save prediction history
def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(history):
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


# Prediction History Database (persisted to file)
history_db = load_history()


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'model_error': MODEL_ERROR,
        'img_size': IMG_SIZE,
        'use_rgb_model': USE_RGB_MODEL,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400

    if file:
        # Save uploaded file
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Preprocess
            processed_img, display_img = preprocess_image(filepath)

            if processed_img is None:
                return jsonify({'error': 'Error processing image'}), 500

            # Predict
            if model:
                if USE_RGB_MODEL:
                    input_img = processed_img.reshape(1, IMG_SIZE[1], IMG_SIZE[0], 3)
                else:
                    input_img = processed_img.reshape(1, IMG_SIZE[1], IMG_SIZE[0], 1)
                prediction = model.predict(input_img)
                score = float(prediction[0][0])

                # Determine result
                is_tumor = score > 0.5
                confidence = score if is_tumor else 1 - score
                label = "Tumor Detected" if is_tumor else "No Tumor Detected"

                # Save processed image for display
                processed_filename = "proc_" + filename
                processed_filepath = os.path.join(RESULTS_FOLDER, processed_filename)
                # Convert RGB back to BGR for OpenCV saving
                cv2.imwrite(processed_filepath, cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))

                # Save to history with real timestamp
                result_data = {
                    'id': str(uuid.uuid4()),
                    'filename': filename,
                    'processed_filename': processed_filename,
                    'is_tumor': is_tumor,
                    'confidence': confidence,
                    'label': label,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                history_db.append(result_data)
                save_history(history_db)  # Persist to file

                return jsonify(result_data)
            else:
                detail = MODEL_ERROR or f"Model not loaded. Checked: {MODEL_PATH}"
                return jsonify({'error': detail}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Load real metrics from file if available
    saved_metrics = load_metrics()

    if saved_metrics:
        # Use saved metrics from training
        metrics = {
            'accuracy': saved_metrics.get('accuracy', 0),
            'precision': saved_metrics.get('precision', 0),
            'recall': saved_metrics.get('recall', 0),
            'f1_score': saved_metrics.get('f1_score', 0),
            'confusion_matrix': saved_metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
            'history': saved_metrics.get('history', {
                'accuracy': [],
                'val_accuracy': [],
                'loss': [],
                'val_loss': []
            }),
            'predictions': {
                'tumor': len([x for x in history_db if x['is_tumor']]),
                'normal': len([x for x in history_db if not x['is_tumor']])
            }
        }
    else:
        # Default metrics if no saved metrics
        metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'history': {
                'accuracy': [],
                'val_accuracy': [],
                'loss': [],
                'val_loss': []
            },
            'predictions': {
                'tumor': len([x for x in history_db if x['is_tumor']]),
                'normal': len([x for x in history_db if not x['is_tumor']])
            }
        }

    return jsonify(metrics)


@app.route('/history', methods=['GET'])
def get_history():
    # Return real prediction history (most recent first)
    return jsonify(list(reversed(history_db[-50:])))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
