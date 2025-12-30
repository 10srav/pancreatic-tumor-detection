import setup_tf  # Setup TensorFlow path for Windows
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import uuid
import json
from datetime import datetime

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = 'pancreas_model.h5'
METRICS_PATH = 'results/metrics.json'
HISTORY_PATH = 'results/prediction_history.json'
IMG_SIZE = (128, 128)

def apply_clahe(img):
    """Apply CLAHE for contrast enhancement (matches training preprocessing)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# Check if model uses RGB (transfer learning) or grayscale
USE_RGB_MODEL = False  # Set to False for grayscale Custom CNN model

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

app = Flask(__name__, static_folder='frontend')
CORS(app) # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = 'pancreas_model.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load Model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
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
                    input_img = processed_img.reshape(1, 128, 128, 3)
                else:
                    input_img = processed_img.reshape(1, 128, 128, 1)
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
                return jsonify({'error': 'Model not loaded'}), 500
                
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
    app.run(debug=True, port=5000)
