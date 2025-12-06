from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import uuid
import json

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = 'laptop_pancreas_model.h5'
IMG_SIZE = (128, 128)

def preprocess_image(filepath):
    """
    Simple preprocessing for the laptop model:
    Grayscale -> Resize (128,128) -> Normalize
    """
    try:
        # Read as Grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
            
        # Resize
        processed_img = cv2.resize(img, IMG_SIZE)
        
        # Normalize (0-1)
        normalized_img = processed_img / 255.0
        
        # Reshape for model (1, 128, 128, 1)
        # input_data = normalized_img.reshape(1, 128, 128, 1)
        
        # Create a display image (RGB) from the grayscale for the frontend
        display_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        
        return normalized_img, display_img
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None

app = Flask(__name__, static_folder='frontend')
CORS(app) # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = 'laptop_pancreas_model.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load Model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Mock Database for Dashboard (In-memory for demo)
history_db = []

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
                
                # Save to history
                result_data = {
                    'id': str(uuid.uuid4()),
                    'filename': filename,
                    'processed_filename': processed_filename,
                    'is_tumor': is_tumor,
                    'confidence': confidence,
                    'label': label,
                    'timestamp': '2025-12-03' # Mock timestamp
                }
                history_db.append(result_data)
                
                return jsonify(result_data)
            else:
                return jsonify({'error': 'Model not loaded'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Mock metrics for dashboard (replace with real evaluation results if available)
    # In a real app, these would be loaded from a file or database
    return jsonify({
        'accuracy': 0.987,
        'precision': 0.965,
        'recall': 0.978,
        'f1_score': 0.971,
        'confusion_matrix': [[45, 2], [3, 50]], # [[TN, FP], [FN, TP]]
        'history': {
            'accuracy': [0.6, 0.75, 0.85, 0.92, 0.95, 0.98],
            'val_accuracy': [0.55, 0.70, 0.82, 0.88, 0.94, 0.97],
            'loss': [0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
            'val_loss': [0.85, 0.65, 0.45, 0.35, 0.25, 0.15]
        },
        'predictions': {
            'tumor': len([x for x in history_db if x['is_tumor']]),
            'normal': len([x for x in history_db if not x['is_tumor']])
        }
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
