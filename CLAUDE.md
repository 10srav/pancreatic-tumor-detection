# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A deep learning system for detecting pancreatic tumors from CT scan images using CNN models. The system includes data preprocessing, model training (custom CNN and VGG16 transfer learning), evaluation, and a Flask web interface for predictions.

## Essential Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle (requires Kaggle API setup)
python download_dataset.py

# Or generate synthetic data for testing
python generate_data.py
```

### Data Pipeline
```bash
# Prepare data (CLAHE preprocessing, resize, normalize, split)
python 1_prepare_data.py
```

### Model Training
```bash
# Train custom CNN (default, recommended for grayscale CT scans)
python train_custom_cnn.py

# Train VGG16 transfer learning model (alternative)
python train_transfer_fixed.py

# Force retrain even if model exists
RETRAIN=1 python run_project.py

# Choose specific model
MODEL=vgg16 python run_project.py
```

### Evaluation and Testing
```bash
# Evaluate model on test set, generate confusion matrix and ROC curve
python 3_test.py

# Test single image prediction (edit 3_test.py to uncomment)
# predict_single_image("path/to/image.jpg")
```

### Web Application
```bash
# Start Flask server on port 5000
python server.py

# Run complete pipeline (download, prepare, train, evaluate, serve)
python run_project.py
```

### Environment Variables
- `MODEL_PATH`: Custom path to trained model file
- `RETRAIN=1`: Force model retraining
- `MODEL=custom|vgg16`: Choose training model type

## Architecture Overview

### Data Flow Pipeline

1. **Data Acquisition** → 2. **Preprocessing** → 3. **Training** → 4. **Evaluation** → 5. **Web Inference**

#### 1. Data Acquisition
- `download_dataset.py`: Downloads Kaggle pancreatic CT images dataset (765 tumor, 646 non-tumor)
- `generate_data.py`: Fallback synthetic data generator for demos
- Data organized in `dataset/tumorous/` and `dataset/non_tumorous/`

#### 2. Preprocessing (`1_prepare_data.py`)
Critical preprocessing steps that MUST be replicated at inference:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: `clipLimit=2.0, tileGridSize=(8,8)` - Enhances contrast in CT scans
- **Resize**: To 128x128 pixels
- **Gaussian Blur**: (3x3) kernel - Reduces noise
- **Normalization**: Pixel values to [0, 1] range
- **Stratified Split**: 80/20 train/test split maintaining class balance
- Outputs: `results/{X_train,y_train,X_test,y_test}.npy`

#### 3. Model Training
**Custom CNN (`train_custom_cnn.py`)** - Primary model:
- **Architecture**: 3 conv blocks (32→64→128 filters) with BatchNorm, Dropout, MaxPooling
- **Input**: 128x128x1 grayscale images
- **Output**: Single sigmoid neuron (binary classification)
- **Data Augmentation**: Rotation, shift, zoom, flip, brightness
- **Class Weighting**: Handles imbalanced dataset
- **Callbacks**: EarlyStopping (patience=15), ReduceLROnPlateau, ModelCheckpoint
- **Target**: 90% validation accuracy with auto-stop
- Saves: `pancreas_custom_cnn.h5` and `pancreas_model.h5` (unified name)

**VGG16 Transfer Learning (`train_transfer_fixed.py`)** - Alternative:
- **Architecture**: Frozen VGG16 base + custom top layers (GlobalAvgPool → Dense 256 → Dense 128 → Sigmoid)
- **Input**: 128x128x3 RGB (grayscale duplicated across channels)
- **Preprocessing**: VGG16-specific preprocessing via `preprocess_input`
- **Two-phase training**: Phase 1 (frozen base), Phase 2 (fine-tune last 4 layers if <85% accuracy)
- Converts grayscale to RGB, expects [0,255] range

#### 4. Evaluation (`3_test.py`)
- Loads test set and trained model
- Generates classification report (precision, recall, F1)
- Saves confusion matrix (`results/confusion_matrix.png`)
- Saves ROC curve (`results/roc_curve.png`)
- Supports single image prediction function

#### 5. Web Inference (`server.py`)
- **Flask API** with CORS enabled on port 5000
- **Model Loading**: Auto-detects from `pancreas_model.h5`, `pancreas_custom_cnn.h5`, or `pancreas_vgg16.h5`
- **Input Shape Inference**: Automatically detects grayscale (1 channel) vs RGB (3 channels)
- **Preprocessing**: Replicates training pipeline (CLAHE → resize → blur → normalize)
- **Prediction**: Threshold at 0.5, returns label + confidence
- **Endpoints**:
  - `POST /predict`: Upload image, get tumor prediction
  - `GET /metrics`: Training metrics (accuracy, precision, recall, F1, confusion matrix, history)
  - `GET /history`: Recent prediction history (last 50, most recent first)
  - `GET /health`: System status and model info
  - `GET /uploads/<filename>`: Serve uploaded images
  - `GET /results/<filename>`: Serve processed images
- **Persistence**: Saves prediction history to `results/prediction_history.json`, metrics to `results/metrics.json`

### Key Files Structure

```
├── 1_prepare_data.py         # Data preprocessing pipeline (CLAHE, resize, split)
├── train_custom_cnn.py       # Custom CNN training (grayscale, 128x128x1)
├── train_transfer_fixed.py   # VGG16 transfer learning (RGB, 128x128x3)
├── 3_test.py                 # Model evaluation and metrics generation
├── server.py                 # Flask web API for predictions
├── run_project.py            # Orchestrates full pipeline
├── setup_tf.py               # Windows TensorFlow path configuration
├── download_dataset.py       # Kaggle dataset downloader
├── generate_data.py          # Synthetic data generator
├── frontend/                 # HTML/CSS/JS web interface
│   ├── index.html           # Landing page
│   ├── upload.html          # Image upload interface
│   ├── dashboard.html       # Metrics visualization
│   ├── results.html         # Prediction history
│   └── js/                  # Frontend logic
├── dataset/                 # Training images
│   ├── tumorous/
│   └── non_tumorous/
├── results/                 # Preprocessed data, metrics, plots
│   ├── X_train.npy, y_train.npy
│   ├── X_test.npy, y_test.npy
│   ├── metrics.json
│   └── prediction_history.json
├── uploads/                 # User-uploaded images
└── pancreas_model.h5        # Trained model (unified name)
```

## Critical Implementation Details

### Preprocessing Consistency
**MUST** maintain identical preprocessing between training and inference:
- Both `1_prepare_data.py` and `server.py` use the same `apply_clahe()` function
- Same resize dimensions (128x128)
- Same Gaussian blur kernel (3x3)
- Same normalization (divide by 255.0)
- For VGG16: grayscale duplicated to 3 channels, expects [0,255] range

### Model Input Shape Handling
The `server.py` and `3_test.py` use `infer_model_settings()` to automatically detect:
- Input dimensions (width, height)
- Number of channels (1 for custom CNN, 3 for VGG16)
- Adjusts preprocessing accordingly

### Windows-Specific Setup
`setup_tf.py` is imported at the top of all training/inference scripts:
- Adds `C:\tf` to Python path for Windows TensorFlow installation
- Suppresses TensorFlow warnings via environment variables
- May need adjustment for different TensorFlow installation paths

### Class Imbalance Handling
Training scripts calculate class weights to handle imbalanced data:
```python
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
```

### Model Checkpointing
All training scripts save:
1. Model-specific checkpoint (e.g., `pancreas_custom_cnn.h5`)
2. Unified checkpoint (`pancreas_model.h5`) for web server
3. Metrics to `results/metrics_{model_name}.json` and `results/metrics.json`

## Dataset Information

- **Source**: Kaggle Pancreatic CT Images dataset
- **Classes**: Binary (Tumor / No Tumor)
- **Size**: ~1411 images (765 tumorous, 646 non-tumorous)
- **Format**: CT scan images (various formats, converted to grayscale)
- **Split**: 80% train, 20% test (stratified)

## Web Application

- **Frontend**: Static HTML/CSS/JS in `frontend/` directory
- **Backend**: Flask REST API serving predictions and static files
- **Pages**: Landing, Upload, Dashboard (metrics), Results (history), How It Works, About
- **Real-time**: Prediction history and metrics persist across server restarts
- **Display**: Shows preprocessed images alongside predictions

## Development Notes

- All training scripts include early stopping when reaching 90% validation accuracy
- Data augmentation is applied during training (rotation, zoom, flip, brightness)
- Metrics include accuracy, precision, recall, F1 score, confusion matrix, ROC-AUC
- Training history (loss/accuracy curves) saved as PNG plots
- The `run_project.py` script handles the complete workflow with intelligent skipping of completed steps
