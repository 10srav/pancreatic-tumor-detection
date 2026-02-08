# Pancreatic Tumor Detection System

An automated system to detect pancreatic tumors from CT scan images using Deep Learning (CNN).

## Features

- **Data Preprocessing**: CLAHE contrast enhancement, Gaussian blur, normalization
- **CNN Model**: Custom CNN architecture optimized for grayscale medical images
- **Web Interface**: Flask-based web application for easy image upload and analysis
- **Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC, Confusion Matrix)

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 95.76% |
| Precision | 100.00% |
| Recall | 92.16% |
| F1 Score | 95.92% |

## Project Structure

```
pancreatic-tumor-detection/
├── frontend/              # Web interface (HTML/CSS/JS)
├── dataset/               # Training data (tumorous/non_tumorous)
├── results/               # Training results and metrics
├── uploads/               # User uploaded images
├── 1_prepare_data.py      # Data preprocessing pipeline
├── train_custom_cnn.py    # Custom CNN training script
├── train_transfer_fixed.py # VGG16 transfer learning (alternative)
├── 3_test.py              # Model evaluation script
├── server.py              # Flask web server
├── generate_data.py       # Synthetic data generator
├── download_dataset.py    # Kaggle dataset downloader
├── run_project.py         # Full pipeline runner
├── setup_tf.py            # TensorFlow configuration
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset (Optional)

If you have Kaggle API configured:
```bash
python download_dataset.py
```

Or generate synthetic data:
```bash
python generate_data.py
```

### 3. Prepare Data

```bash
python 1_prepare_data.py
```

### 4. Train Model

```bash
python train_custom_cnn.py
```

### 5. Run Web Application

```bash
python server.py
```

Open http://localhost:5000 in your browser.

## Usage

1. Open the web application in your browser
2. Upload a CT scan image (JPG/PNG)
3. Click "Analyze Image"
4. View the prediction result and confidence score

## Technical Details

- **Input**: 128x128 grayscale images
- **Architecture**: 3-block CNN with BatchNormalization, GlobalAveragePooling, and Dropout
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam (lr=0.0005)
- **Data Augmentation**: Rotation, shift, zoom, horizontal flip
- **GPU Support**: Configured for NVIDIA RTX 2050 (requires tensorflow[and-cuda])

## Dataset

Uses the Kaggle Pancreatic CT Images dataset:
- 765 tumor images
- 646 non-tumor images
- Total: 1411 CT scan images

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV
- Flask
- NumPy, Pandas, Scikit-learn
