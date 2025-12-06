# Pancreatic Tumor Detection System

An automated system to detect pancreatic tumors from CT scan images using Deep Learning (CNN).

## Features
- **Data Preprocessing**: Noise reduction, CLAHE, Segmentation, Normalization.
- **CNN Model**: Custom CNN architecture for binary classification.
- **Web Interface**: Streamlit-based UI for easy interaction.
- **Evaluation**: Comprehensive metrics (Accuracy, ROC, Confusion Matrix).

## Project Structure
```
pancreatic-tumor-detection/
├── data/                  # Training and Test data
├── src/
│   ├── preprocessing.py   # Image processing pipeline
│   ├── model.py           # CNN Model architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── utils.py           # Synthetic data generator
├── app.py                 # Streamlit Web App
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Synthetic Data** (if you don't have real data):
    ```bash
    python src/utils.py
    ```

3.  **Train the Model**:
    ```bash
    python src/train.py
    ```
    This will save the model to `pancreas_tumor_model.h5`.

4.  **Evaluate the Model**:
    ```bash
    python src/evaluate.py
    ```

5.  **Run the Web App**:
    ```bash
    streamlit run app.py
    ```

## Usage
- Open the Streamlit app in your browser.
- Upload a CT scan image (JPG/PNG).
- Click "Analyze Image".
- View the result and confidence score.

## Technical Details
- **Input**: 224x224 RGB images.
- **Model**: Sequential CNN with 3 Convolutional blocks and Dense layers.
- **Loss**: Binary Crossentropy.
- **Optimizer**: Adam.
