import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

# Configuration
MODEL_PATH = 'laptop_pancreas_model.h5'
IMG_SIZE = (128, 128)

def predict_single_image(image_path):
    """
    Function to predict tumor on a single image file.
    """
    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return

    model = load_model(MODEL_PATH)
    
    try:
        # Read and Preprocess
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Could not read image.")
            return
            
        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized / 255.0
        img_reshaped = img_normalized.reshape(1, 128, 128, 1)
        
        # Predict
        prediction = model.predict(img_reshaped)
        score = prediction[0][0]
        
        print("\n--- Single Image Prediction ---")
        print(f"Image: {image_path}")
        if score > 0.5:
            print(f"Result: TUMOR DETECTED (Confidence: {score:.2%})")
        else:
            print(f"Result: No Tumor (Confidence: {(1-score):.2%})")
        print("-------------------------------\n")
        
    except Exception as e:
        print(f"Error: {e}")

def evaluate_full_test_set():
    if not os.path.exists('X_test.npy'):
        print("Test data not found!")
        return

    print("Loading test data...")
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    print("Predicting...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Tumorous', 'Tumorous']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
    plt.title('Confusion Matrix')
    plt.savefig('laptop_confusion_matrix.png')
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('laptop_roc_curve.png')
    
    print("Evaluation complete. Results saved.")

if __name__ == "__main__":
    # 1. Evaluate on full test set
    evaluate_full_test_set()
    
    # 2. Example of single image prediction (Uncomment to use)
    # predict_single_image("dataset/tumorous/sample_image.jpg")
