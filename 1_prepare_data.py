import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)

def apply_clahe(img):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance contrast in CT scan images.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_ct_image(img):
    """
    Preprocess CT scan image for better model performance.
    1. Apply CLAHE for contrast enhancement
    2. Resize to target size
    3. Apply Gaussian blur to reduce noise
    """
    # Apply CLAHE
    img = apply_clahe(img)

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # Light Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img

def load_and_process_data():
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' not found or empty.")
        print("Please run 'python download_dataset.py' first to download the dataset.")
        return

    data = []
    labels = []
    categories = ['non_tumorous', 'tumorous']  # 0: Non-Tumor, 1: Tumor

    print("="*50)
    print("PREPARING DATA WITH ENHANCED PREPROCESSING")
    print("="*50)
    print("\nScanning dataset...")

    for category in categories:
        path = os.path.join(DATASET_DIR, category)
        class_num = categories.index(category)

        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue

        images = os.listdir(path)
        print(f"\nFound {len(images)} images in {category}")

        for img_name in tqdm(images, desc=f"Processing {category}"):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    continue

                # Apply enhanced preprocessing
                processed_img = preprocess_ct_image(img_array)

                data.append(processed_img)
                labels.append(class_num)
            except Exception as e:
                pass

    if len(data) == 0:
        print("No images found.")
        return

    X = np.array(data).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    y = np.array(labels)
    X = X / 255.0  # Normalize to [0, 1]

    print(f"\n{'='*50}")
    print(f"Total Images: {len(X)}")
    print(f"  - Non-Tumorous: {np.sum(y == 0)}")
    print(f"  - Tumorous: {np.sum(y == 1)}")
    print(f"Image Shape: {X.shape[1:]}")
    print(f"{'='*50}")

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class balance in split
    )

    print(f"\nTraining set: {len(X_train)} images")
    print(f"  - Non-Tumorous: {np.sum(y_train == 0)}")
    print(f"  - Tumorous: {np.sum(y_train == 1)}")

    print(f"\nTest set: {len(X_test)} images")
    print(f"  - Non-Tumorous: {np.sum(y_test == 0)}")
    print(f"  - Tumorous: {np.sum(y_test == 1)}")

    # Save to results folder
    os.makedirs('results', exist_ok=True)
    np.save('results/X_train.npy', X_train)
    np.save('results/y_train.npy', y_train)
    np.save('results/X_test.npy', X_test)
    np.save('results/y_test.npy', y_test)

    print(f"\n{'='*50}")
    print("Data saved to results folder:")
    print("  - results/X_train.npy")
    print("  - results/y_train.npy")
    print("  - results/X_test.npy")
    print("  - results/y_test.npy")
    print(f"{'='*50}")

if __name__ == "__main__":
    load_and_process_data()
