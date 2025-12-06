import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)

def load_and_process_data():
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' not found or empty.")
        print("Please run 'python generate_data.py' first to create synthetic data.")
        return

    data = []
    labels = []
    categories = ['non_tumorous', 'tumorous'] # 0: Non-Tumor, 1: Tumor
    
    print("Scanning dataset...")
    for category in categories:
        path = os.path.join(DATASET_DIR, category)
        class_num = categories.index(category)
        
        if not os.path.exists(path):
            continue
            
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    continue
                
                new_array = cv2.resize(img_array, IMG_SIZE)
                data.append(new_array)
                labels.append(class_num)
            except Exception as e:
                pass

    if len(data) == 0:
        print("No images found.")
        return

    X = np.array(data).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    y = np.array(labels)
    X = X / 255.0 # Normalize

    print(f"Total Images: {len(X)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    
    print("Data saved to .npy files.")

if __name__ == "__main__":
    load_and_process_data()
