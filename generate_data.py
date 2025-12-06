import os
import numpy as np
import cv2
import random
import shutil

# Configuration
DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)
NUM_SAMPLES = 200 # 100 tumorous, 100 non-tumorous

def generate_synthetic_data():
    print(f"Generating {NUM_SAMPLES} synthetic images...")
    
    # Clean up existing if any (to ensure fresh start)
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    
    categories = ['tumorous', 'non_tumorous']
    
    for category in categories:
        path = os.path.join(DATASET_DIR, category)
        os.makedirs(path, exist_ok=True)
        
        count = NUM_SAMPLES // 2
        
        for i in range(count):
            # 1. Create background (simulating body cross-section)
            img = np.zeros(IMG_SIZE, dtype=np.uint8)
            
            # Body oval
            center = (IMG_SIZE[0]//2, IMG_SIZE[1]//2)
            axes = (random.randint(45, 55), random.randint(35, 45))
            angle = random.randint(-5, 5)
            cv2.ellipse(img, center, axes, angle, 0, 360, 100, -1)
            
            # Spine (small circle at bottom)
            spine_center = (center[0], center[1] + 30)
            cv2.circle(img, spine_center, 8, 180, -1)
            
            # Pancreas (elongated shape in middle)
            panc_center = (center[0], center[1] - 10)
            panc_axes = (random.randint(15, 20), random.randint(6, 10))
            panc_angle = random.randint(-10, 10)
            cv2.ellipse(img, panc_center, panc_axes, panc_angle, 0, 360, 160, -1)
            
            # 2. Add Tumor if category is tumorous
            if category == 'tumorous':
                # Tumor is usually a different intensity spot within the pancreas
                # Random position within pancreas area (roughly)
                t_x = panc_center[0] + random.randint(-10, 10)
                t_y = panc_center[1] + random.randint(-3, 3)
                t_radius = random.randint(3, 6)
                
                # Tumor intensity: can be darker (hypodense) or brighter (hyperdense)
                # Let's make it brighter for visibility in this demo
                cv2.circle(img, (t_x, t_y), t_radius, 220, -1)
            
            # 3. Add Noise (CT grain)
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            # 4. Save
            filename = f"{category}_{i}.jpg"
            cv2.imwrite(os.path.join(path, filename), img)
            
    print(f"Successfully generated data in '{DATASET_DIR}' folder.")

if __name__ == "__main__":
    generate_synthetic_data()
