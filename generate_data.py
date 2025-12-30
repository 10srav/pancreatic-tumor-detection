import os
import numpy as np
import cv2
import random
import shutil

# Configuration
DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)
NUM_SAMPLES = 400  # 200 tumorous, 200 non-tumorous for better training

def add_texture(img, intensity=10):
    """Add realistic CT texture"""
    noise = np.random.normal(0, intensity, img.shape).astype(np.float32)
    textured = img.astype(np.float32) + noise
    return np.clip(textured, 0, 255).astype(np.uint8)

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
            
            # Body oval with varying intensity
            center = (IMG_SIZE[0]//2, IMG_SIZE[1]//2)
            axes = (random.randint(45, 55), random.randint(35, 45))
            angle = random.randint(-5, 5)
            body_intensity = random.randint(80, 100)
            cv2.ellipse(img, center, axes, angle, 0, 360, body_intensity, -1)
            
            # Spine (small circle at bottom)
            spine_center = (center[0], center[1] + 30)
            cv2.circle(img, spine_center, 8, 180, -1)
            
            # Pancreas (elongated shape in middle) - CONSISTENT INTENSITY for non-tumor
            panc_center = (center[0], center[1] - 10)
            panc_axes = (random.randint(18, 25), random.randint(8, 12))
            panc_angle = random.randint(-10, 10)
            panc_intensity = random.randint(140, 160)
            cv2.ellipse(img, panc_center, panc_axes, panc_angle, 0, 360, panc_intensity, -1)
            
            # 2. Add Tumor if category is tumorous - VERY DISTINCT FEATURES
            if category == 'tumorous':
                # Multiple tumor features to make it learnable
                
                # Primary tumor - MUCH brighter irregular shape
                t_x = panc_center[0] + random.randint(-8, 8)
                t_y = panc_center[1] + random.randint(-4, 4)
                t_radius = random.randint(5, 10)
                tumor_intensity = random.randint(220, 255)  # Very bright
                
                # Draw irregular tumor shape (multiple overlapping circles)
                cv2.circle(img, (t_x, t_y), t_radius, tumor_intensity, -1)
                cv2.circle(img, (t_x + random.randint(-3, 3), t_y + random.randint(-2, 2)), 
                          t_radius - 2, tumor_intensity - 10, -1)
                
                # Add tumor ring/halo effect (distinct boundary)
                cv2.circle(img, (t_x, t_y), t_radius + 2, tumor_intensity - 40, 2)
                
                # Add satellite lesions (small bright spots nearby) - 50% chance
                if random.random() > 0.5:
                    for _ in range(random.randint(1, 3)):
                        sat_x = t_x + random.randint(-12, 12)
                        sat_y = t_y + random.randint(-6, 6)
                        sat_r = random.randint(2, 3)
                        cv2.circle(img, (sat_x, sat_y), sat_r, tumor_intensity - 20, -1)
                
                # Add overall brightness increase in pancreas region (inflammation effect)
                mask = np.zeros_like(img)
                cv2.ellipse(mask, panc_center, (panc_axes[0]+5, panc_axes[1]+5), 
                           panc_angle, 0, 360, 20, -1)
                img = cv2.add(img, mask)
            else:
                # Non-tumorous: Add slight uniform texture but NO bright spots
                # Keep pancreas uniformly colored
                pass
            
            # 3. Add Noise (CT grain) - less for clearer distinction
            img = add_texture(img, intensity=3)
            
            # 4. Apply slight Gaussian blur for realism
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # 5. Save
            filename = f"{category}_{i}.jpg"
            cv2.imwrite(os.path.join(path, filename), img)
            
    print(f"Successfully generated data in '{DATASET_DIR}' folder.")
    print(f"  - Tumorous: {NUM_SAMPLES // 2} images")
    print(f"  - Non-tumorous: {NUM_SAMPLES // 2} images")

if __name__ == "__main__":
    generate_synthetic_data()
