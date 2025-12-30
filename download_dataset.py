"""
Pancreatic Tumor Dataset Downloader & Organizer
Downloads real CT scan data from Kaggle and organizes it for training.

BEST DATASET: jayaprakashpondy/pancreatic-ct-images
- 1411 real CT scan images
- Balanced: ~54% tumor, ~46% healthy
- Pre-categorized into tumor/healthy folders

PREREQUISITES:
1. Install Kaggle: pip install kaggle
2. Get your Kaggle API key:
   - Go to kaggle.com -> Your Profile -> Account -> Create New API Token
   - This downloads kaggle.json
   - Place it in: C:/Users/<YourUsername>/.kaggle/kaggle.json
"""

import os
import shutil
import zipfile
import numpy as np
import cv2
from tqdm import tqdm

# Try importing optional libraries
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Note: nibabel not installed. Install with: pip install nibabel")

# Configuration
DATASET_DIR = "dataset"
OUTPUT_TUMOROUS = os.path.join(DATASET_DIR, "tumorous")
OUTPUT_NON_TUMOROUS = os.path.join(DATASET_DIR, "non_tumorous")
IMG_SIZE = (128, 128)

def setup_directories():
    """Create output directories"""
    # Clear existing synthetic data
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        print(f"Cleared existing dataset folder")

    os.makedirs(OUTPUT_TUMOROUS, exist_ok=True)
    os.makedirs(OUTPUT_NON_TUMOROUS, exist_ok=True)
    print(f"Created directories: {OUTPUT_TUMOROUS}, {OUTPUT_NON_TUMOROUS}")

def download_from_kaggle():
    """Download dataset using Kaggle API"""
    try:
        import kaggle
        print("Downloading Pancreatic CT Images dataset from Kaggle...")
        print("This dataset contains 1411 real CT scan images")

        # Primary dataset - best quality for pancreatic tumor detection
        datasets = [
            "jayaprakashpondy/pancreatic-ct-images",  # Best: 1411 images, balanced
            "mugabireklphpnst/pancreas-tumor-detection",
            "salihayesilyurt/pancreas-ct",
        ]

        for dataset in datasets:
            try:
                print(f"\nTrying: {dataset}")
                kaggle.api.dataset_download_files(dataset, path="downloads", unzip=True)
                print(f"Successfully downloaded: {dataset}")
                return "downloads"
            except Exception as e:
                print(f"Failed to download {dataset}: {e}")
                continue

        print("Could not download from Kaggle. Please download manually.")
        return None

    except ImportError:
        print("Kaggle library not installed. Install with: pip install kaggle")
        return None
    except Exception as e:
        print(f"Kaggle API error: {e}")
        print("\nTo fix this:")
        print("1. Go to kaggle.com -> Account -> Create New API Token")
        print("2. Place kaggle.json in C:/Users/<YourUsername>/.kaggle/")
        return None

def process_nifti_files(data_path):
    """
    Process NIfTI (.nii or .nii.gz) files from Medical Decathlon format.
    Extracts 2D slices and classifies based on tumor masks.
    """
    if not NIBABEL_AVAILABLE:
        print("nibabel required for NIfTI processing. Install with: pip install nibabel")
        return False
    
    # Find imagesTr and labelsTr folders
    images_dir = None
    labels_dir = None
    
    for root, dirs, files in os.walk(data_path):
        if "imagesTr" in dirs:
            images_dir = os.path.join(root, "imagesTr")
        if "labelsTr" in dirs:
            labels_dir = os.path.join(root, "labelsTr")
    
    if not images_dir or not labels_dir:
        print("Could not find imagesTr/labelsTr directories")
        return False
    
    print(f"Found images: {images_dir}")
    print(f"Found labels: {labels_dir}")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.nii', '.nii.gz'))])
    
    tumor_count = 0
    normal_count = 0
    
    for img_file in tqdm(image_files, desc="Processing NIfTI files"):
        try:
            # Load image and corresponding mask
            img_path = os.path.join(images_dir, img_file)
            mask_file = img_file.replace("_0000", "")  # Medical Decathlon naming convention
            mask_path = os.path.join(labels_dir, mask_file)
            
            if not os.path.exists(mask_path):
                continue
            
            img_nii = nib.load(img_path)
            mask_nii = nib.load(mask_path)
            
            img_data = img_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
            
            # Process each slice (assuming axial slices along axis 2)
            num_slices = img_data.shape[2]
            
            for slice_idx in range(num_slices):
                img_slice = img_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]
                
                # Apply windowing (Abdomen window: Level=40, Width=400)
                level, width = 40, 400
                min_val = level - width // 2
                max_val = level + width // 2
                img_slice = np.clip(img_slice, min_val, max_val)
                img_slice = ((img_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                
                # Resize
                img_resized = cv2.resize(img_slice, IMG_SIZE)
                
                # Check for tumor (label 2 in Medical Decathlon = tumor)
                has_tumor = np.any(mask_slice == 2)
                
                # Save
                if has_tumor:
                    save_path = os.path.join(OUTPUT_TUMOROUS, f"tumor_{tumor_count}.jpg")
                    tumor_count += 1
                else:
                    # Limit non-tumor slices to balance dataset
                    if normal_count < tumor_count * 2:  # Keep 2:1 ratio max
                        save_path = os.path.join(OUTPUT_NON_TUMOROUS, f"normal_{normal_count}.jpg")
                        normal_count += 1
                    else:
                        continue
                
                cv2.imwrite(save_path, img_resized)
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"\nExtracted: {tumor_count} tumor slices, {normal_count} normal slices")
    return tumor_count > 0 or normal_count > 0

def process_png_jpg_files(data_path):
    """
    Process already-extracted 2D image files (PNG/JPG).
    Handles jayaprakashpondy/pancreatic-ct-images dataset structure.
    """
    tumor_count = 0
    normal_count = 0

    # Keywords indicating tumor images
    tumor_keywords = ['tumor', 'tumour', 'cancer', 'positive', 'malignant', 'abnormal', 'sick']
    # Keywords indicating healthy/normal images
    normal_keywords = ['normal', 'healthy', 'negative', 'benign', 'non_tumor', 'nontumor', 'no_tumor']

    print("\nScanning for CT scan images...")

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                file_path = os.path.join(root, file)
                # Get full path for keyword matching
                full_path_lower = file_path.lower()
                folder_name = os.path.basename(root).lower()

                try:
                    # Read image - try grayscale first, then color conversion
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        img = cv2.imread(file_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    if img is None:
                        continue

                    # Apply contrast enhancement for medical images
                    img = cv2.equalizeHist(img)

                    # Resize to model input size
                    img_resized = cv2.resize(img, IMG_SIZE)

                    # Determine category based on folder name or file path
                    is_tumor = any(kw in full_path_lower for kw in tumor_keywords)
                    is_normal = any(kw in full_path_lower for kw in normal_keywords)

                    if is_tumor and not is_normal:
                        save_path = os.path.join(OUTPUT_TUMOROUS, f"tumor_{tumor_count}.jpg")
                        cv2.imwrite(save_path, img_resized)
                        tumor_count += 1
                    elif is_normal and not is_tumor:
                        save_path = os.path.join(OUTPUT_NON_TUMOROUS, f"normal_{normal_count}.jpg")
                        cv2.imwrite(save_path, img_resized)
                        normal_count += 1
                    # If ambiguous, skip the image

                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue

    print(f"\n{'='*50}")
    print(f"DATASET PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Tumor images:  {tumor_count}")
    print(f"Normal images: {normal_count}")
    print(f"Total images:  {tumor_count + normal_count}")
    print(f"{'='*50}")

    return tumor_count > 0 or normal_count > 0

def manual_download_instructions():
    """Print instructions for manual download"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
Since automatic download failed, please download manually:

OPTION 1 (RECOMMENDED - Best dataset for this project):
1. Go to: https://www.kaggle.com/datasets/jayaprakashpondy/pancreatic-ct-images
2. Click 'Download' button (you need a Kaggle account)
3. Extract the ZIP to: downloads/
4. This dataset has 1411 real CT images (tumor + healthy)

OPTION 2 (Alternative - Pancreas Tumor Detection):
1. Go to: https://www.kaggle.com/datasets/mugabireklphpnst/pancreas-tumor-detection
2. Click 'Download' button
3. Extract the ZIP to: downloads/

OPTION 3 (Medical Decathlon - 3D NIfTI format):
1. Go to: https://www.kaggle.com/datasets/salihayesilyurt/pancreas-ct
2. Download and extract to: downloads/

After downloading, run this script again to process the data.
    """)
    print("="*60)

def main():
    print("="*60)
    print("PANCREATIC TUMOR DATASET DOWNLOADER")
    print("="*60)
    
    # Setup
    setup_directories()
    
    # Check if downloads folder already has data
    if os.path.exists("downloads") and os.listdir("downloads"):
        print("Found existing downloads folder. Processing...")
        data_path = "downloads"
    else:
        # Try Kaggle download
        data_path = download_from_kaggle()
        
        if data_path is None:
            manual_download_instructions()
            return
    
    # Process the downloaded data
    print("\nProcessing downloaded data...")
    
    # Try NIfTI first (Medical Decathlon format)
    success = False
    if NIBABEL_AVAILABLE:
        success = process_nifti_files(data_path)
    
    # If NIfTI didn't work, try PNG/JPG
    if not success:
        success = process_png_jpg_files(data_path)
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS! Dataset is ready.")
        print(f"Tumor images: {OUTPUT_TUMOROUS}")
        print(f"Normal images: {OUTPUT_NON_TUMOROUS}")
        print("\nNext step: Run 'python run_project.py' to train the model.")
        print("="*60)
    else:
        print("\nNo valid images found. Please check the download folder.")
        manual_download_instructions()

if __name__ == "__main__":
    main()
