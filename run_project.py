import os
import subprocess
import time
import webbrowser
import sys

def run_step(command, description):
    print(f"\n{'='*60}")
    print(f"[STEP] {description}...")
    print('='*60)
    try:
        subprocess.check_call(command, shell=True)
        print(f"\n[SUCCESS] {description} completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed: {e}")
        return False

def check_kaggle_setup():
    """Check if Kaggle API is configured"""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        return True
    # Windows alternative location
    kaggle_json_win = os.path.join(os.environ.get('USERPROFILE', ''), '.kaggle', 'kaggle.json')
    return os.path.exists(kaggle_json_win)

def main():
    print("="*60)
    print("   PANCREATIC TUMOR DETECTION SYSTEM")
    print("   Training with REAL CT Scan Data from Kaggle")
    print("="*60)

    # Check if we should use real data or need manual download
    use_real_data = True
    downloads_exist = os.path.exists("downloads") and os.listdir("downloads") if os.path.exists("downloads") else False
    dataset_exists = os.path.exists("dataset") and os.listdir("dataset/tumorous") if os.path.exists("dataset/tumorous") else False

    # 1. Download Real Dataset from Kaggle
    if not dataset_exists and not downloads_exist:
        print("\n[INFO] No dataset found. Attempting to download real CT scan data...")

        if check_kaggle_setup():
            print("[INFO] Kaggle API credentials found. Downloading dataset...")
            success = run_step("python download_dataset.py", "Downloading Real CT Scan Dataset from Kaggle")
            if not success:
                print("\n[WARNING] Kaggle download failed. Please download manually:")
                print("  1. Go to: https://www.kaggle.com/datasets/jayaprakashpondy/pancreatic-ct-images")
                print("  2. Download and extract to 'downloads/' folder")
                print("  3. Run this script again")
                use_real_data = False
        else:
            print("\n[INFO] Kaggle API not configured.")
            print("[INFO] To use real data:")
            print("  1. Create Kaggle account at kaggle.com")
            print("  2. Go to Account -> Create New API Token")
            print("  3. Place kaggle.json in ~/.kaggle/ folder")
            print("  4. Or manually download from:")
            print("     https://www.kaggle.com/datasets/jayaprakashpondy/pancreatic-ct-images")
            print("  5. Extract to 'downloads/' folder")
            print("\n[INFO] Falling back to synthetic data for demo...")
            use_real_data = False

    elif downloads_exist and not dataset_exists:
        # Process downloaded data
        print("\n[INFO] Found downloaded data. Processing...")
        run_step("python download_dataset.py", "Processing Downloaded CT Scan Data")
    else:
        print("\n[INFO] Dataset already exists.")

    # 2. Generate synthetic data if real data not available
    if not use_real_data and not dataset_exists:
        print("\n[INFO] Generating synthetic data for demonstration...")
        run_step("python generate_data.py", "Generating Synthetic CT Scan Data")

    # 3. Prepare Data (Resize & Normalize)
    if not os.path.exists("results/X_train.npy"):
        run_step("python 1_prepare_data.py", "Preparing Data (Resize & Normalize)")
    else:
        print("\n[INFO] Prepared data already exists.")

    # 4. Train Model (always retrain if using real data for first time)
    model_exists = os.path.exists("pancreas_model.h5")
    force_retrain = os.environ.get('RETRAIN', '0') == '1'

    if not model_exists or force_retrain:
        print("\n[INFO] Training model with data...")
        model_choice = os.environ.get('MODEL', 'custom').lower()
        if model_choice == 'vgg16':
            train_cmd = "python train_transfer_fixed.py"
            train_label = "Training VGG16 Transfer Learning Model"
        else:
            train_cmd = "python train_custom_cnn.py"
            train_label = "Training Custom CNN Model"
        run_step(train_cmd, train_label)
    else:
        print("\n[INFO] Model already exists. Use RETRAIN=1 to force retraining.")

    # 5. Evaluate Model
    print("\n[INFO] Evaluating model performance...")
    run_step("python 3_test.py", "Evaluating Model")

    # 6. Start Web Server
    print("\n" + "="*60)
    print("   STARTING WEB APPLICATION")
    print("="*60)
    print("\n[INFO] Opening browser in 3 seconds...")
    print("[INFO] Server will start at: http://localhost:5000")

    time.sleep(3)
    webbrowser.open("http://localhost:5000")

    # Run server
    print("\n[INFO] Starting Flask server... (Press Ctrl+C to stop)")
    os.system("python server.py")

if __name__ == "__main__":
    main()
