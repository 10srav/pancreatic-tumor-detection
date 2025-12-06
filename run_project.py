import os
import subprocess
import time
import webbrowser

def run_step(command, description):
    print(f"\n[STEP] {description}...")
    try:
        subprocess.check_call(command, shell=True)
        print(f"[SUCCESS] {description} completed.")
    except subprocess.CalledProcessError:
        print(f"[ERROR] {description} failed.")
        exit(1)

def main():
    print("=== Pancreatic Tumor Detection System Launcher ===")
    
    # 1. Check for Data
    if not os.path.exists("dataset"):
        print("Dataset not found. Generating synthetic data...")
        run_step("python generate_data.py", "Generating Synthetic Data")
    
    # 2. Prepare Data
    if not os.path.exists("X_train.npy"):
        run_step("python 1_prepare_laptop_data.py", "Preparing Data (Resize & Normalize)")
        
    # 3. Train Model
    if not os.path.exists("laptop_pancreas_model.h5"):
        run_step("python 2_train_laptop_model.py", "Training Model")
    else:
        print("[INFO] Model already exists. Skipping training.")

    # 4. Run Server
    print("\n[STEP] Starting Web Server...")
    print("Opening browser in 5 seconds...")
    
    # Open browser after a short delay to allow server to start
    time.sleep(2)
    webbrowser.open("http://localhost:5000")
    
    # Run server
    os.system("python server.py")

if __name__ == "__main__":
    main()
