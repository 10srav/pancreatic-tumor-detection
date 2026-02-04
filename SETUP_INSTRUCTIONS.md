# Setup Instructions for Pancreatic Tumor Detection System

This guide will help you set up and run the Pancreatic Tumor Detection System on your local machine.

## Prerequisites

- **Python 3.8 - 3.12** (Python 3.12 recommended)
- **Git** (for cloning the repository)
- **pip** (Python package installer)

## Step 1: Clone the Repository

```bash
git clone https://github.com/10srav/pancreatic-tumor-detection.git
cd pancreatic-tumor-detection
```

## Step 2: Install Dependencies

### Important: NumPy Version

This project requires NumPy 1.x (NOT NumPy 2.x). Install dependencies with:

```bash
pip install -r requirements.txt
```

If you encounter NumPy compatibility issues, ensure NumPy 1.x is installed:

```bash
pip uninstall numpy -y
pip install "numpy<2"
```

## Step 3: Download or Generate Dataset

### Option A: Download Real Dataset from Kaggle (Recommended)

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → Create New API Token
3. Place `kaggle.json` in `~/.kaggle/` folder (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
4. Run the download script:

```bash
python download_dataset.py
```

### Option B: Generate Synthetic Data (For Testing)

```bash
python generate_data.py
```

## Step 4: Prepare Data

This step preprocesses the images with CLAHE enhancement, resizing, and normalization:

```bash
python 1_prepare_data.py
```

## Step 5: Train Model (Optional)

If you want to train the model from scratch:

### Option A: Custom CNN (Recommended)

```bash
python train_custom_cnn.py
```

### Option B: VGG16 Transfer Learning

```bash
python train_transfer_fixed.py
```

**Note**: A pre-trained model (`pancreas_model.h5`) is already included in the repository, so you can skip this step if you just want to run the application.

## Step 6: Run the Web Application

Start the Flask server:

```bash
python server.py
```

The application will be available at:
- **http://localhost:5000**
- **http://127.0.0.1:5000**

## Step 7: Use the Application

1. Open your browser and go to http://localhost:5000
2. Click "Upload Scan" in the navigation menu
3. Upload a CT scan image (JPG or PNG format)
4. Click "Analyze Scan"
5. View the prediction results

## Complete Workflow (All Steps)

Alternatively, run the complete pipeline with a single command:

```bash
python run_project.py
```

This will:
1. Download/generate dataset (if needed)
2. Prepare data (if needed)
3. Train model (if needed)
4. Evaluate model
5. Start web server
6. Open browser automatically

## Common Issues and Solutions

### Issue 1: NumPy Version Incompatibility

**Error**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**Solution**:
```bash
pip uninstall numpy -y
pip install "numpy<2"
```

If you have a custom TensorFlow installation (e.g., in `C:\tf`), also remove NumPy from there:
```bash
# Windows
rm -rf C:\tf\numpy C:\tf\numpy-*.dist-info

# Linux/Mac
rm -rf /path/to/tf/numpy /path/to/tf/numpy-*.dist-info
```

### Issue 2: Upload Not Working in Browser

**Solution**: Clear your browser cache:
- **Windows**: Press `Ctrl + F5` or `Ctrl + Shift + R`
- **Mac**: Press `Cmd + Shift + R`

Or manually:
1. Press `F12` to open Developer Tools
2. Right-click the refresh button → "Empty Cache and Hard Refresh"

### Issue 3: TensorFlow Import Error

**Solution**: Make sure you have compatible versions:
```bash
pip install tensorflow>=2.10.0 numpy<2
```

### Issue 4: Model File Not Found

**Error**: `Model not found!`

**Solution**: Either download the pre-trained model or train a new one:
```bash
python train_custom_cnn.py
```

### Issue 5: localhost Not Accessible

The server now binds to `0.0.0.0`, making it accessible via:
- http://localhost:5000
- http://127.0.0.1:5000
- http://[your-local-ip]:5000

If you still have issues, check your firewall settings.

## Project Structure

```
pancreatic-tumor-detection/
├── frontend/              # Web interface (HTML/CSS/JS)
├── dataset/               # Training images (not in git)
├── results/               # Preprocessed data and metrics (not in git)
├── uploads/               # User uploaded images (not in git)
├── server.py              # Flask web server
├── train_custom_cnn.py    # Custom CNN training
├── train_transfer_fixed.py # VGG16 transfer learning
├── 1_prepare_data.py      # Data preprocessing
├── 3_test.py              # Model evaluation
├── run_project.py         # Complete pipeline runner
├── pancreas_model.h5      # Pre-trained model (not in git)
├── CLAUDE.md              # Developer documentation
├── README.md              # Project overview
└── requirements.txt       # Python dependencies
```

## System Requirements

- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: 2GB free space (more if storing dataset)
- **GPU**: Optional (CUDA-compatible GPU will speed up training)

## Development

For detailed development information, see [CLAUDE.md](CLAUDE.md).

### Running Tests

```bash
python 3_test.py
```

### Evaluation Metrics

After training or testing, check:
- `results/metrics.json` - Model performance metrics
- `results/confusion_matrix.png` - Confusion matrix visualization
- `results/roc_curve.png` - ROC curve
- `results/training_custom_cnn.png` - Training history plots

## Dataset Information

- **Source**: Kaggle Pancreatic CT Images
- **Classes**: Binary (Tumor / No Tumor)
- **Size**: ~1411 images (765 tumor, 646 non-tumor)
- **Format**: CT scan images
- **Split**: 80% train, 20% test (stratified)

## API Endpoints

The Flask server provides these endpoints:

- `GET /` - Main page
- `GET /upload.html` - Upload page
- `GET /dashboard.html` - Metrics dashboard
- `GET /results.html` - Prediction history
- `POST /predict` - Upload image for prediction
- `GET /health` - Server health check
- `GET /metrics` - Training metrics
- `GET /history` - Prediction history

## Support

For issues or questions:
1. Check this SETUP_INSTRUCTIONS.md
2. Review CLAUDE.md for developer documentation
3. Check the GitHub Issues page

## License

See LICENSE file in the repository.

## Credits

Developed for Vignan's Institute of Engineering for Women (A)
Department of Computer Science & Engineering
