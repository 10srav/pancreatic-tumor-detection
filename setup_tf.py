# TensorFlow path setup for Windows
import sys
import os

# Add TensorFlow installation path
TF_PATH = r'C:\tf'
if TF_PATH not in sys.path:
    sys.path.insert(0, TF_PATH)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
