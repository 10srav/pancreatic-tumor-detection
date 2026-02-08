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

# Configure GPU memory growth to avoid OOM on RTX 2050
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {[g.name for g in gpus]}")
    else:
        print("No GPU detected, using CPU.")
except Exception as e:
    print(f"GPU config note: {e}")
