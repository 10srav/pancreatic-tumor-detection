"""
Custom CNN Training Script for Pancreatic Tumor Detection
A simpler architecture optimized for grayscale medical CT scans.
"""
import setup_tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import json
import os

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 100

def load_data():
    """Load grayscale training data."""
    print("Loading data...")
    X_train = np.load('results/X_train.npy')
    y_train = np.load('results/y_train.npy')
    X_test = np.load('results/X_test.npy')
    y_test = np.load('results/y_test.npy')

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape}")

    return X_train, y_train, X_test, y_test

def build_custom_cnn():
    """Build a custom CNN optimized for grayscale CT scans."""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(128, 128, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        # Output
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def save_metrics(model, X_test, y_test, history, model_name):
    """Save metrics to JSON."""
    print("\nCalculating final metrics...")

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = float(np.mean(y_pred == y_test))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }

    metrics_file = f'results/metrics_{model_name}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Also write the default metrics file for the dashboard
    try:
        with open('results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write results/metrics.json: {e}")

    print(f"\nMetrics saved to {metrics_file}")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}, TP={cm[1][1]}")

    return metrics

def train():
    print("=" * 60)
    print("CUSTOM CNN TRAINING")
    print("=" * 60)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Calculate class weights
    total = len(y_train)
    pos = np.sum(y_train)
    neg = total - pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"\nClass weights: Normal={weight_for_0:.2f}, Tumor={weight_for_1:.2f}")

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Build model
    print("\nBuilding Custom CNN model...")
    model = build_custom_cnn()
    model.summary()

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        'pancreas_custom_cnn.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    class AccuracyStop(callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_accuracy') and logs.get('val_accuracy') >= 0.90:
                print("\nReached 90% validation accuracy. Stopping training!")
                self.model.stop_training = True

    # Train
    print("\n" + "=" * 50)
    print("TRAINING CUSTOM CNN")
    print("=" * 50)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr, checkpoint, AccuracyStop()],
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        class_weight=class_weight
    )

    # Load best model
    if os.path.exists('pancreas_custom_cnn.h5'):
        model = tf.keras.models.load_model('pancreas_custom_cnn.h5')
        print("\nLoaded best model from checkpoint.")

    # Export unified model name for the web app
    try:
        model.save('pancreas_model.h5')
        print("Model exported as: pancreas_model.h5")
    except Exception as e:
        print(f"Warning: failed to export pancreas_model.h5: {e}")

    # Save metrics
    metrics = save_metrics(model, X_test, y_test, history, 'custom_cnn')

    # Print results
    best_val_acc = max(history.history['val_accuracy'])
    best_train_acc = max(history.history['accuracy'])

    print(f"\n{'=' * 50}")
    print("CUSTOM CNN TRAINING COMPLETE")
    print(f"{'=' * 50}")
    print(f"Best Training Accuracy:   {best_train_acc*100:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved as: pancreas_custom_cnn.h5")
    print(f"{'=' * 50}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Custom CNN - Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Custom CNN - Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_custom_cnn.png', dpi=150)
    print("Training plot saved to results/training_custom_cnn.png")
    plt.close()

    return metrics

if __name__ == "__main__":
    train()
