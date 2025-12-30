"""
VGG16 Transfer Learning Script for Pancreatic Tumor Detection
Properly configured transfer learning with correct preprocessing.
"""
import setup_tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import json
import os

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 50

def load_and_convert_to_rgb():
    """Load grayscale data and convert to RGB for VGG16."""
    print("Loading data...")
    X_train = np.load('results/X_train.npy')
    y_train = np.load('results/y_train.npy')
    X_test = np.load('results/X_test.npy')
    y_test = np.load('results/y_test.npy')

    print("Converting grayscale to RGB...")
    # Data is already normalized to [0,1], convert to [0,255] for VGG preprocessing
    X_train_255 = (X_train * 255.0)
    X_test_255 = (X_test * 255.0)

    # Convert to RGB (3 channels)
    X_train_rgb = np.repeat(X_train_255, 3, axis=-1)
    X_test_rgb = np.repeat(X_test_255, 3, axis=-1)

    print(f"Training samples: {len(X_train_rgb)}")
    print(f"Test samples: {len(X_test_rgb)}")
    print(f"Input shape: {X_train_rgb.shape}")

    return X_train_rgb, y_train, X_test_rgb, y_test

def build_vgg16_model():
    """Build VGG16 transfer learning model with frozen base."""
    # Load VGG16 without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )

    # Freeze all base model layers
    base_model.trainable = False

    # Build model
    inputs = layers.Input(shape=(128, 128, 3))

    # VGG16 preprocessing (expects RGB in [0,255])
    x = layers.Lambda(lambda img: preprocess_input(img))(inputs)

    # Base model
    x = base_model(x, training=False)

    # Custom top layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

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
    print("VGG16 TRANSFER LEARNING TRAINING")
    print("=" * 60)

    # Load data
    X_train, y_train, X_test, y_test = load_and_convert_to_rgb()

    # Calculate class weights
    total = len(y_train)
    pos = np.sum(y_train)
    neg = total - pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"\nClass weights: Normal={weight_for_0:.2f}, Tumor={weight_for_1:.2f}")

    # Data augmentation (no normalization - data already in [0,255])
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
    print("\nBuilding VGG16 transfer learning model...")
    model, base_model = build_vgg16_model()
    model.summary()

    # Phase 1 Callbacks
    early_stop_p1 = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr_p1 = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        'pancreas_vgg16.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Phase 1: Train top layers only
    print("\n" + "=" * 50)
    print("PHASE 1: Training top layers (base frozen)")
    print("=" * 50)

    history1 = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS_PHASE1,
        validation_data=(X_test, y_test),
        callbacks=[early_stop_p1, reduce_lr_p1, checkpoint],
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        class_weight=class_weight
    )

    # Phase 2: Fine-tune
    best_val_acc_p1 = max(history1.history['val_accuracy'])
    print(f"\nPhase 1 best validation accuracy: {best_val_acc_p1*100:.2f}%")

    if best_val_acc_p1 < 0.85:
        print("\n" + "=" * 50)
        print("PHASE 2: Fine-tuning (unfreezing last 4 conv layers)")
        print("=" * 50)

        # Unfreeze last 4 layers
        base_model.trainable = True
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        # Recompile with lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop_p2 = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )

        reduce_lr_p2 = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=1
        )

        history2 = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS_PHASE2,
            validation_data=(X_test, y_test),
            callbacks=[early_stop_p2, reduce_lr_p2, checkpoint],
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            class_weight=class_weight
        )

        # Combine histories
        for key in history1.history:
            history1.history[key].extend(history2.history[key])

    # Load best model
    if os.path.exists('pancreas_vgg16.h5'):
        model = tf.keras.models.load_model('pancreas_vgg16.h5')
        print("\nLoaded best model from checkpoint.")

    # Save metrics
    metrics = save_metrics(model, X_test, y_test, history1, 'vgg16')

    # Print results
    best_val_acc = max(history1.history['val_accuracy'])
    best_train_acc = max(history1.history['accuracy'])

    print(f"\n{'=' * 50}")
    print("VGG16 TRANSFER LEARNING COMPLETE")
    print(f"{'=' * 50}")
    print(f"Best Training Accuracy:   {best_train_acc*100:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved as: pancreas_vgg16.h5")
    print(f"{'=' * 50}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history1.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history1.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('VGG16 - Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history1.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history1.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('VGG16 - Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_vgg16.png', dpi=150)
    print("Training plot saved to results/training_vgg16.png")
    plt.close()

    return metrics

if __name__ == "__main__":
    train()
