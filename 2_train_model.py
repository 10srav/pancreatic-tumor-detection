import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# Configuration
IMG_SIZE = (128, 128, 1)
BATCH_SIZE = 32
EPOCHS = 20

def build_lightweight_model():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten & Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    # Load Data
    if not os.path.exists('X_train.npy'):
        print("Data files not found! Run 1_prepare_laptop_data.py first.")
        return

    print("Loading data from .npy files...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    print(f"Training on {len(X_train)} samples, Validating on {len(X_test)} samples.")

    # Build Model
    model = build_lightweight_model()
    model.summary()

    # Callbacks
    # Stop if accuracy reaches 98% on validation to save time
    class AccuracyStop(callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_accuracy') >= 0.98):
                print("\nReached 98% validation accuracy. Stopping training!")
                self.model.stop_training = True

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, AccuracyStop()]
    )

    # Save Model
    model.save('laptop_pancreas_model.h5')
    print("Model saved as laptop_pancreas_model.h5")

    # Plot
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('laptop_training_plot.png')
    print("Training plot saved.")

if __name__ == "__main__":
    train()
