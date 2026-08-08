# =============================================================================
# Lab 4: CNN for Image Classification - Apple Disease Dataset
# Using tf.data.AUTOTUNE for data pipeline optimization
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# ======================== 1. Configuration ========================
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4
CLASS_NAMES = ['apple_scab', 'black_rot', 'cedar_apple_rust', 'healthy']

# Use AUTOTUNE to let TensorFlow decide optimal number of parallel operations
AUTOTUNE = tf.data.AUTOTUNE

print("TensorFlow Version:", tf.__version__)
print(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"Classes: {CLASS_NAMES}")

# ======================== 2. Load Dataset with AUTOTUNE ========================
print("\n--- Loading Datasets ---")

# Load training dataset from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training'
)

# Load validation dataset from training directory (20% split)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

class_names = train_ds.class_names
print(f"Detected Classes: {class_names}")

# ======================== 3. AUTOTUNE - Data Pipeline Optimization ========================
# The AUTOTUNE function dynamically tunes the buffer size for
# prefetching, caching, and shuffling at runtime for best performance.

def configure_for_performance(ds, shuffle=False):
    """Apply caching, shuffling, and prefetching with AUTOTUNE."""
    ds = ds.cache()                          # Cache dataset in memory
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)    # Shuffle with buffer
    ds = ds.prefetch(buffer_size=AUTOTUNE)   # Prefetch next batch while training
    return ds

train_ds = configure_for_performance(train_ds, shuffle=True)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

print("[OK] AUTOTUNE applied: cache() + prefetch(AUTOTUNE) for optimal pipeline performance")

# ======================== 4. Visualize Sample Images ========================
plt.figure(figsize=(12, 8))
for images, labels in train_ds.take(1):
    for i in range(min(16, len(images))):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]], fontsize=9)
        plt.axis("off")
plt.suptitle("Sample Training Images - Apple Disease Dataset", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'sample_images.png'), dpi=100)
plt.show()

# ======================== 5. Data Augmentation Layer ========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# ======================== 6. Build CNN Model ========================
print("\n--- Building CNN Model ---")

model = models.Sequential([
    # Input and Normalization
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Rescaling(1.0 / 255),           # Normalize pixel values to [0, 1]
    data_augmentation,                      # Apply data augmentation

    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Convolutional Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')   # 4-class output
])

model.summary()

# ======================== 7. Compile the Model ========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("[OK] Model compiled with Adam optimizer and Sparse Categorical Crossentropy loss")

# ======================== 8. Train the Model ========================
print("\n--- Training the Model ---")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# ======================== 9. Plot Training History ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss Plot
axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Training History - Apple Disease CNN', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'training_history.png'), dpi=100)
plt.show()

# ======================== 10. Evaluate on Test Set ========================
print("\n--- Evaluating on Test Set ---")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\n>> Test Loss:     {test_loss:.4f}")
print(f">> Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ======================== 11. Classification Report ========================
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# ======================== 12. Confusion Matrix ========================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Apple Disease Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'), dpi=100)
plt.show()

# ======================== 13. Predict on Sample Test Images ========================
plt.figure(figsize=(14, 8))
for images, labels in test_ds.take(1):
    predictions = model.predict(images, verbose=0)
    for i in range(min(16, len(images))):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[labels[i]]
        confidence = np.max(predictions[i]) * 100
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f"P: {pred_class}\nT: {true_class}\n({confidence:.1f}%)",
                  fontsize=8, color=color)
        plt.axis("off")

plt.suptitle("Predictions on Test Images (Green=Correct, Red=Wrong)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'predictions.png'), dpi=100)
plt.show()

# ======================== 14. Save the Model ========================
model.save(os.path.join(os.path.dirname(__file__), 'apple_disease_cnn_model.h5'))
print("\n[OK] Model saved as 'apple_disease_cnn_model.h5'")

print("\n" + "=" * 60)
print("  Lab 4 Complete: CNN for Apple Disease Classification")
print(f"  Final Test Accuracy: {test_accuracy*100:.2f}%")
print("  AUTOTUNE used for: cache(), prefetch(), data pipeline")
print("=" * 60)
