import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix
from data_preprocessing import load_images_and_labels, encode_labels, split_data
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
DATASET_PATH = r'E:\OCR_Calculator\Dataset\img_dataset'
images, labels, label_maps = load_images_and_labels(DATASET_PATH)

# Encode labels
encoded_labels, _ = encode_labels(labels)

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, encoded_labels)

# Debug initial shape
print(f"Original X_train shape: {X_train.shape}")
print(f"Original X_val shape: {X_val.shape}")
print(f"Original X_test shape: {X_test.shape}")

# Normalize and resize images to 96x96
X_train = np.array([tf.image.resize(img[..., np.newaxis], (96, 96)).numpy() for img in X_train])
X_val = np.array([tf.image.resize(img[..., np.newaxis], (96, 96)).numpy() for img in X_val])
X_test = np.array([tf.image.resize(img[..., np.newaxis], (96, 96)).numpy() for img in X_test])

# Debug shapes after resizing
print(f"X_train shape after resizing: {X_train.shape}")  # Should be (40800, 96, 96, 1)
print(f"X_val shape after resizing: {X_val.shape}")  # Should be (5100, 96, 96, 1)
print(f"X_test shape after resizing: {X_test.shape}")  # Should be (5100, 96, 96, 1)

# Convert grayscale to RGB (3 channels)
X_train = np.repeat(X_train, 3, -1)
X_val = np.repeat(X_val, 3, -1)
X_test = np.repeat(X_test, 3, -1)

# Debug shapes after RGB conversion
print(f"X_train shape after RGB conversion: {X_train.shape}")  # Should be (40800, 96, 96, 3)
print(f"X_val shape after RGB conversion: {X_val.shape}")  # Should be (5100, 96, 96, 3)
print(f"X_test shape after RGB conversion: {X_test.shape}")  # Should be (5100, 96, 96, 3)

# Check that the number of samples matches
if X_train.shape[0] != y_train.shape[0]:
    raise ValueError(f"Mismatch: X_train has {X_train.shape[0]} samples but y_train has {y_train.shape[0]} labels.")

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=len(label_maps))
y_val = to_categorical(y_val, num_classes=len(label_maps))
y_test = to_categorical(y_test, num_classes=len(label_maps))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# Pretrained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_maps), activation='softmax')  # Output layer
])

print(model.summary())

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint_dir = r'E:\OCR_Calculator\checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_model_transfer.keras'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint_callback, reduce_lr]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Predictions and evaluation metrics
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# F1 Score, Recall, and Classification Report
f1 = f1_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print(f'F1 Score: {f1:.4f}')
print(f'Recall: {recall:.4f}')
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
