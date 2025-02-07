import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMAGE_SIZE = (28, 28)
DATASET_DIR = r'OCR Dataset/img_dataset'
OUTPUT_FILE = "preprocessed_data.pkl"

# Label mapping for classes
label_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5 ,'6': 6, '7': 7,
    '8': 8, '9': 9, 'plus': 10, 'minus': 11, 'mul': 12, 'div': 13, '(': 14, ')': 15
}

def load_images_and_labels(dataset_dir, label_map):
    images = []
    labels = []

    for label_name, label_idx in label_map.items():
        class_dir = os.path.join(dataset_dir, str(label_name))
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist!")
            continue

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_name.endswith(".png" or ".jpg"):
                try:
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, IMAGE_SIZE)
                    image = image / 255.0
                    images.append(image)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int32")

    return images, labels

# Define ImageDataGenerator for augmentation
augment_datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

def augment_data(X_train, y_train, target_class, target_count):
    class_indices = np.where(y_train == target_class)[0]
    augmented_images = []
    augmented_labels = []

    for idx in class_indices:
        image = X_train[idx].reshape((1, *IMAGE_SIZE, 1))
        augment_iter = augment_datagen.flow(image, batch_size=1)
        num_augmentations_per_image = target_count // len(class_indices)
        for _ in range(num_augmentations_per_image):
            augmented_image = next(augment_iter)[0]
            augmented_images.append(augmented_image)
            augmented_labels.append(target_class)

    return np.array(augmented_images, dtype="float32"), np.array(augmented_labels, dtype="int32")

# Load images and labels
print("Loading images and labels...")
images, labels = load_images_and_labels(DATASET_DIR, label_map)
print(f"Loaded {len(images)} images and {len(labels)} labels.")

# Reshape images for CNN input
images = images.reshape(images.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Augment training data for all sensitive classes
sensitive_classes = ['1', 'minus', 'mul', '(', ')']
target_count = 3000  # Desired number of images

for sensitive_class in sensitive_classes:
    target_class = label_map[sensitive_class]
    X_augmented, y_augmented = augment_data(X_train, y_train, target_class, target_count)
    X_augmented = X_augmented.reshape(X_augmented.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)  # Ensure correct shape
    X_train = np.concatenate((X_train, X_augmented))
    y_train = np.concatenate((y_train, y_augmented))

# Save preprocessed data
print("Saving preprocessed data...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "label_map": label_map
    }, f)
print(f"Preprocessed data saved to {OUTPUT_FILE}")
