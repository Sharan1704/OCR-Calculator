import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from contour_detection import process_images


# Step 1: Define Constants
IMAGE_SIZE = (96, 96)  # Resize images to 96x96
DATASET_PATH = r'E:\OCR_Calculator\Dataset\img_dataset'  # Path to your dataset images
PREPROCESSED_PATH = r'E:\OCR_Calculator\Dataset\preprocessed_img'  # Path to save preprocessed images (if needed)

# Step 2: Load Images and Labels
def load_images_and_labels(dataset_path, target_size=(96, 96)):
    images = []
    labels = []
    label_map = {}

    # Iterate over each class subdirectory
    for label_index, label in enumerate(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):  # Ensure it's a directory
            label_map[label_index] = label  # Map index to label name
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize image to the target size
                    img_resized = cv2.resize(img, target_size)
                    images.append(img_resized)
                    labels.append(label_index)  # Use the index as the label
                else:
                    print(f"Warning: Unable to load image {img_path}")

    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return np.array(images), np.array(labels), label_map

# Step 3: Encode Labels
def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder.classes_  # classes_ will give you the label mapping

# Step 4: Split Dataset
def split_data(images, labels):
    if len(images) == 0:
        raise ValueError("No images to split.")
    if len(labels) == 0:
        raise ValueError("No labels to split.")
    if len(images) != len(labels):
        raise ValueError("Mismatch between number of images and labels.")
    
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Optional: Save Preprocessed Data
def save_preprocessed_data(X, y, save_path):
    np.save(os.path.join(save_path, 'X.npy'), X)
    np.save(os.path.join(save_path, 'y.npy'), y)

# Step 5: Run the Preprocessing Function
if __name__ == '__main__':
    # Load images and labels
    images, labels, label_map = load_images_and_labels(DATASET_PATH, target_size=IMAGE_SIZE)
    print("Images shape: ", images.shape, "Labels shape: ", labels.shape, "Label Map: ", label_map)

    # Encode labels
    encoded_labels, class_names = encode_labels(labels)

    # Split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, encoded_labels)

    # Reshape images for CNN input (adding a channel dimension)
    X_train = X_train.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    X_val = X_val.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    X_test = X_test.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    # Optionally save the preprocessed data
    os.makedirs(PREPROCESSED_PATH, exist_ok=True)  # Create directory if it doesn't exist
    save_preprocessed_data(X_train, y_train, PREPROCESSED_PATH)
    save_preprocessed_data(X_val, y_val, PREPROCESSED_PATH)
    save_preprocessed_data(X_test, y_test, PREPROCESSED_PATH)

    print(f'Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}')
