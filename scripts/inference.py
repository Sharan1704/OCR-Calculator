# inference.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from contour_recognition import process_images  # Import from contour_recognition.py

# Paths
MODEL_PATH = r'E:\OCR_Calculator\checkpoints\best_model.keras'
IMAGE_PATH = r'E:\OCR_Calculator\Dataset\img_dataset\(\(_22.jpg'  # Path to the test image

# Step 1: Load the OCR Model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

def preprocess_and_segment(image_path):
    """
    Load an image, process it using contour detection,
    and return a list of segmented character images.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return []

    # Process the image to segment characters using contour recognition functions
    segmented_characters = process_images([image])  # Use the function from contour_recognition.py
    return segmented_characters

def predict_characters(segmented_images):
    """
    Take segmented images of characters, resize them,
    and return the predicted character for each image.
    """
    predictions = []
    for char_img in segmented_images:
        # Expand dimensions to match model input (1, 28, 28, 1)
        char_img = np.expand_dims(char_img, axis=-1)  # Add channel dimension
        char_img = np.expand_dims(char_img, axis=0)   # Add batch dimension
        prediction = model.predict(char_img)
        
        # Get the predicted class label (character)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predictions.append(predicted_label)
    return predictions

if __name__ == '__main__':
    # Step 2: Preprocess the image and get segmented characters
    segmented_images = preprocess_and_segment(IMAGE_PATH)
    
    if not segmented_images:
        print("No characters were segmented. Check the image and contour settings.")
    else:
        print("Character segmentation complete.")

    # Step 3: Predict characters
    predictions = predict_characters(segmented_images)
    
    # Display predictions
    print("Predicted characters:", predictions)
