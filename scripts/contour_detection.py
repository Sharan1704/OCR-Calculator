import os
import cv2
import numpy as np

# Constants
IMAGE_SIZE = (96, 96)  # Resize all images to 96x96
DATASET_PATH = r'E:\OCR_Calculator\Dataset\img_dataset'
OUTPUT_PATH = r'E:\OCR_Calculator\Dataset\segmented_characters'

def detect_contours(image):
    if len(image.shape) == 3:  # Convert to grayscale if not already
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours, min_area=100, max_area=1000):
    return [c for c in contours if min_area < cv2.contourArea(c) < max_area]

def extract_roi(image, contours):
    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 5:  # Ignore very small bounding boxes
            continue
        roi = image[y:y+h, x:x+w]
        rois.append(roi)
    return rois

def process_images(images):
    segmented_images = []
    for i, image in enumerate(images):
        contours = detect_contours(image)
        filtered_contours = filter_contours(contours)
        
        if not filtered_contours:
            print(f"Warning: No valid contours detected in image {i}")
            continue

        print(f"Image {i}: {len(filtered_contours)} valid contours detected.")
        for j, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            char_img = image[y:y+h, x:x+w]  # Crop the character
            
            char_resized = cv2.resize(char_img, IMAGE_SIZE)  # Resize to (96, 96)
            char_normalized = char_resized.astype('float32') / 255.0  # Normalize to [0, 1]
            
            # Save the processed character for inspection
            save_path = os.path.join(OUTPUT_PATH, f'char_{i}_{j}.png')
            cv2.imwrite(save_path, char_resized)
            
            segmented_images.append(char_normalized)

    print(f"Total segmented images processed: {len(segmented_images)}")
    return segmented_images

if __name__ == '__main__':
    print("This module is not intended to be run directly.")
