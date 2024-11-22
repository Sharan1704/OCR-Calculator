import os
import cv2
import numpy as np

# Constants
IMAGE_SIZE = (28, 28)  # Size to which each character is resized
DATASET_PATH = r'E:\OCR_Calculator\Dataset\img_dataset'  # Dataset path for input images
SEGMENTED_PATH = r'E:\OCR_Calculator\Dataset\segmented_characters'  # Path to save segmented characters

def detect_contours(image):
    """
    Detect contours in the given image.
    Converts the image to grayscale (if not already), applies Gaussian blur,
    and uses binary thresholding for contour detection.
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply Gaussian blur and thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours, min_area=100, max_area=1000):
    """
    Filters contours by area, keeping only those within the specified range.
    This helps to exclude noise or very small contours that are not relevant.
    """
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    return filtered_contours

def extract_roi(image, contours):
    """
    Extracts regions of interest (ROIs) from an image based on contours.
    Skips any ROIs that are too small (to avoid noise).
    """
    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 5 and h >= 5:  # Skip very small ROIs
            roi = image[y:y+h, x:x+w]
            rois.append(roi)
    return rois

def segment_characters(image, contours):
    """
    Segments individual characters from the given image based on contours.
    Each character is resized and normalized for model input.
    """
    segmented_chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Ensure the contour is large enough
            char_img = image[y:y+h, x:x+w]
            char_resized = cv2.resize(char_img, IMAGE_SIZE)
            char_normalized = char_resized.astype('float32') / 255.0
            segmented_chars.append(char_normalized)
    return segmented_chars

def save_segmented_characters(segmented_chars, img_index):
    """
    Saves segmented characters to disk with unique filenames.
    Each character image is saved as an individual file in the segmented directory.
    """
    if not os.path.exists(SEGMENTED_PATH):
        os.makedirs(SEGMENTED_PATH)
    
    for j, char in enumerate(segmented_chars):
        # Saving as an 8-bit grayscale image
        char_path = os.path.join(SEGMENTED_PATH, f'char_{img_index}_{j}.png')
        cv2.imwrite(char_path, (char * 255).astype(np.uint8))

def process_images(images):
    """
    Processes each image in the input list by detecting contours, filtering them,
    and extracting individual characters. Saves each character as a separate image.
    Returns a list of all segmented characters as normalized arrays.
    """
    all_segmented_chars = []
    for i, image in enumerate(images):
        contours = detect_contours(image)  # Step 1: Detect contours
        filtered_contours = filter_contours(contours)  # Step 2: Filter contours
        segmented_chars = segment_characters(image, filtered_contours)  # Step 3: Segment characters
        save_segmented_characters(segmented_chars, i)  # Step 4: Save segmented characters
        all_segmented_chars.extend(segmented_chars)  # Collect for further processing
    return all_segmented_chars

if __name__ == '__main__':
    print("This module is not intended to be run directly.")
