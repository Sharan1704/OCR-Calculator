import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect contours in an image
def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to sort contours based on position
def sort_contours(contours):
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes.sort(key=lambda box: box[0])  # Sort by x-coordinate first
    return bounding_boxes

# Function to extract individual characters from sorted contours
def extract_characters(image, contours):
    characters = []
    sorted_contours = sort_contours(contours)
    for x, y, w, h in sorted_contours:
        if w > 5 and h > 5:  # Filter out small contours (noise)
            char_img = image[y:y + h, x:x + w]
            char_resized = cv2.resize(char_img, (28, 28))  # Resize to match model input size
            char_normalized = char_resized.astype('float32') / 255.0  # Normalize to [0, 1]
            characters.append(char_normalized)
    return characters

# Function to draw bounding boxes on the original image
def draw_bounding_boxes(image, contours):
    image_copy = image.copy()  # Create a copy to avoid modifying the original
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Filter out small contours (noise)
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green color, 1px thickness
    return image_copy

# Process image to segment characters (contour detection + character extraction)
def process_images(images, visualize=False):
    all_segmented_chars = []
    for image in images:
        contours = detect_contours(image)  # Step 1: Detect contours
        if visualize:
            # Draw and display bounding boxes
            boxed_image = draw_bounding_boxes(image, contours)
            plt.imshow(boxed_image, cmap='gray')  # Display with matplotlib
            plt.title("Bounding Boxes")
            plt.axis('off')
            plt.show()

            
        segmented_chars = extract_characters(image, contours)  # Step 2: Extract characters
        all_segmented_chars.extend(segmented_chars)  # Collect for further processing
    return all_segmented_chars

if __name__ == "__main__":
    print("This code file is not intended to be executed directly")
