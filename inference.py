import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from contour_detection import process_images
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (28, 28)
MODEL_PATH = 'ocr_model.keras'  # Path to your trained model
LABELS_FILE = 'preprocessed_data.pkl'  # Path to preprocessed data (for label mapping)

# Load the trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Load label mapping from preprocessed_data.pkl
def load_label_mapping(label_file):
    with open(label_file, 'rb') as f:
        data = pickle.load(f)
    label_map = {v: k for k, v in data['label_map'].items()}  # Reverse to map indices to characters
    print(f"Loaded label map: {label_map}")
    return label_map

# Segment and predict with confidence-based filtering and visualization
def segment_and_predict(image_path, model, label_map, visualize=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return []

    segmented_chars = process_images([image], visualize)
    predictions = []

    for i, char in enumerate(segmented_chars):
        char = np.expand_dims(char, axis=-1)  # Add channel dimension
        char = np.expand_dims(char, axis=0)  # Add batch dimension

        pred_prob = model.predict(char)
        confidence = np.max(pred_prob)
        predicted_class = np.argmax(pred_prob, axis=1)[0]
        predicted_char = label_map.get(predicted_class, None)

        print(f"Character {i}: Predicted class={predicted_class}, Char={predicted_char}, Confidence={confidence}")

        # Confidence-based adjustments
        if predicted_class in [label_map.get('(', -1), label_map.get(')', -1)] and confidence > 0.9 and predicted_char == '0':
            # Correct high-confidence misclassifications of parentheses as '0'
            predicted_char = '(' if predicted_class == label_map.get('(', -1) else ')'
        elif confidence < 0.5:
            # Low-confidence predictions are flagged
            predicted_char = '?'

        predictions.append(predicted_char)

        # Visualize each segmented character
        if visualize:
            plt.imshow(char.squeeze(), cmap='gray')
            plt.title(f"Segmented Character {i}")
            plt.axis('off')
            plt.show()

    return predictions

# Reconstruct the mathematical expression from predictions
def reconstruct_expression(predictions):
    operator_map = {
        'plus': '+', 'minus': '-', 'mul': '*', 'div': '/',
        '(': '(', ')': ')'
    }
    expression = []
    for char in predictions:
        if char in operator_map:
            expression.append(operator_map[char])
        elif char.isdigit() or char == '?':
            expression.append(char)
    return ''.join(expression)

# Evaluate the reconstructed expression
def evaluate_expression(expression):
    try:
        result = eval(expression)  # Evaluate the expression
        return result
    except Exception as e:
        print(f"Error evaluating expression: {e}")
        return None

# Main function to process the input image and predict the result
def main(image_path):
    model = load_trained_model(MODEL_PATH)
    print("Loading label mappings...")
    label_map = load_label_mapping(LABELS_FILE)

    # Segment and predict
    print("Segmenting and predicting characters...")
    predictions = segment_and_predict(image_path, model, label_map, visualize=True)
    print(f"Predicted characters: {predictions}")

    expression = reconstruct_expression(predictions)
    print(f"Reconstructed expression: {expression}")
    result = evaluate_expression(expression)
    print(f"Result: {result}")

    if result is not None:
        with open('results.txt', 'w') as f:
            f.write(f"Expression: {expression}\n")
            f.write(f"Result: {result}")
        print("Results saved to results.txt")

# Run the main function
if __name__ == '__main__':
    image_path = r'C:\Users\user\Pictures\Screenshots\Screenshot 2024-12-03 111936.png'  # Replace with your test image
    main(image_path)
