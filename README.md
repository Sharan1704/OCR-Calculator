# OCR Calculator - Handwritten Mathematical Expression Recognition

**Overview**

The OCR Calculator is a machine learning-based project designed to recognize and evaluate handwritten mathematical expressions. It processes handwritten digits and mathematical symbols, classifies them using a deep learning model, and evaluates the resulting expressions.

**Features**

Handwritten digit and operator recognition (0-9, +, -, *, /, (, )).
Image preprocessing and data augmentation for improved accuracy.
Deep learning-based classification using convolutional neural networks (CNNs).
Expression evaluation using Python.

**Dataset**

The dataset consists of images of handwritten digits (0-9) and mathematical symbols (+, -, *, /, (, )). It was downloaded from Kaggle and preprocessed before training the model.

**Preprocessing Steps**

Images were converted to grayscale and resized to 28x28 pixels.
Normalization was applied by scaling pixel values between 0 and 1.
Data augmentation was used to increase samples of underrepresented classes.

**Model Training**

Model: Convolutional Neural Network (CNN)
Input Size: 28x28x1 (grayscale images)
Layers:
Convolutional layers with ReLU activation
Max pooling layers
Fully connected layers
Softmax output layer
Loss Function: Categorical Cross-Entropy
Optimizer: Adam
Evaluation Metrics: Accuracy, Precision, Recall, F1-score

Performance
The trained model achieved the following results:
Test Accuracy: 96.97%
F1 Score: 96.97%
Recall: 96.97%

**Prerequisites**

Ensure you have the following installed:
Python 3.x
TensorFlow / Keras
OpenCV
NumPy
Scikit-learn
Kaggle API (for dataset download)
