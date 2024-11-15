# Convolutional Neural Network for Image Classification
## Overview
This program defines and trains a Convolutional Neural Network (CNN) for multi-class image classification using the PyTorch framework. The CNN architecture is designed to classify images into one of 10 classes from the CIFAR-10 dataset.

The program demonstrates key machine learning tasks, including:

Data preprocessing and augmentation
Model definition and training
Evaluation of model performance
Visualization of predictions and results
Key Features
## Custom CNN Architecture:

3 Convolutional Layers followed by MaxPooling layers.
3 Fully Connected Layers with ReLU activation functions.
Dropout layer to prevent overfitting.
Dataset:

CIFAR-10 dataset, which contains 60,000 images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
Loss and Optimization:

Loss Function: CrossEntropyLoss
Optimizer: Adam (Adaptive Moment Estimation)
Training and Validation:

Data split into training, validation, and test sets.
Model saves the weights with the lowest validation loss during training.
GPU Support:

Automatically detects GPU availability for faster computations.
### Visualization:

Plots training images with labels.
Visualizes model predictions with true labels.

## Dataset
The CIFAR-10 dataset is used in this project:

Training Data: 80% of the dataset (48,000 images)
Validation Data: 20% of the training set (12,000 images)
Test Data: 10,000 images
The dataset is automatically downloaded using the torchvision.datasets module.

## Program Workflow
Data Loading and Preprocessing:

Augments the training dataset with random horizontal flips and rotations.
Normalizes images to have zero mean and unit variance.
Model Definition:

Implements a custom CNN architecture with ReLU activation, dropout, and softmax layers.
Training:

Trains the model for a specified number of epochs (default: 3).
Tracks and displays training and validation loss.
Saves the model with the best validation loss.
Testing:

Evaluates the model's performance on the test set.
Reports accuracy for each class and overall accuracy.
Visualization:

Displays sample images from the dataset with their predicted and actual labels.
## Model Architecture
- Convolution Layer 1	Input: 3 channels, Output: 16
- Convolution Layer 2	Input: 16 channels, Output: 32
- Convolution Layer 3	Input: 32 channels, Output: 64
- Max Pooling	Kernel Size: 2x2, Stride: 2
- Fully Connected 1	Input: 64x4x4, Output: 256
- Fully Connected 2	Input: 256, Output: 128
- Fully Connected 3	Input: 128, Output: 10 (classes)
- Dropout	Probability: 0.2
- Activation Function	ReLU
- Output Function	LogSoftmax (for classification)
## Results
Training and Validation Loss: 
- The program tracks and prints loss values during training and validation.
  
Test Accuracy:
- Per-class accuracy for all 10 classes.
- Overall accuracy percentage.
  
Visual Predictions:
- Sample test images with predicted and actual labels are displayed.
- Correct predictions are marked in green, incorrect in red.
  

## Contact
Author: Elian Iluk
Email: elian10119@gmail.com

Feel free to reach out for any questions or feedback regarding the program.

