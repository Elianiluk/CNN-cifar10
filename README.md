
# **Convolutional Neural Network for Image Classification**

## **Overview**
This program defines and trains a Convolutional Neural Network (CNN) for multi-class image classification using the PyTorch framework. The CNN architecture is designed to classify images into one of 10 classes from the CIFAR-10 dataset. 

The program demonstrates key machine learning tasks, including:
- Data preprocessing and augmentation
- Model definition and training
- Evaluation of model performance
- Visualization of predictions and results

## **Key Features**
1. **Custom CNN Architecture:**
   - 3 Convolutional Layers followed by MaxPooling layers.
   - 3 Fully Connected Layers with ReLU activation functions.
   - Dropout layer to prevent overfitting.

2. **Dataset:**
   - CIFAR-10 dataset, which contains 60,000 images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

3. **Loss and Optimization:**
   - **Loss Function**: CrossEntropyLoss
   - **Optimizer**: Adam (Adaptive Moment Estimation)

4. **Training and Validation:**
   - Data split into training, validation, and test sets.
   - Model saves the weights with the lowest validation loss during training.

5. **GPU Support:**
   - Automatically detects GPU availability for faster computations.

6. **Visualization:**
   - Plots training images with labels.
   - Visualizes model predictions with true labels.

---

## **Usage**

### **Requirements**
Install the required libraries before running the code:
```bash
pip install torch torchvision matplotlib numpy
```

### **File Structure**
- **Code Implementation**: The provided code defines all steps, from data preprocessing to evaluation and visualization.

### **Run the Program**
1. Clone the repository or download the script.
2. Run the program:
   ```bash
   python cnn_image_classification.py
   ```

---

## **Dataset**
The CIFAR-10 dataset is used in this project:
- **Training Data**: 80% of the dataset (48,000 images)
- **Validation Data**: 20% of the training set (12,000 images)
- **Test Data**: 10,000 images

The dataset is automatically downloaded using the `torchvision.datasets` module.

---

## **Program Workflow**
1. **Data Loading and Preprocessing:**
   - Augments the training dataset with random horizontal flips and rotations.
   - Normalizes images to have zero mean and unit variance.

2. **Model Definition:**
   - Implements a custom CNN architecture with ReLU activation, dropout, and softmax layers.

3. **Training:**
   - Trains the model for a specified number of epochs (default: 3).
   - Tracks and displays training and validation loss.
   - Saves the model with the best validation loss.

4. **Testing:**
   - Evaluates the model's performance on the test set.
   - Reports accuracy for each class and overall accuracy.

5. **Visualization:**
   - Displays sample images from the dataset with their predicted and actual labels.

---

## **Model Architecture**
| Layer Type          | Parameters                         |
|---------------------|------------------------------------|
| Convolution Layer 1 | Input: 3 channels, Output: 16      |
| Convolution Layer 2 | Input: 16 channels, Output: 32     |
| Convolution Layer 3 | Input: 32 channels, Output: 64     |
| Max Pooling         | Kernel Size: 2x2, Stride: 2       |
| Fully Connected 1   | Input: 64x4x4, Output: 256         |
| Fully Connected 2   | Input: 256, Output: 128            |
| Fully Connected 3   | Input: 128, Output: 10 (classes)   |
| Dropout             | Probability: 0.2                  |
| Activation Function | ReLU                               |
| Output Function     | LogSoftmax (for classification)    |

---

## **Results**
1. **Training and Validation Loss:**
   The program tracks and prints loss values during training and validation.

2. **Test Accuracy:**
   - Per-class accuracy for all 10 classes.
   - Overall accuracy percentage.

3. **Visual Predictions:**
   - Sample test images with predicted and actual labels are displayed.
   - Correct predictions are marked in green, incorrect in red.

---

## **Customization**
1. **Adjust Training Parameters:**
   - Modify the number of epochs, batch size, or learning rate as needed.

   Example:
   ```python
   n_epochs = 10
   batch_size = 32
   learning_rate = 0.0005
   ```

2. **Modify Model Architecture:**
   - Add or remove layers to experiment with performance.

3. **Dataset:**
   - Replace CIFAR-10 with a custom dataset by modifying the data loaders.

---

## **Contact**
**Author**: Elian Iluk  
**Email**: elian10119@gmail.com  

Feel free to reach out for any questions or feedback regarding the program.

