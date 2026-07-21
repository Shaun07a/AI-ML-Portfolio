# CIFAR-10 Image Classification using CNN

## Project Overview

This project implements an image classification model using a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The model is developed using TensorFlow and Keras and is trained on the official CIFAR-10 dataset containing 60,000 RGB images.

The CNN is capable of classifying images into one of ten different object categories with a test accuracy of **83.53%**.

---

## Dataset

The project uses the official CIFAR-10 dataset provided by TensorFlow.

- Training Images: 50,000
- Testing Images: 10,000
- Image Size: 32 × 32 pixels
- Channels: RGB (3)

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset Source:
https://www.cs.toronto.edu/~kriz/cifar.html

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Model Architecture

The CNN consists of multiple convolutional blocks followed by fully connected layers.

Architecture:

- Data Augmentation
- Conv2D (32 Filters)
- Batch Normalization
- Conv2D (32 Filters)
- MaxPooling2D
- Dropout

- Conv2D (64 Filters)
- Batch Normalization
- Conv2D (64 Filters)
- MaxPooling2D
- Dropout

- Conv2D (128 Filters)
- Batch Normalization
- Conv2D (128 Filters)
- MaxPooling2D
- Dropout

- Flatten Layer

- Dense (256 Neurons)
- Batch Normalization
- Dropout

- Output Layer (10 Classes, Softmax)

---

## Model Training

Loss Function:

- Sparse Categorical Crossentropy

Optimizer:

- Adam

Callbacks Used:

- Early Stopping
- ReduceLROnPlateau

Regularization Techniques:

- Batch Normalization
- Dropout
- Data Augmentation

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **83.53%** |
| Classes | 10 |
| Training Images | 50,000 |
| Testing Images | 10,000 |

---

## Outputs

The project generates:

- Training Accuracy Graph
- Validation Accuracy Graph
- Training Loss Graph
- Validation Loss Graph
- Confusion Matrix
- Classification Report
- Sample Predictions
- Saved Trained Model (.keras)

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/CIFAR10-Image-Classification-CNN.git
```

Go inside the project

```bash
cd CIFAR10-Image-Classification-CNN
```

Install dependencies

```bash
pip install tensorflow matplotlib numpy seaborn scikit-learn
```

Run the project

```bash
python cifar-10.py
```

---

## Project Structure

```
CIFAR10-Image-Classification-CNN
│
├── cifar-10.py
├── CIFAR10_CNN_Model.keras
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Implement Transfer Learning using EfficientNetB0
- Experiment with ResNet architecture
- Hyperparameter tuning
- Improve accuracy beyond 90%
- Deploy the model using Flask or Streamlit

---

## Author

Shaun Joseph
