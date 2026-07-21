# Animal Face Detection Using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) to classify animal face images into one of 37 different cat and dog breeds using the Oxford-IIIT Pet Dataset. The model is built using TensorFlow and Keras and demonstrates the complete workflow of an image classification project including dataset loading, preprocessing, model training, evaluation, visualization, and prediction.

---

## Dataset

* **Dataset:** Oxford-IIIT Pet Dataset
* **Total Images:** Approximately 7,300
* **Classes:** 37 cat and dog breeds
* **Framework Used:** TensorFlow Datasets (TFDS)

---

## Features

* Automatic dataset download
* Image preprocessing and normalization
* CNN model built using TensorFlow/Keras
* Training and validation
* Accuracy and loss visualization
* Model saving for future predictions
* Prediction on custom images

---

## Model Architecture

The CNN consists of:

* Conv2D (32 Filters)
* MaxPooling2D
* Conv2D (64 Filters)
* MaxPooling2D
* Conv2D (128 Filters)
* MaxPooling2D
* Flatten Layer
* Dense Layer (256 Neurons)
* Dropout Layer
* Output Layer (37 Classes)

Total Parameters: **6,525,541**

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## Project Structure

```
Animal-Face-Detection/

│── Face_Detection.py
│── OxfordPetClassifier.keras
│── README.md
│── requirements.txt
│── accuracy.png
│── loss.png
```

---

## Training Results

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 78.53% |
| Validation Accuracy | 12.91% |
| Test Accuracy       | 12.91% |

---

## Observations

The model achieved high training accuracy but relatively low validation and test accuracy, indicating overfitting. This is expected for a basic CNN trained without advanced regularization or transfer learning.

---

## Future Improvements

* Implement Transfer Learning (MobileNetV2 or EfficientNet)
* Apply Data Augmentation
* Increase training epochs with Early Stopping
* Hyperparameter tuning
* Fine-tune pretrained models
* Improve generalization using Batch Normalization



## Author

Shaun Joseph
