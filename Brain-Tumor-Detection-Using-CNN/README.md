# Brain Tumor Detection Using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The model is trained on MRI images using image preprocessing and data augmentation techniques to improve generalization.

---

## Features

- MRI image classification
- TensorFlow & Keras implementation
- CNN architecture
- Image preprocessing
- Data augmentation
- Training and validation
- Accuracy and loss visualization
- Model saving for future predictions

---

## Dataset

Dataset used:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset contains two folders:

```
Training/
Testing/
```

Each folder contains four classes:

```
glioma
meningioma
notumor
pituitary
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow

---

## Project Structure

```
Brain-Tumor-Detection-Using-CNN/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── brain_tumor_detection.py
├── brain_tumor_model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

---

## CNN Architecture

The CNN consists of:

- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Conv2D (128 Filters)
- MaxPooling2D
- Flatten Layer
- Dense (256 Neurons)
- Dropout
- Dense (128 Neurons)
- Dropout
- Output Layer (Softmax)

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Brain-Tumor-Detection-Using-CNN.git
```

Move into the project

```bash
cd Brain-Tumor-Detection-Using-CNN
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python brain_tumor_detection.py
```

---

## Results

The trained model typically achieves:

- Training Accuracy: 90–97%
- Validation Accuracy: 88–96%

Results may vary depending on the TensorFlow version, hardware, and number of epochs.

---

## Output

The project generates:

- Trained CNN model (`brain_tumor_model.keras`)
- Accuracy graph
- Loss graph
- Test accuracy
- Test loss

---

## Future Improvements

- Transfer Learning using ResNet50
- MobileNetV2 implementation
- MRI image segmentation
- Web application using Flask
- Real-time prediction interface

---

## Author

Shaun Joseph
