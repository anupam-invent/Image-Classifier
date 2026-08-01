# 🐱 Image Classifier

A deep learning web application built with **TensorFlow** and **Streamlit** that classifies uploaded images as either **Cat** or **Dog**.

## 🚀 Features

- Upload JPG, JPEG, or PNG images
- AI-powered image classification
- Confidence score
- Image preview
- Responsive Streamlit interface

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pillow

## Model Architecture

The classifier uses a custom Convolutional Neural Network (CNN) built from scratch with TensorFlow/Keras using the Sequential API.

Architecture:
- Input Layer (256 × 256 × 3)
- Data Augmentation
  - Random Flip
  - Random Rotation
  - Random Zoom
  - Random Contrast
  - Random Translation
- Conv2D (32 filters) + Batch Normalization + MaxPooling
- Conv2D (64 filters) + Batch Normalization + MaxPooling
- Conv2D (128 filters) + Batch Normalization + MaxPooling
- Flatten
- Dense (128) + ReLU + Dropout
- Dense (64) + ReLU + Dropout
- Output Layer (Sigmoid)

## 📂 Project Structure

```
Image-Classifier/
│── app.py
│── best_model.keras
│── requirements.txt
│── README.md
│── .gitignore
```