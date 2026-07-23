# 🧠 Alzheimer's Detection & Classification using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/tensorflow-2.12+-orange.svg)](https://tensorflow.org)
[![Flask 2.3+](https://img.shields.io/badge/flask-2.3+-g.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end medical computer vision project that leverages Deep Convolutional Neural Networks (CNNs) to detect and classify Alzheimer's Disease stages from brain Magnetic Resonance Imaging (MRI) scans. The project features a comparative training pipeline supporting custom architectures and advanced transfer learning models, complete with a modern, responsive web application for real-time inference and analysis.

---

## 📌 Table of Contents

1. [🧠 Project Overview](#-project-overview)
2. [🔬 Methodology & Architecture](#-methodology--architecture)
3. [📊 Dataset & Data Distribution](#-dataset--data-distribution)
4. [🤖 Model Training Pipeline](#-model-training-pipeline)
5. [📈 Evaluation & Performance](#-evaluation--performance)
6. [🌐 Web Application Interface](#-web-application-interface)
7. [⚙️ Installation & Usage](#-installation--usage)
8. [🛠️ Configuration Options](#-configuration-options)

---

## 🧠 Project Overview

Alzheimer's Disease is a progressive neurodegenerative disorder that accounts for 60-80% of dementia cases. Early and accurate diagnosis of Alzheimer's stages is critical for effective clinical intervention and patient management. 

This project aims to automate this classification using MRI brain scans. It provides:
- **Comprehensive Training Pipeline**: Automatic training, evaluation, and selection of the best-performing neural network architecture.
- **Support for Multiple Architectures**: Custom Deep CNN, MobileNetV2, and EfficientNetB0.
- **Robust Image Preprocessing**: Includes class balancing via oversampling and custom class-weighted loss optimization to handle severe dataset imbalances.
- **Interactive Web Platform**: A sleek, user-friendly interface for clinicians or researchers to upload MRI scans, preview the input, visualize the results, and track prediction history.

---

## 🔬 Methodology & Architecture

The project utilizes a structured pipeline starting from dataset ingestion, image preprocessing, class imbalance handling, model training, evaluation, and deployment.

### 📐 System Methodology Diagram

Below is the structured methodology representing the workflow from data preprocessing to Flask web app deployment.

![Methodology Diagram](Methdology%20Diagram.png)

---

## 📊 Dataset & Data Distribution

The dataset contains brain MRI scans classified into four distinct stages of Alzheimer's disease:
1. **Non Demented**: Healthy control brains.
2. **Very Mild Demented**: Mild cognitive impairment/early stage Alzheimer's.
3. **Mild Demented**: Moderate cognitive decline.
4. **Moderate Demented**: Advanced cognitive impairment and severe symptoms.

### 📈 Dataset Distribution & Sample Views
To understand the dataset structure, we analyze the distribution of classes and visualize samples below:

| Class Distribution Chart | Sample MRI Images Preview |
| :---: | :---: |
| ![Dataset Classes Pi Chart](Dataset%20Classes%20Pi%20chart.png) | ![Dataset View](dataset%20view.png) |

> [!NOTE]
> Class imbalance is a significant challenge in medical datasets (specifically with a low number of samples in `Moderate_Demented`). The pipeline handles this through oversampling minority classes to a configurable threshold and adjusting model class weights during loss computation.

---

## 🤖 Model Training Pipeline

The pipeline compares three different deep learning models:

1. **Custom CNN**: A tailored Convolutional Neural Network built with multiple layers of convolutions, batch normalization, max-pooling, and dropout to prevent overfitting.
2. **MobileNetV2**: A lightweight architecture utilizing depthwise separable convolutions, ideal for fast inference and deployment on resource-constrained environments.
3. **EfficientNetB0**: A state-of-the-art model that uses compound scaling to achieve high accuracy with minimal parameters.

To start training, configure the hyperparameters in `config.py` and run:
```bash
python train_models.py
```

---

## 📈 Evaluation & Performance

During evaluation, test dataset metrics are calculated including Accuracy, Precision, Recall, Macro F1-score, and Weighted F1-score. The model with the highest **Validation Macro F1-score** is selected as the production candidate.

### 📊 Model Performance Analytics

| Confusion Matrix | Actual vs Predicted Plots | Model Accuracy / Loss Performance |
| :---: | :---: | :---: |
| ![CNN Custom Confusion Matrix](CNN%20Custome%20Confusion%20matrix%20.png) | ![CNN Actual vs Predicted](CNN%20model%20results%20sample%20Actual%20vs%20predicted%20plot%20.png) | ![Model Performance](Model%20Performce%20and%20Segmentation%20of%20alzamier%20.png) |

---

## 🌐 Web Application Interface

A Flask-based web application provides an interactive portal for uploading MRI files, displaying classification confidence percentages, visualizing probability distributions, and keeping track of upload history.

### 🖼️ UI Walkthrough & Features

#### 1. Landing & Welcome Page
The landing page introduces the system, its capabilities, and allows navigation to the prediction portal.
![App Landing Page](APP%20landing%20page%20.jpg)

---

#### 2. Scan Upload & Prediction Interface
Users can upload an MRI scan with a live preview before submitting for classification.
![Web App Input Page](Web%20App%20Input%20page.jpg)

---

#### 3. Real-Time Diagnostics & Probabilities
Detailed diagnosis reports show the predicted category, prediction confidence, and comparative probabilities for all four states.
![Final Prediction Page](Final%20Prediction%20page%20with%20view%20.jpg)

---

#### 4. Prediction Session History
An ongoing history panel allows clinicians to compare different MRI uploads within the current session.
![Prediction History](Prediction%20history%20.jpg)

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8 or higher
- GPU support recommended (CUDA-enabled graphics card) for faster training

### Step 1: Clone the Repository and Install Dependencies
```bash
git clone <repository-url>
cd code
pip install -r requirements.txt
```

### Step 2: Prepare the Dataset
Place your MRI brain scans under the `dataset/` directory inside class-specific subfolders:
```text
code/dataset/
├── Mild_Demented/
├── Moderate_Demented/
├── Non_Demented/
└── Very_Mild_Demented/
```

### Step 3: Run the Training Pipeline
```bash
python train_models.py
```
This script will:
- Load and preprocess the images (resized to 224x224).
- Train all three model architectures (Custom CNN, MobileNetV2, EfficientNetB0).
- Save the best-performing model to `models/best_alzheimer_model.keras`.

### Step 4: Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)** to access the web application.

---

## 🛠️ Configuration Options

You can tune training parameters, dataset directories, and image properties directly in [config.py](file:///c:/Users/ABC/Downloads/AI%20removal%20projects/Alzmier%20Detection%20using%20AI/code/config.py):

* **`IMG_SIZE`**: Dimensions to resize MRI images (`(224, 224)` default).
* **`BATCH_SIZE`**: Training batch size (default is `32`).
* **`VAL_SPLIT` & `TEST_SPLIT`**: Ratios for validation and testing splits.
* **`EPOCHS`**: Maximum number of training epochs (default is `50` with EarlyStopping patience of `10`).
* **`OVERSAMPLE_MIN_PER_CLASS`**: Minimum class sample size for balancing (default is `800`).
* **`CLASS_WEIGHT_MAX`**: Maximum scaling factor limit for training class weights.

---

*Disclaimer: This system is a deep learning research tool and should not be used as a standalone clinical diagnostic device without professional verification.*
