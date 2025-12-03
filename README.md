# Final-Project
Breast Cancer (Predicting invasive ductal carcinoma in tissue slices)
# 🩺 Breast Cancer Histopathology Image Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange)

**Deep Learning for Breast Cancer Histopathology Image Classification**

</div>

## 📋 Overview

This project implements a deep learning system for automatic detection of **Invasive Ductal Carcinoma (IDC)** in breast histopathology images. Using a ResNet18-based model, it classifies 50×50 pixel image patches into two categories: **cancer (1)** and **no cancer (0)**.

### Key Metrics
- **Dataset**: 277,524 images from 162 patients
- **Accuracy**: High detection accuracy for cancerous cells
- **Visualization**: Ability to visualize predictions on original tissue images
- **Performance**: Optimized for large-scale data processing

## ✨ Features

- 🏥 **Automatic cancer detection** in histopathology images
- 🧠 **Transfer Learning** with pre-trained ResNet18
- 📊 **Detailed visualization** of results and metrics
- 🔧 **Flexible training parameters** (LR search, augmentations, class weights)
- 💾 **Result caching** for faster repeated runs
- 📈 **Training monitoring** with loss and accuracy plots
- 🎯 **Weighted loss function** for handling imbalanced classes

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
