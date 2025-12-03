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

### Install Dependencies

```bash
# Basic dependencies
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install scikit-learn pillow tqdm
pip install kagglehub

# For image processing
pip install scikit-image imageio

# For Colab/Google Drive
pip install -U -q kagglehub

### Download Datasets

```bash
import kagglehub
pretrained_models = kagglehub.dataset_download('pvlima/pretrained-pytorch-models')
breast_images = kagglehub.dataset_download('paultimothymooney/breast-histopathology-images')
breast_model = kagglehub.dataset_download('allunia/breastcancermodel')

### Usage
# Set configuration
run_training = False  # Set to True for training, False for inference
retrain = False
find_learning_rate = False

# Model parameters
BATCH_SIZE = 32
NUM_CLASSES = 2
NUM_EPOCHS = 30

# Paths
OUTPUT_PATH = "./output/"
MODEL_PATH = "/root/.cache/kagglehub/datasets/allunia/breastcancermodel/versions/8/"

# Training execution
if run_training:
    # Training code here
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01)
    scheduler = get_scheduler(optimizer, start_lr, end_lr, 2*NUM_EPOCHS)
    results = train_loop(model, criterion, optimizer, scheduler=scheduler, num_epochs=NUM_EPOCHS)
else:
    # Inference code here
    model.load_state_dict(torch.load(MODEL_PATH + "_cuda.pth"))
    model.eval()

###Complete Pipeline


```bash
# 1. Data Loading
base_path = "/kaggle/input/breast-histopathology-images/IDC_regular_ps50_idx5/"
data = pd.DataFrame(columns=["patient_id", "path", "target"])

# 2. Data Preprocessing
train_df, test_df, dev_df = train_test_split(data)

# 3. Model Initialization
model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
    nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)

# 4. Training/Inference
if run_training:
    results = train_loop(model, criterion, optimizer, num_epochs=NUM_EPOCHS)
else:
    predictions = evaluate_model(model, test_dataloader)

###Architecture

```bash
ResNet18 Backbone
    ↓
Global Average Pooling
    ↓
Fully Connected Layers:
    • Linear(512) → ReLU → BatchNorm → Dropout(0.5)
    • Linear(256) → ReLU → BatchNorm → Dropout(0.5)
    • Linear(2) → Softmax

###Training Loop

```bash
def train_loop(model, criterion, optimizer, lr_find=False, scheduler=None, num_epochs=30):
    """Main training loop with validation"""
    for epoch in range(num_epochs):
        for phase in ["train", "dev", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            for batch in dataloaders[phase]:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

###Training Loop

```bash
def train_loop(model, criterion, optimizer, lr_find=False, scheduler=None, num_epochs=30):
    """Main training loop with validation"""
    for epoch in range(num_epochs):
        for phase in ["train", "dev", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            for batch in dataloaders[phase]:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

###Where Training Happens

```bash
if run_training:
    NUM_EPOCHS = 30
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01)
    scheduler = get_scheduler(optimizer, start_lr, end_lr, 2*NUM_EPOCHS)
    results = train_loop(model, criterion, optimizer, scheduler=scheduler, num_epochs=NUM_EPOCHS)
    torch.save(model.state_dict(), MODEL_PATH + "_cuda.pth")
