# SAGA Emotion Recognition UI - README

## Overview

The **UI** folder contains an interactive Streamlit web application for real-time emotion recognition using the trained SAGA models. This demo allows you to:

- **Upload Images:** Test emotion detection on your own facial images
- **Live Webcam:** Real-time emotion recognition from your webcam feed
- **Model Comparison:** Switch between baseline and SAGA-augmented models
- **Batch Processing:** Upload and analyze multiple images at once
- **Confidence Adjustment:** Filter predictions by confidence threshold

## Folder Structure

```
ui/
├── streamlit_app.py              # Main Streamlit application
├── webcam_emotion.py             # Webcam integration utilities
├── requirements.txt              # Python dependencies
├── models/                       # Pre-trained model weights
│   ├── vit_baseline_best_model.pth      # Baseline ViT
│   ├── vit_augmented_best_model.pth     # SAGA-augmented ViT
│   └── saga_emotion_model.onnx          # ONNX-optimized model (optional)
└── README.md                     # This file
```

## Prerequisites

- **Python 3.9+**
- **Webcam** (optional, for live detection mode)
- **Modern Web Browser** (Chrome, Firefox, Edge, Safari)
- **GPU** (strongly recommended, but CPU will work slower)
- **7 GB+ Storage** for model weights

## Installation

### Step 1: Clone/Download Repository

```bash
cd ui
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Download Model Weights

**From Kaggle (Recommended)**

1. Visit [Kaggle Models Link](https://www.kaggle.com/datasets/manavukani/my-models/data)
2. Download:
   - `vit_baseline_best_model.pth`
   - `vit_augmented_best_model.pth`
3. Place in `ui/models/` folder

### Step 4: Install Dependencies

#### For CPU-Only Setup

```bash
# Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

#### For GPU Setup (CUDA 12.1)

```bash
# Install PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## Running the Application

### Start the Streamlit Server

```bash
streamlit run streamlit_app.py
```

### Access the Application

The app will automatically open in your browser. If not:
- **Local:** http://localhost:8501
- **Network:** Use the URL shown in terminal (if running remotely)

## Quick Start Command

```bash
cd ui
python -m venv venv && venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run streamlit_app.py
```
