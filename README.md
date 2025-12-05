# From Scarcity to Scale: Generative Augmentation in Big Data Ecosystems

> Semantic Active Generative Augmentation (SAGA)

**Team:** Group 7 (Manav Ukani, Dhruv Shah, Param Patel, Neel Modi, Astha Soni)

## Project Overview

This project introduces **Semantic Active Generative Augmentation (SAGA)**, a framework designed to address data scarcity in long-tail visual domains. While traditional augmentation (rotation, flipping) adds limited value, SAGA leverages **Generative AI** to synthesize targeted, realistic training data for underrepresented classes.

We applied this framework to the **AffectNet** dataset (facial emotion recognition), specifically targeting minority classes like "Fear" and "Contempt." The system moves beyond passive augmentation by implementing a **tight feedback loop** where classification errors and dataset imbalances actively shape the generation of new samples.

## The Pipeline

Our solution follows a multi-stage approach:

1.  **Data Preparation:** Curation of AffectNet and creation of a "Golden Test Set" (1,200 real images) strictly isolated from training.
2.  **Targeted Generation:** A `TargetBalancingManager` identifies class deficits and actively prompts **Gemini 1.5 Flash (NanoBanana)** to generate missing samples.
3.  **The "Hallucination Guard" (Quality Control):** A multi-gate filter to prevent generative artifacts:
      * **Gate 1 (Structure):** **SSIM** check to ensure identity preservation.
      * **Gate 2 (Semantics):** **CLIP** cosine similarity to ensure the image matches the emotion prompt.
      * **Gate 3 (Confidence):** **Classifier-in-the-Loop (ViT)** to reject ambiguous expressions.
4.  **Evaluation:** Training a Vision Transformer (ViT-Base) on *Real* vs. *Real + SAGA* datasets to measure performance deltas.

## Repository Structure (Key Files)

```bash
│
├── generation/                  
│   ├── augmentation_pipeline.py # Main script
│   ├── combine_labels.ipynb     # Helper script to merge Original + Augmented Dataset Labels
│   └── Hallucination_Guard.md   # Logic explanation
│
├── training/                    
│   ├── baseline_training.ipynb  # Baseline Model Script
│   ├── augmented_training.ipynb # Model Retraining Script on Combined Data
│   └── Model_Decisions.md       # Architecture choices
│
├── ui/                          # Demo Application
│   ├── streamlit_app.py         # Streamlit web app for real-time emotion detection
│   ├── requirements.txt         # UI dependencies
│   └── models/                  # Pre-trained model weights (download from Kaggle)
│
├── .gitignore                   # Ignore heavy dataset, etc.
└── README.md                    # What you are reading ;)
```

### **Note:**
- The AffectNet dataset is large (\>5GB). Use this [Kaggle Link - Dataset](https://www.kaggle.com/datasets/manavukani/affectnet) to access the augmented dataset.
- Download fine tuned models from this [Kaggle Link - Models](https://www.kaggle.com/datasets/manavukani/my-models/data) 

-----

## How to Run - Interactive Demo

We've built a real-time emotion detection demo using Streamlit that allows you to:

* **Upload Images:** Test the model on your own facial images
* **Live Webcam Detection:** Real-time emotion recognition from your webcam
* **Model Comparison:** Switch between baseline and SAGA-augmented models
* **Batch Processing:** Upload multiple images at once

### Installation

1. **Navigate to the UI folder:**
   ```bash
   cd ui
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate # Windows
   ```

3. **Install dependencies:**
   
   For **CPU-only** setup:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   
   For **GPU (CUDA 12.1)** setup:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

4. Downloading the Model Weights from Kaggle:
      - Use this [link](https://www.kaggle.com/datasets/manavukani/my-models/data) for downloading the model weights.
      - Add them to `ui/models` folder

### Running the Demo

1. **Start the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the app:**
   Open your browser and navigate to `http://localhost:8501`

3. **Features:**
   * **Image Upload Mode:** Upload single or multiple facial images to see emotion predictions
   * **Webcam Mode:** Enable real-time emotion detection (requires webcam access)
   * **Confidence Threshold:** Adjust the minimum confidence for predictions
   * **Model Selection:** Compare results between baseline and augmented models

### Model Files

The UI requires pre-trained model weights in the `ui/models/` directory:
* `vit_baseline_best_model.pth` - Baseline ViT model
* `vit_augmented_best_model.pth` - SAGA-augmented ViT model

> **Note:** Webcam functionality requires `streamlit-webrtc` and system `ffmpeg` for full real-time support. The app gracefully falls back to image upload mode if webcam dependencies are unavailable.

-----

## How to Run - SAGA Pipeline, Model Training

**Prerequisites:** This project requires GPU acceleration (Kaggle T4 x2 or Colab Pro recommended).

### 1\. Environment Setup

  * Open a new Notebook (Kaggle/Colab).
  * Install dependencies.

### 2\. API Key Configuration

This project uses the Gemini API for generation.

  * Add your keys to Kaggle Secrets as `GEMINI_API_KEY`.
  * *Note:* The `KeyManager` class in `augmentation_pipeline.py` supports multiple keys to handle rate limits via rotation.

### 3\. Execution Steps

**Phase A: Training**

1.  Use original dataset as an input.
3.  Run `baseline_training.ipynb`. This script automatically handles:
      * Stratified train/val splitting.
      * Exclusion of the "Golden Test Set"
      * Saving the `testset_saved.csv` for evaluation across retrained model/
      * Fine-tuning the ViT-Base model.

**Phase B: Augmentation (Long-Running)**

1.  Run `augmentation_pipeline.py`.
2.  Set `TEST_MODE = False`.
3.  The script includes a `TargetBalancingManager` that will stop automatically when all classes reach the cap (Equally balanced distribution).
4.  Download `affectnet_augmented.zip` from the outputs.

**Phase C: Re-Training**

1.  Upload the augmented zip as an input dataset.
2.  Run `combine_labels.ipynb` to create the master CSV mapping.
3.  Run `augmented_training.ipynb`. This script automatically handles:
      * Stratified train/val splitting.
      * Exclusion of the "Golden Test Set" from the `baseline_training.ipynb` outputs.
      * Fine-tuning the ViT-Base model.

-----

## Key Results
The README section has been updated with the new performance metrics extracted from your logs.

Here is the revised section:

***

We compared a Baseline ViT (trained on raw data) against the SAGA-Augmented ViT. Both were evaluated on the **exact same** unseen real-world test set.

| Metric | Baseline Model | SAGA Augmented | Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **60.17%** | **64.67%** | **+4.50%** |
| **Contempt (Minority)** | 52.67% | 59.33% | +6.66% |
| **Fear (Minority)** | 60.67% | 66.67% | +6.00% |

**Insights:**

* **Targeted Improvement:** The most significant gains were achieved in the minority, or "long-tail," classes, with **Contempt** seeing the largest improvement of **+6.66%**.
* **Semantic Distinction:** The augmentation successfully improved the model's ability to classify difficult emotions, reducing confusion between similar classes (e.g., 'Fear' and 'Surprise' in the confusion matrix).
* **Quality Over Quantity:** The **SAGA Hallucination Guard** proved effective, leading to a substantial **4.50% absolute accuracy increase** without overfitting to synthetic data.