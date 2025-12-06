# Model Training Pipeline - README

## Overview

The **training** folder contains the complete workflow for training and evaluating Vision Transformer (ViT) models on the AffectNet emotion recognition task. This includes:
- **Baseline Training:** Model trained on original real-world data
- **Augmented Training:** Model retrained on original + SAGA-generated synthetic data
- **Golden Test Set Protocol:** Rigorous evaluation methodology ensuring fair comparison

## Folder Structure

```
training/
├── baseline_training.ipynb      # Phase A: Train baseline model
├── augmented_training.ipynb     # Phase C: Retrain with augmented data
├── Model_Decisions.md           # Architecture & strategy documentation
├── models.txt                   # Model configuration details
└── README.md                    # This file
```

## Prerequisites

- **GPU Access:** Kaggle T4 GPU (2x recommended) or Google Colab Pro with GPU
- **Python 3.8+** (typically available in Kaggle/Colab)
- **7 GB+ RAM** (minimum for ViT-Base training)
- **Storage:** 50GB for dataset + model checkpoints

## Architecture Overview

```
Input: 224×224×3 RGB Image
  ↓
Patch Embedding: 16×16 patches → 196 patches of 768 dims
  ↓
Transformer Encoder: 12 blocks (only last 2 trainable)
  ↓
Custom Classification Head:
  Linear(768 → 512) → BatchNorm → GELU → Dropout(0.3) → Linear(512 → 8)
  ↓
Output: 8 emotion classes
```

### Fine-Tuning Strategy

| Layer | Status | Reason |
|-------|--------|--------|
| Transformer Blocks 0-9 | **Frozen** | Preserve ImageNet visual features |
| Transformer Blocks 10-11 | **Trainable** | Adapt semantic understanding |
| Custom MLP Head | **Trainable** | Learn emotion-specific features |

## Installation & Setup

### 1. Kaggle Notebook Setup

1. **Create New Notebook** on Kaggle
2. **Add Datasets as Input:**
   - AffectNet dataset: [Kaggle Link](https://www.kaggle.com/datasets/manavukani/affectnet)
3. **Enable GPU:**
   - Settings → Accelerator → GPU (T4 if available)

## Running the Pipeline

### Phase A: Baseline Training

**Steps:**

1. **Open `baseline_training.ipynb` in Kaggle/Jupyter**

2. **Configure Paths (Cell 1):**
   ```python
   DATASET_ROOT = '/kaggle/input/affectnet'
   OUTPUT_DIR = '/kaggle/working/baseline_model'
   LABELS_CSV = '/kaggle/input/affectnet/labels.csv'
   ```

3. **Run All Cells:**
   - The notebook handles data loading, stratified splitting, and model training automatically
   - Estimated runtime: **2-4 hours** on Kaggle T4

4. **Outputs Generated:**
   ```
   /kaggle/working/baseline_model/
   ├── vit_baseline_best_model.pth     # Best checkpoint
   ├── testset_saved.csv               # Golden Test Set IDs
   └── confusion_matrix_baseline.png   # Evaluation plot
   ```

### Phase B: Data Augmentation (refer `./generation`)

### Phase C: Augmented Training

**Purpose:** Retrain ViT model using original + SAGA-augmented data; compare with baseline.

**Prerequisites:**
- ✓ Baseline model trained (Phase A complete)
- ✓ Augmented dataset generated (Phase B complete)
- ✓ `testset_saved.csv` available from Phase A

**Steps:**

1. **Upload Augmented Dataset:**
   - Download `augmented_affectnet_dataset.zip` from generation phase
   - Upload to Kaggle as new dataset
   - Add to notebook as input

2. **Prepare Combined Labels:**
   - Run `../generation/combine_labels.ipynb`
   - Generates `combined_labels.csv` merging:
     - Original dataset labels
     - Augmented dataset labels
     - Ensures proper emotion class mapping

3. **Open `augmented_training.ipynb` in Kaggle**

4. **Configure Paths (Cell 1):**
   ```python
   DATASET_ROOT = '/kaggle/input/affectnet'
   AUGMENTED_ROOT = '/kaggle/input/augmented-affectnet-saga'
   COMBINED_LABELS = '/kaggle/working/combined_labels.csv'
   TESTSET_CSV = '/kaggle/working/testset_saved.csv'  # From baseline
   OUTPUT_DIR = '/kaggle/working/augmented_model'
   ```

5. **Run All Cells:**
   - Loads training data (original + augmented)
   - Excludes test set images mathematically
   - Trains new ViT model for specified epochs
   - Evaluates on frozen test set
   - Estimated runtime: **3-5 hours** on Kaggle T4

6. **Outputs Generated:**
   ```
   /kaggle/working/augmented_model/
   ├── vit_augmented_best_model.pth    # Best checkpoint
   └── confusion_matrix_augmented.png  # Evaluation plot
   ```