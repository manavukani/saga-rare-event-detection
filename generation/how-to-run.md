# SAGA Generation Pipeline - README

## Folder Structure

```
generation/
├── augmentation_pipeline.py      # Main execution script
├── combine_labels.ipynb          # Helper to merge original + augmented labels
├── Hallucination_Guard.md        # Quality control documentation
├── requirements.txt              # Dependencies
└── combined_labels.csv           # Output CSV with merged labels
```

## Prerequisites

- **GPU Access:** Kaggle T4 GPU (x2 recommended) or Google Colab Pro with GPU
- **Python 3.8+**
- **Gemini API Key** (get from [Google AI Studio](https://aistudio.google.com/apikey))
- **Kaggle Account** (for dataset access)

## Installation

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Kaggle Secrets Configuration

For **Kaggle Notebook Environment:**

1. Open your Kaggle Notebook
2. Go to **Settings** → **Secrets**
3. Add your Gemini API keys

**Why multiple keys?** The `KeyManager` class rotates through API keys to handle rate limits gracefully.

### 3. Dataset Preparation

1. Download the **AffectNet dataset** from [Kaggle Link](https://www.kaggle.com/datasets/manavukani/affectnet)
2. Ensure the following structure:
   ```
   /kaggle/input/affectnet/
   ├── labels.csv
   ├── ... (8 emotion folders)
   ```
3. Update `DATASET_ROOT` in `augmentation_pipeline.py` if using a different path


## Running the Pipeline

1. **Upload Dataset:**
   - In Kaggle Notebook, add the AffectNet dataset as input
   - Add the baseline ViT model weights from [Kaggle](https://www.kaggle.com/datasets/manavukani/my-models)

2. **Run the Script:**
   ```bash
   !python augmentation_pipeline.py
   ```

## The Pipeline Execution Flow

### Phase 1: Initialization
- Loads ViT classifier weights
- Initializes CLIP and SSIM models
- Sets up output directories
- Analyzes original dataset class distribution

### Phase 2: Targeted Generation
- **TargetBalancingManager** identifies class deficits
- For each minority class, generates new samples via Gemini API
- Maintains a queue of images waiting for quality checks

### Phase 3: Hallucination Guard Filtering
```
Generated Image
    ↓
[Gate 1: SSIM Check] → Identity preserved?
    ↓ (Pass)
[Gate 2: CLIP Check] → Semantic match to emotion?
    ↓ (Pass)
[Gate 3: ViT Check] → High classifier confidence?
    ↓ (Pass)
[Save Image] → Added to augmented dataset
```

### Phase 4: Output Generation
- Saves all passing synthetic images to `OUTPUT_DIR`
- Generates `augmented_labels.csv.csv`
- After successful execution, you'll find augmented images inside the `augmented_affectnet` folder


### After generation completes

1. **Combine Labels:**
   - Run `combine_labels.ipynb` to merge original + augmented labels into a single CSV
   - This prepares data for the training pipeline

2. **Train Augmented Model:**
   - Upload the augmented dataset to Kaggle
   - Run `../training/augmented_training.ipynb`
   - Compare with baseline model performance