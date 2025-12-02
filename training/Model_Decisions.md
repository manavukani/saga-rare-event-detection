# Model Architecture & Training Strategy

**Project:** AffectNet Emotion Classification (Augmentation Impact Analysis)
**Context:** Small-data regime (~30k images) focusing on mitigating severe class imbalance.

## 1. Core Architecture: Vision Transformer (ViT)

* **Model:** `vit_base_patch16_224` (Pretrained on ImageNet-1k).
* **Rationale:** ViT processes images as sequences of patches via Self-Attention. This allows the model to capture global, long-range dependencies (e.g., the geometric relationship between a mouth curve and eye shape) regardless of pixel distance, which is superior to CNNs for capturing subtle facial micro-expressions.

## 2. Classification Head Design (Custom)

We replaced the standard linear projection with a non-linear MLP head to better bridge the semantic gap between generic ImageNet features and specific AffectNet emotion classes.

* **Structure:** `Linear(768 -> 512)` $\to$ `BatchNorm` $\to$ `GELU` $\to$ `Dropout(0.3)` $\to$ `Linear(512 -> 8)`
* **Rationale:** The intermediate hidden layer allows for feature recombination before classification. `Dropout` mitigates overfitting on the small dataset, and `GELU` ensures smoother gradient flow than ReLU for Transformer architectures.

## 3. Fine-Tuning Strategy (Partial Unfreezing)

We employed a **"block-wise" fine-tuning approach** to balance adaptation with stability.

* **Frozen:** Transformer Encoder Blocks 0–9 (General visual features: edges, textures, shapes).
* **Trainable:** Transformer Encoder Blocks 10–11 + MLP Head.
* **Impact:** This preserves the robust low-level vision capabilities learned from ImageNet (preventing catastrophic forgetting) while forcing the final layers to adapt their semantic understanding to the nuances of human facial muscles.

## 4. Data Integrity & The "Fixed Balanced" Test Protocol

To strictly measure the impact of Generative AI augmentation, we implemented a rigorous "Golden Test Set" protocol. We rejected standard random splitting in favor of a fixed isolation strategy:

* **Source Integrity (Original Only):** The test set consists *exclusively* of real images from the original AffectNet dataset. No augmented or synthetic images are permitted in evaluation.
* **Balanced Class Distribution:** We isolated a fixed set of **150 images per class** (Total: 1,200 images). This balance is critical; it prevents the majority classes (e.g., "Happy") from skewing the accuracy metrics, forcing the evaluation to weight every emotion equally.
* **Strict Isolation & Reuse:**
  1. This test set was generated and saved to disk (`saved_testset.csv`) during the Baseline data prep.
  2. The Augmented experiments were forced to load this exact CSV.
  3. These IDs were mathematically excluded from the Augmented training pool.
  * **Result:** Both the Baseline and Augmented models are evaluated on the exact same unseen, real-world data. Any performance delta is therefore directly attributable to the training data quality, not sampling variance.

## 5. Evaluation Methodology

Global accuracy is insufficient for imbalanced data. We implemented a granular evaluation pipeline:

* **Minority Class Analysis:** By using a balanced test set (150 per class), we can directly compare the "Accuracy Delta" for minority classes. The primary objective is to determine if GenAI augmentation specifically resolves the model's struggle with under-represented classes (e.g., "Contempt," "Disgust") compared to the baseline.
* **Confusion Matrix Analysis:** Final evaluation includes a breakdown of misclassifications (e.g., confusing "Fear" with "Surprise") to diagnose if augmentation reduced specific inter-class ambiguities.

## 6. Augmentation Pipeline

To improve generalization and invariance to head pose, the following transforms are applied during training:

* **Geometric:** `RandomHorizontalFlip`, `RandomRotation(±15°)` (Crucial for handling natural head tilts in "in-the-wild" images).
* **Photometric:** `ColorJitter` (Brightness 0.2, Contrast 0.2) to account for lighting variations.
* **Normalization:** Standard ImageNet mean/std normalization.

## 7. Hyperparameters

* **Optimizer:** `AdamW` (Weight Decay: $1e-4$). Standard for Transformers to handle parameter magnitude.
* **Learning Rate:** $1e-4$ with **Cosine Annealing** scheduler (warm restart/decay).
* **Loss Function:** `CrossEntropyLoss` with **Label Smoothing (0.1)**.
  * *Reasoning:* Facial expressions are inherently ambiguous. Smoothing prevents the model from becoming overconfident on noisy labels and encourages tighter clustering in the embedding space.
* **Early Stopping:** Patience of 5 epochs (monitoring Validation Accuracy) to prevent memorization of the 30k+ dataset.