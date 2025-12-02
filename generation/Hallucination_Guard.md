# Advanced Generation: Hallucination Guard Logic

We are addressing the **LLM Hallucinations** challenge using **Semantic Active Generative Augmentation**, which consists of these 3 checks:

- SSIM Check: Ensures identity preservation (Statistical fidelity).
- CLIP Check: Semantic filtering (Cosine similarity).
- ViT Check: Classifier-in-the-Loop confidence filtering.

---

### 1. Semantic Filtering (CLIP Consistency)

Semantic Filtering is done to calculate the **cosine similarity** between the generated image and the source text prompt.

  * **Concept:** If we asked Gemini for "anger" but it generated a confusing or neutral face, a Vision-Language model (like CLIP) will give it a low score because the image doesn't match the word "anger."
  * **Rule:** Only images with a high similarity score are accepted.
  * **Why it helps:** It catches "semantic hallucinations" where the image looks real (photorealistic) but is the **wrong emotion**.

**Implementation:** OpenAI's CLIP model (available via HuggingFace) to score `(image, prompt)`.

### 2. Confidence Filtering (Safety Valve)

This is the simplest method. We apply classifier confidence filtering, dropping low-confidence synthetic samples.

  * **Concept:** Even if the CITL (Classifier-in-the-Loop) says the image is "Happy", check *how sure* it is.
  * **Rule:** If the ViT classifier predicts "Happy" but with only **51% probability**, discard it.
  * **Why it helps:** Hallucinations often manifest as ambiguous or "messy" images. Classifiers usually have low confidence on these artifacts. Setting a high bar ensures only distinct, high-quality emotions are kept.

### 3. Structural Similarity (Identity Preservation)

Its important to maintain statistical fidelity to the original dataset and avoiding low-quality artifacts.

  * **Concept:** We want to change the *emotion*, not the *person*. If Gemini hallucinates a completely new person or turns a photo into a cartoon, the structure of the image changes too much.
  * **Metric:** **SSIM (Structural Similarity Index)** is used to measure this.
  * **Implementation:** Compare the **Original Image** vs. **Augmented Image**.
      * If SSIM is too low: The image changed too much (Hallucination/Identity loss).
      * If SSIM is too high: The model didn't change anything (Failure).

## Summary of Hallucination Guard in Generation

Semantic Active Generative Augmentation is a robust framework, the filtering pipeline looks like this:

1.  **Generate** (Gemini)
2.  **Filter 1: Structure Check** (Did the face disappear or change?)
3.  **Filter 2: Semantic Check** (Does CLIP think this looks like `target_emotion`?)
4.  **Filter 3: Classifier Check** (Does ViT think this is `target_emotion` with high confidence?)

If an image passes all three, it is saved. Else, discarded and the slot is refunded so `TargetBalancingManager` knows the actual count of images required to balance the dataset. 