# !pip install -q transformers torch scikit-image

import os
import random
import time
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm
import numpy as np

# --- AI & CV LIBRARIES ---
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient
import io
import itertools
import shutil
import torch
from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    CLIPProcessor, CLIPModel
)
from skimage.metrics import structural_similarity as ssim
from IPython.display import FileLink

# --- CONFIGURATION ---
DATASET_ROOT = '/kaggle/input/affectnet' 
LABELS_PATH = os.path.join(DATASET_ROOT, 'labels.csv')
OUTPUT_DIR = '/kaggle/working/augmented_affectnet'

# ️ SET TO FALSE FOR PRODUCTION
TEST_MODE = False 
TEST_SIZE = 20

RESUME_CSV_PATH = None 

# --- THRESHOLDS (Tunable Hyperparameters) ---
# 1. Structural Similarity: Don't change the person too much, but ensure SOME change happens.
SSIM_MIN = 0.35  # Below this = Identity lost / Hallucination
SSIM_MAX = 0.95  # Above this = The model didn't change anything

# 2. Semantic Consistency (CLIP): Score between Image and "A photo of a [emotion] face"
CLIP_THRESHOLD = 0.22 

# 3. Classifier Confidence (ViT): Must predict target emotion with high confidence
VIT_THRESHOLD = 0.65

# --- SECRETS SETUP ---
user_secrets = UserSecretsClient()
API_KEYS_DICT = {
    "ManavTesting": user_secrets.get_secret("ManavTesting"),
    "ManavFresh": user_secrets.get_secret("ManavFresh"),
    "Astha": user_secrets.get_secret("Astha"),
    "Neel": user_secrets.get_secret("Neel"),
    "Dhruv": user_secrets.get_secret("Dhruv"),
    "Param": user_secrets.get_secret("Param"),
}

# --- CONFIGURATION FOR CUSTOM MODEL ---
LOCAL_VIT_WEIGHTS_PATH = '/kaggle/input/baseline-vit/vit_baseline_best_model.pth' 
BASE_ARCH_NAME = 'google/vit-base-patch16-224' # Base model for fine-tuned architecture

# --- 1. HALLUCINATION GUARD ---
'''
"Hallucination Guard" uses multi-stage processing:
- SSIM Check: Ensures identity preservation (Statistical fidelity).
- CLIP Check: Semantic filtering (Cosine similarity).
- ViT Check: Classifier-in-the-Loop confidence filtering.
'''
class HallucinationGuard:
    def __init__(self):
        print("\n Initializing Hallucination Guard (Custom Baseline)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- 1. LOAD YOUR CUSTOM CLASSIFIER ---
        print(f"   - Loading Custom ViT Weights from: {LOCAL_VIT_WEIGHTS_PATH}")
        
        # A. Define the Label Map (MUST Match your training exactly)
        # Assuming you used the standard mapping from your proposal
        self.id2label = {
            0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
            4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # B. Initialize Empty Architecture
        # We assume you used a standard ViT architecture with 8 output classes
        try:
            self.vit_model = ViTForImageClassification.from_pretrained(
                BASE_ARCH_NAME,
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True # Necessary when resizing the classification head
            )
            
            # C. Load Your State Dict (The Weights)
            # map_location ensures it loads even if saved on GPU but running on CPU (or vice versa)
            state_dict = torch.load(LOCAL_VIT_WEIGHTS_PATH, map_location=self.device)
            
            # Handle cases where state_dict keys might have prefixes like 'module.' or 'model.'
            # (Common if trained with PyTorch Lightning or DataParallel)
            clean_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "").replace("module.", "")
                clean_state_dict[new_key] = v
                
            self.vit_model.load_state_dict(clean_state_dict, strict=False)
            self.vit_model.to(self.device)
            self.vit_model.eval() # Set to evaluation mode!
            
            # D. Load Standard Processor (for resizing/normalization)
            # Use the processor associated with the BASE architecture
            self.vit_processor = ViTImageProcessor.from_pretrained(BASE_ARCH_NAME)
            
            print("   Custom ViT Loaded Successfully.")
            
        except Exception as e:
            print(f"  FATAL: Could not load custom model. Error: {e}")
            raise e

        # --- 2. LOAD CLIP (Standard) ---
        print("   - Loading CLIP (Semantic Filter)...")
        clip_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)

    def check_ssim(self, original_pil, generated_pil):
        """Checks structural similarity to preserve identity."""
        # Convert to grayscale numpy arrays
        img1 = np.array(original_pil.convert('L'))
        img2 = np.array(generated_pil.convert('L'))
        
        # Resize generated to match original if needed
        if img1.shape != img2.shape:
            img2 = np.array(generated_pil.resize(original_pil.size).convert('L'))
            
        score, _ = ssim(img1, img2, full=True)
        return SSIM_MIN <= score <= SSIM_MAX, score

    def check_semantic_clip(self, image_pil, target_emotion):
        """Checks if image matches the text prompt semantically."""
        text = [f"a photo of a {target_emotion} face"]
        inputs = self.clip_processor(text=text, images=image_pil, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            
        # CLIP score (normalized roughly)
        score = outputs.logits_per_image.item() / 100.0
        return score > CLIP_THRESHOLD, score

    def check_classifier_vit(self, image_pil, target_emotion):
        # Ensure image matches training size (usually 224x224 for ViT)
        # The processor handles this, but explicit check doesn't hurt
        inputs = self.vit_processor(images=image_pil, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_idx = torch.max(probs, 1)
        
        # Use YOUR label map
        pred_label_idx = top_idx.item()
        pred_label = self.id2label[pred_label_idx].lower()
        confidence = top_prob.item()
        
        is_match = (pred_label == target_emotion.lower())
        is_confident = (confidence >= VIT_THRESHOLD)
        
        return (is_match and is_confident), pred_label, confidence

    def validate(self, original_img, generated_img, target_emotion):
        """Runs the full battery of tests."""
        
        # 1. SSIM Check
        pass_ssim, ssim_score = self.check_ssim(original_img, generated_img)
        if not pass_ssim:
            return False, f"SSIM Fail ({ssim_score:.2f})"
        
        # 2. CLIP Check
        pass_clip, clip_score = self.check_semantic_clip(generated_img, target_emotion)
        if not pass_clip:
            return False, f"CLIP Fail (Score: {clip_score:.2f} < {CLIP_THRESHOLD})"
            
        # 3. ViT Check (Classifier-in-the-Loop)
        pass_vit, pred_label, conf = self.check_classifier_vit(generated_img, target_emotion)
        if not pass_vit:
            return False, f"ViT Reject (Pred: {pred_label} @ {conf:.2f})"
            
        return True, "APPROVED"


# --- 2. BASE INFRASTRUCTURE (KeyManager, Balancer, API) ---

class KeyManager:
    def __init__(self, key_dict):
        self.clients = []
        for name, key_string in key_dict.items():
            if key_string:
                client = genai.Client(api_key=key_string)
                self.clients.append({"name": name, "client": client})
        if not self.clients: raise ValueError("No valid API keys found!")
        self.client_cycle = itertools.cycle(self.clients)

    def get_next_client_info(self):
        return next(self.client_cycle)

key_manager = KeyManager(API_KEYS_DICT)

EMOTION_MAP = {0:'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt'}
EMOTIONS = list(EMOTION_MAP.values())

class TargetBalancingManager:
    def __init__(self, df, label_col='label'):
        self.counts = df[label_col].value_counts().to_dict()
        for emo in EMOTIONS:
            if emo not in self.counts: self.counts[emo] = 0
        self.cap = max(self.counts.values()) if self.counts else 100
        self.emotion_cycle = itertools.cycle(EMOTIONS)

    def load_previous_progress(self, csv_path):
        if not csv_path or not os.path.exists(csv_path): return
        print(f" Loading progress from: {csv_path}")
        try:
            prev_df = pd.read_csv(csv_path)
            prev_df.columns = [c.strip() for c in prev_df.columns]
            if 'augmented_label' in prev_df.columns:
                prev_df['augmented_label'] = prev_df['augmented_label'].astype(str).str.strip().str.lower()
                generated_counts = prev_df['augmented_label'].value_counts().to_dict()
                for emo, count in generated_counts.items():
                    if emo in self.counts: self.counts[emo] += count
                print(f"    Counts updated.")
        except Exception as e: print(f"    Error reading resume file: {e}")

    def print_remaining_work(self):
        print("\n Current Balancing Status:")
        total_remaining = 0
        for emo in EMOTIONS:
            current = self.counts.get(emo, 0)
            remaining = max(0, self.cap - current)
            total_remaining += remaining
            print(f"   {emo:<10} | {current:<6} | {self.cap:<6} | {remaining}")
        print("-" * 40)
        print(f" TOTAL TO GENERATE: {total_remaining}\n")

    def get_target_emotion(self, current_label):
        for _ in range(len(EMOTIONS)):
            candidate = next(self.emotion_cycle)
            if candidate == current_label: continue
            if self.counts.get(candidate, 0) >= self.cap: continue
            self.counts[candidate] += 1
            return candidate
        return None

def call_gemini_with_backoff(image_path, current_emotion, target_emotion, max_retries=3):
    try:
        # Load Original Image for returning later (needed for SSIM check)
        original_pil = Image.open(image_path).convert('RGB')
        
        # Prepare bytes for API
        img_byte_arr = io.BytesIO()
        original_pil.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        
    except FileNotFoundError:
        return None, None, None

    prompt = (
        f"Edit this image. The person currently has a {current_emotion} expression. "
        f"Change their facial expression to {target_emotion}. "
        f"Maintain the exact same identity, lighting, and background. "
        f"Output only the modified image."
    )

    delay = 1 

    for attempt in range(max_retries):
        try:
            client_info = key_manager.get_next_client_info()
            current_client = client_info['client']
            
            response = current_client.models.generate_content(
                model='gemini-2.5-flash-image',
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE")
                    ]
                )
            )

            if response.candidates:
                candidate = response.candidates[0]
                if not candidate.content: return None, None, original_pil # Safety Block

                if candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data:
                            generated_pil = Image.open(io.BytesIO(part.inline_data.data)).convert('RGB')
                            return "Success", generated_pil, original_pil

            return None, None, original_pil

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(delay)
            else:
                return None, None, original_pil

    return None, None, original_pil

# --- 3. MAIN LOOP ---

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    for emo in EMOTIONS: os.makedirs(os.path.join(OUTPUT_DIR, f"aug_{emo.lower()}"), exist_ok=True)

    print(f"Loading labels from {LABELS_PATH}...")
    df = pd.read_csv(LABELS_PATH)
    df.columns = [c.strip() for c in df.columns]
    df['label'] = df['label'].astype(str).str.strip().str.lower()

    # Initialize Guards
    target_manager = TargetBalancingManager(df, label_col='label')
    
    guard = HallucinationGuard()

    results = []
    processed_source_images = set()

    checkpoint_path = os.path.join(OUTPUT_DIR, 'augmented_labels_checkpoint.csv')
    resume_file = RESUME_CSV_PATH if (RESUME_CSV_PATH and os.path.exists(RESUME_CSV_PATH)) else None
    if not resume_file and os.path.exists(checkpoint_path): resume_file = checkpoint_path

    if resume_file:
        try:
            target_manager.load_previous_progress(resume_file)
            prev_df = pd.read_csv(resume_file)
            results = prev_df.to_dict('records')
            if 'original_path' in prev_df.columns:
                processed_source_images = set(prev_df['original_path'].astype(str).str.strip())
        except: pass

    target_manager.print_remaining_work()

    if TEST_MODE:
        processing_queue_df = df.sample(n=min(TEST_SIZE, len(df)), random_state=42).reset_index(drop=True)
    else:
        if processed_source_images:
            df_filtered = df[~df['pth'].isin(processed_source_images)]
            processing_queue_df = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            processing_queue_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Queue ready: {len(processing_queue_df)} tasks.")
    success_count_session = 0

    for index, row in tqdm(processing_queue_df.iterrows(), total=len(processing_queue_df)):
        try:
            current_emotion = row['label']
            rel_path = row['pth']
            target_emotion = target_manager.get_target_emotion(current_emotion)
            
            if target_emotion is None: continue

            full_img_path = os.path.join(DATASET_ROOT, rel_path)
            
            # --- GENERATE ---
            status, generated_image, original_image = call_gemini_with_backoff(full_img_path, current_emotion, target_emotion)

            if generated_image:
                # --- HALLUCINATION CHECK ---
                is_valid, reason = guard.validate(original_image, generated_image, target_emotion)
                
                if is_valid:
                    # Save Logic
                    original_stem = Path(rel_path).stem
                    filename = f"aug_{original_stem}_{current_emotion}_to_{target_emotion}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, f"aug_{target_emotion}", filename)
                    generated_image.save(save_path)

                    results.append({
                        'original_path': rel_path,
                        'original_label': current_emotion,
                        'augmented_label': target_emotion,
                        'augmented_path': save_path,
                        'augmentation_type': 'generative_balance'   
                    })

                    success_count_session += 1
                    if success_count_session % 10 == 0:
                        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                    time.sleep(1)
                else:
                    # Filter Rejected - Refund Slot
                    print(f"   REJECTED ({target_emotion}): {reason}")
                    target_manager.counts[target_emotion] -= 1
            else:
                target_manager.counts[target_emotion] -= 1

        except Exception as e:
            if target_emotion: target_manager.counts[target_emotion] -= 1
            continue

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'augmented_labels.csv'), index=False)
    
    shutil.make_archive('/kaggle/working/affectnet_augmented', 'zip', OUTPUT_DIR)
    print("Zipping complete.")
    return FileLink('affectnet_augmented.zip')

if __name__ == "__main__":
    main()