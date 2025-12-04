# streamlit run streamlit_app.py

import io
from pathlib import Path
from typing import List, Tuple
import threading

import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# --- OPTIMIZATION: MediaPipe & WebRTC Imports ---
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import av
    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# -----------------------------
# 1. CONSTANTS AND LABELS
# -----------------------------
NUM_CLASSES = 8
EMOTION_LABELS = [
    "Angry",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# -----------------------------
# 2. MODEL LOADING
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Weights not found at {weights_path}.")
    
    model.to(device)
    model.eval()
    return model, device

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# 3. FACE DETECTION + INFERENCE
# -----------------------------

# --- OPTIMIZATION: Helper to get detector ---
def get_face_detector():
    """Returns a MediaPipe face detector if available, else None."""
    if HAS_MEDIAPIPE:
        return mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    return None

# --- OPTIMIZATION: Updated detect_faces to use MediaPipe or Haar ---
def detect_faces(image_np, detector=None):
    """
    Detects faces using MediaPipe (fast) or Haar (fallback).
    Returns list of (x, y, w, h).
    """
    h, w, _ = image_np.shape
    faces = []

    # 1. Try MediaPipe (Fastest)
    if detector:
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = detector.process(image_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                w_box = int(bboxC.width * w)
                h_box = int(bboxC.height * h)
                # Sanity check bounds
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w_box, h_box))
        return faces

    # 2. Fallback to Haar Cascades (Slower)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces_haar = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
    )
    return faces_haar

def predict_emotions(
    frame_bgr: np.ndarray, model: torch.nn.Module, device, detector, conf_floor: float
) -> Tuple[np.ndarray, List[dict]]:
    annotated = frame_bgr.copy()
    results = []
    
    # Use the unified detection logic
    faces = detect_faces(frame_bgr, detector)

    for (x, y, w, h) in faces:
        if w < 10 or h < 10: continue

        face_roi = frame_bgr[y : y + h, x : x + w]
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        img_t = transform(rgb_face).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_t)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)

        conf_val = float(conf.item())
        if conf_val < conf_floor:
            continue

        label = EMOTION_LABELS[int(idx.item())]
        results.append({
            "box": (x, y, w, h),
            "emotion": label,
            "confidence": conf_val,
            "probs": probs.cpu().numpy(),
        })

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} ({conf_val*100:.1f}%)",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

    return annotated, results

# -----------------------------
# 4. OPTIMIZED VIDEO PROCESSOR
# -----------------------------
if HAS_WEBRTC:
    class SagaEmotionProcessor(VideoTransformerBase):
        def __init__(self):
            self.model = None
            self.device = None
            self.detector = None
            self.conf_floor = 0.35
            
            # Optimization: Thread safety and Frame Skipping
            self.lock = threading.Lock()
            self.last_results = []
            self.frame_count = 0
            self.skip_rate = 5  # Inference every 5th frame
            
        def update_config(self, model, device, detector, conf_floor):
            """Pass Streamlit configuration into the processor"""
            self.model = model
            self.device = device
            self.detector = detector
            self.conf_floor = conf_floor

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # 1. INFERENCE STEP (Throttled)
            # Only run heavy model if frame_count % skip_rate == 0
            if self.frame_count % self.skip_rate == 0 and self.model is not None:
                new_faces = detect_faces(img, self.detector)
                current_results = []
                
                for (x, y, w, h) in new_faces:
                    # ROI extraction
                    if w < 5 or h < 5: continue
                    face_roi = img[y : y + h, x : x + w]
                    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess
                    img_t = transform(rgb_face).unsqueeze(0).to(self.device)
                    
                    # Inference
                    with torch.no_grad():
                        logits = self.model(img_t)
                        probs = torch.softmax(logits, dim=1).squeeze(0)
                        conf, idx = torch.max(probs, dim=0)
                    
                    conf_val = float(conf.item())
                    if conf_val >= self.conf_floor:
                        current_results.append({
                            "box": (x, y, w, h),
                            "label": EMOTION_LABELS[int(idx.item())],
                            "conf": conf_val
                        })

                # Update shared results safely
                with self.lock:
                    self.last_results = current_results

            self.frame_count += 1
            
            # 2. RENDERING STEP (Every Frame)
            # Draw the *last known* results on the *current* frame
            annotated = img.copy()
            with self.lock:
                for res in self.last_results:
                    x, y, w, h = res["box"]
                    label = res["label"]
                    conf = res["conf"]
                    
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{label} ({int(conf*100)}%)",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            return av.VideoFrame.from_ndarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), format="rgb24")

# -----------------------------
# 5. UI HELPERS
# -----------------------------
def render_metrics():
    st.markdown("#### SAGA Impact Snapshot")
    cols = st.columns(3)
    cols[0].metric("Overall Accuracy", "64.67%", "+4.50% vs Baseline")
    cols[1].metric("Contempt (Minority)", "59.33%", "+6.66%")
    cols[2].metric("Fear (Minority)", "66.67%", "+6.00%")
    st.caption(
        "Gains come from targeted generation (SAGA) plus Hallucination Guard "
        "(SSIM + CLIP + classifier-in-the-loop)."
    )

def render_project_blurb():
    st.markdown(
        """
        **Semantic Active Generative Augmentation (SAGA)** tackles long-tail scarcity by
        actively generating minority-class images, filtering them through a Hallucination
        Guard (SSIM → CLIP → ViT confidence), and retraining ViT-Base.
        """
    )
    st.markdown(
        """
        **Pipeline:** TargetBalancingManager → Gemini Generation → Hallucination Guard →
        Augmented training → Evaluation on a sealed golden test set.
        """
    )

def annotate_and_display(image_bgr: np.ndarray, model, device, detector, conf_floor: float):
    annotated, results = predict_emotions(image_bgr, model, device, detector, conf_floor)
    rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(
        rgb_image,
        caption=f"Detections ({len(results)} face(s) ≥ threshold)",
        use_container_width=True,
    )

    if len(results) == 0:
        st.info("No faces above confidence threshold.")
        return rgb_image, results

    st.markdown("#### Per-face details")
    for idx, r in enumerate(results, start=1):
        prob_pairs = [
            f"{EMOTION_LABELS[i]}: {r['probs'][i]*100:.1f}%"
            for i in np.argsort(r["probs"])[::-1][:3]
        ]
        st.write(
            f"Face {idx}: **{r['emotion']}** ({r['confidence']*100:.1f}%) | Top-3 → "
            + " · ".join(prob_pairs)
        )

    return rgb_image, results

# -----------------------------
# 6. STREAMLIT PAGE
# -----------------------------
def main():
    st.set_page_config(
        page_title="SAGA Emotion Lab",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("SAGA Emotion Lab")
    st.caption("Multi-face emotion recognition powered by ViT + SAGA augmentation.")

    sidebar = st.sidebar
    sidebar.header("Controls")
    weights_path = sidebar.text_input("Weights file", "models/vit_augmented_best_model.pth")
    conf_floor = sidebar.slider("Confidence threshold", 0.0, 1.0, 0.35, 0.05)
    show_info = sidebar.checkbox("Show SAGA summary", value=True)

    if not HAS_MEDIAPIPE:
        print("\n\n=========== ⚠️ mediapipe not installed. Using slower Haar Cascades. ===========\n\n")

    if not Path(weights_path).exists():
        st.error(f"Weights not found at '{weights_path}'. Please update the path.")
        return

    with st.spinner("Loading model..."):
        model, device = load_model(weights_path)
    
    # Initialize the Face Detector (MediaPipe or None)
    detector = get_face_detector()

    tab_realtime, tab_live, tab_upload, tab_project = st.tabs(
        ["Realtime Stream", "Camera Snapshot", "Upload Image", "Project Insights"]
    )

    # --- TAB 1: OPTIMIZED REALTIME STREAM ---
    with tab_realtime:
        st.subheader("Realtime webcam")
        st.write(
            "Stream your camera feed to see bounding boxes and emotions update live. "
            "Adjust the confidence threshold in the sidebar."
        )
        if not HAS_WEBRTC:
            st.warning(
                "Install realtime dependencies to enable this tab: "
                "`pip install streamlit-webrtc av`."
            )
        else:
            ctx = webrtc_streamer(
                key="saga-stream",
                video_transformer_factory=SagaEmotionProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_transform=True, # Critical for performance
            )
            
            # Inject configuration into the running processor
            if ctx.video_transformer:
                ctx.video_transformer.update_config(model, device, detector, conf_floor)
    
    with tab_live:
        st.subheader("Capture from camera")
        st.write("Use the camera widget to take a snapshot; all faces will be labeled with emotion + confidence.")
        snapshot = st.camera_input("Take a photo")
        if snapshot is not None:
            image = Image.open(io.BytesIO(snapshot.getvalue())).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image, _ = annotate_and_display(frame_bgr, model, device, detector, conf_floor)
            buf = io.BytesIO()
            Image.fromarray(rgb_image).save(buf, format="PNG")
            st.download_button(
                "Download annotated image",
                data=buf.getvalue(),
                file_name="annotated_snapshot.png",
                mime="image/png",
            )

    with tab_upload:
        st.subheader("Upload an image")
        uploaded = st.file_uploader("Drop a photo", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image, _ = annotate_and_display(frame_bgr, model, device, detector, conf_floor)
            buf = io.BytesIO()
            Image.fromarray(rgb_image).save(buf, format="PNG")
            st.download_button(
                "Download annotated image",
                data=buf.getvalue(),
                file_name="annotated_upload.png",
                mime="image/png",
            )

    with tab_project:
        st.subheader("SAGA in a nutshell")
        if show_info:
            render_project_blurb()
            render_metrics()
            st.markdown(
                """
                **Guard Rails:** SSIM keeps structure, CLIP enforces semantics, ViT-in-loop rejects ambiguous faces.  
                **Result:** +4.5% accuracy uplift and better minority-class recognition on a sealed golden test set.
                """
            )

if __name__ == "__main__":
    main()