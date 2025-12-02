# streamlit run streamlit_app.py

import io
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

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
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# -----------------------------
# 3. FACE DETECTION + INFERENCE
# -----------------------------


def detect_faces(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
    )
    return faces


def predict_emotions(
    frame_bgr: np.ndarray, model: torch.nn.Module, conf_floor: float
) -> Tuple[np.ndarray, List[dict]]:
    annotated = frame_bgr.copy()
    results = []
    faces = detect_faces(frame_bgr)

    for (x, y, w, h) in faces:
        face_roi = frame_bgr[y : y + h, x : x + w]
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        img_t = transform(rgb_face).unsqueeze(0)

        with torch.no_grad():
            logits = model(img_t)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)

        conf_val = float(conf.item())
        if conf_val < conf_floor:
            continue

        label = EMOTION_LABELS[int(idx.item())]
        results.append(
            {
                "box": (x, y, w, h),
                "emotion": label,
                "confidence": conf_val,
                "probs": probs.cpu().numpy(),
            }
        )

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
# 4. UI HELPERS
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


def annotate_and_display(image_bgr: np.ndarray, model, conf_floor: float):
    annotated, results = predict_emotions(image_bgr, model, conf_floor)
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
# 5. STREAMLIT PAGE
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
    weights_path = sidebar.text_input("Weights file", "models/vit_augmented_best_model.pth") # model weights path
    conf_floor = sidebar.slider("Confidence threshold", 0.0, 1.0, 0.35, 0.05)
    show_info = sidebar.checkbox("Show SAGA summary", value=True)

    if not Path(weights_path).exists():
        st.error(f"Weights not found at '{weights_path}'. Please update the path.")
        return

    with st.spinner("Loading model..."):
        model = load_model(weights_path)

    tab_realtime, tab_live, tab_upload, tab_project = st.tabs(
        ["Realtime Stream", "Camera Snapshot", "Upload Image", "Project Insights"]
    )

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
            class EmotionTransformer(VideoTransformerBase):
                def __init__(self):
                    self.model = model
                    self.conf_floor = conf_floor

                def recv(self, frame):
                    bgr = frame.to_ndarray(format="bgr24")
                    annotated, _ = predict_emotions(bgr, self.model, self.conf_floor)
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    return av.VideoFrame.from_ndarray(rgb, format="rgb24")

            webrtc_streamer(
                key="emotion-realtime",
                video_transformer_factory=EmotionTransformer,
                media_stream_constraints={"video": True, "audio": False},
                async_transform=False,
            )

    with tab_live:
        st.subheader("Capture from camera")
        st.write("Use the camera widget to take a snapshot; all faces will be labeled with emotion + confidence.")
        snapshot = st.camera_input("Take a photo")
        if snapshot is not None:
            image = Image.open(io.BytesIO(snapshot.getvalue())).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image, _ = annotate_and_display(frame_bgr, model, conf_floor)
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
            rgb_image, _ = annotate_and_display(frame_bgr, model, conf_floor)
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
