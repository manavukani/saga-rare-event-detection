# python webcam_emotion.py

import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from torchvision import transforms

# -----------------------------
# 1. LOAD TRAINED MODEL
# -----------------------------
NUM_CLASSES = 8

def load_model(weights_path):
    model = timm.create_model("vit_base_patch16_224", pretrained=True)

    # Custom MLP head must match training
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )

    # Load weights
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


# -----------------------------
# 2. TRANSFORMS (same as training)
# -----------------------------
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# -----------------------------
# 3. EMOTION LABELS
# -----------------------------
emotion_labels = [
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
# 4. REAL-TIME WEBCAM LOOP
# -----------------------------
def run_webcam(model):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not access webcam")
        return

    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            cv2.putText(
                frame,
                "No face detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
        else:
            for (x, y, w, h) in faces:
                face_roi = frame[y : y + h, x : x + w]
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                img_t = transform(rgb_face).unsqueeze(0)

                with torch.no_grad():
                    logits = model(img_t)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, dim=1)

                emotion = emotion_labels[pred.item()]
                conf_pct = conf.item() * 100

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} ({conf_pct:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Emotion Detection - Press 'q' to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# 5. MAIN
# -----------------------------
if __name__ == "__main__":
    weights = "models/vit_augmented_best_model.pth"  # model weights path
    model = load_model(weights)

    run_webcam(model)
