# detection/yolo_detect.py
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

MODEL_PATH = "detection/yolov8n.onnx"
IMG_SIZE = 640

@st.cache_resource(show_spinner=False)
def load_session():
    return ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

def preprocess(img_pil: Image.Image):
    img = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)    # add batch
    return img

def detect_image_pil(img_pil, conf=0.25):
    session = load_session()
    inp = preprocess(img_pil)

    outputs = session.run(None, {"images": inp})[0]

    detections = []
    for det in outputs[0]:
        score = det[4]
        if score >= conf:
            detections.append({
                "xyxy": det[:4].tolist(),
                "conf": float(score),
                "class": int(det[5])
            })

    return detections
