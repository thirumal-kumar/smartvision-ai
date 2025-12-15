# app.py

import streamlit as st
from PIL import Image
import os

from classification.models_loader import predict_topk
from detection.yolo_detect import get_model, detect_image_pil
from utils.viz import np_bgr_to_bytes

st.set_page_config(page_title="SmartVision AI", layout="wide")

APP_TITLE = "SmartVision AI – Intelligent Multi-Class Object Recognition System"

COCO_25_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","truck",
    "traffic light","stop sign","bench","dog","cat","horse","bird","cow","elephant",
    "bottle","cup","bowl","pizza","cake","chair","couch","bed","potted plant"
]

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Home", "Classification", "Object Detection", "Model Comparison", "About"]
    )

# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
def home_page():
    st.title(APP_TITLE)
    st.markdown("""
    **SmartVision AI** is a modular computer vision system supporting  
    image classification and real-time object detection using modern deep learning models.
    """)

    st.subheader("Supported Detection Classes")
    st.write(COCO_25_CLASSES)

# -----------------------------------------------------------------------------
# CLASSIFICATION
# -----------------------------------------------------------------------------
def classification_page():
    st.header("Image Classification")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image", use_column_width=True)

        models = ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]
        cols = st.columns(2)

        for i, name in enumerate(models):
            with cols[i % 2]:
                st.subheader(name)
                with st.spinner(f"Running {name}…"):
                    try:
                        preds = predict_topk(img, name, k=5)
                        for label, score in preds:
                            st.write(f"{label} — {score:.3f}")
                    except Exception as e:
                        st.error(str(e))
    else:
        st.info("Upload an image to begin classification.")

# -----------------------------------------------------------------------------
# DETECTION
# -----------------------------------------------------------------------------
def detection_page():
    st.header("Object Detection – YOLOv8")

    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="det"
    )

    conf = st.sidebar.slider("Confidence", 0.10, 0.90, 0.25)
    iou = st.sidebar.slider("IoU", 0.10, 0.70, 0.45)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image", width=450)

        with st.spinner("Loading YOLO model…"):
            model = get_model()

        with st.spinner("Running detection…"):
            img_bgr, detections = detect_image_pil(
                img, model=model, conf=conf, iou=iou
            )

        st.subheader("Detection Result")
        st.image(np_bgr_to_bytes(img_bgr))

        if not detections:
            st.warning("No objects detected.")
        else:
            for d in detections:
                st.write(f"{d['class']} — {d['conf']:.2f} — {d['xyxy']}")
    else:
        st.info("Upload an image to run detection.")

# -----------------------------------------------------------------------------
# MODEL COMPARISON
# -----------------------------------------------------------------------------
def model_comparison_page():
    st.header("Model Comparison")
    st.markdown("""
    - **VGG16** – high accuracy, heavy model  
    - **ResNet50** – balanced and robust  
    - **MobileNetV2** – lightweight, fast  
    - **EfficientNetB0** – optimal scaling  
    - **YOLOv8** – real-time object detection
    """)

# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------
def about_page():
    st.header("About SmartVision AI")
    st.markdown("""
    SmartVision AI is designed for **research, education, and rapid prototyping**  
    with a clean, modular architecture suitable for deployment.
    """)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    page = sidebar()

    if page == "Home":
        home_page()
    elif page == "Classification":
        classification_page()
    elif page == "Object Detection":
        detection_page()
    elif page == "Model Comparison":
        model_comparison_page()
    elif page == "About":
        about_page()


if __name__ == "__main__":
    main()
