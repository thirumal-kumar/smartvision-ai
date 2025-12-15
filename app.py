# app.py
import streamlit as st
from PIL import Image

from classification.models_loader import predict_topk
from utils.viz import np_bgr_to_bytes

st.set_page_config(page_title="SmartVision AI", layout="wide")

APP_TITLE = "SmartVision AI – Intelligent Multi-Class Object Recognition System"

def sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Home", "Classification", "Object Detection", "About"]
    )

def home_page():
    st.title(APP_TITLE)
    st.markdown("""
    **SmartVision AI** integrates image classification and object detection  
    in a modular, deployment-safe Streamlit architecture.
    """)

def classification_page():
    st.header("Image Classification")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload an image to begin.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    models = ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]
    cols = st.columns(2)

    for i, name in enumerate(models):
        with cols[i % 2]:
            st.subheader(name)
            with st.spinner(f"Running {name}…"):
                preds = predict_topk(img, name, k=5)
                for label, score in preds:
                    st.write(f"{label} — {score:.3f}")

def detection_page():
    st.header("Object Detection – YOLOv8")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload an image to run detection.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=450)

    # 🔑 Import detection logic ONLY here
    from detection.yolo_detect import get_model, detect_image_pil

    conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.25)
    iou = st.sidebar.slider("IoU", 0.1, 0.7, 0.45)

    with st.spinner("Loading YOLO model…"):
        model = get_model()

    with st.spinner("Running detection…"):
        img_bgr, detections = detect_image_pil(
            img, model=model, conf=conf, iou=iou
        )

    st.image(np_bgr_to_bytes(img_bgr))

    if not detections:
        st.warning("No objects detected.")
    else:
        for d in detections:
            st.write(f"{d['class']} — {d['conf']:.2f} — {d['xyxy']}")

def about_page():
    st.header("About")
    st.markdown("SmartVision AI — deployment-safe computer vision demo.")

def main():
    page = sidebar()
    if page == "Home":
        home_page()
    elif page == "Classification":
        classification_page()
    elif page == "Object Detection":
        detection_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
