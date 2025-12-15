# app.py
import streamlit as st
from PIL import Image

from classification.models_loader import predict_topk
from detection.yolo_detect import detect_image_pil

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    layout="wide"
)

APP_TITLE = "SmartVision AI – Intelligent Vision System"

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Image Classification",
            "Object Detection (YOLOv8 ONNX)",
            "Model Comparison",
            "About"
        ]
    )

# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
def home_page():
    st.title(APP_TITLE)
    st.markdown("""
    **SmartVision AI** is a cloud-safe computer vision application demonstrating:

    - 🧠 Image classification using pretrained CNNs  
    - 🎯 Object detection using **YOLOv8 (ONNX Runtime, CPU-only)**  
    - ☁️ Deployment-ready architecture (no OpenCV / no Torch at runtime)  

    This design ensures reliable execution on platforms such as **Streamlit Cloud**.
    """)

# -----------------------------------------------------------------------------
# IMAGE CLASSIFICATION
# -----------------------------------------------------------------------------
def classification_page():
    st.header("Image Classification")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is None:
        st.info("Upload an image to run classification.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    st.markdown("### Top-5 Predictions (ImageNet)")

    model_names = [
        "vgg16",
        "resnet50",
        "mobilenetv2",
        "efficientnetb0"
    ]

    cols = st.columns(2)

    for i, model_name in enumerate(model_names):
        with cols[i % 2]:
            st.subheader(model_name)
            with st.spinner(f"Running {model_name}…"):
                try:
                    preds = predict_topk(img, model_name, k=5)
                    for label, score in preds:
                        st.write(f"{label} — {score:.3f}")
                except Exception as e:
                    st.error(str(e))

# -----------------------------------------------------------------------------
# OBJECT DETECTION (YOLOv8 ONNX)
# -----------------------------------------------------------------------------
def detection_page():
    st.header("Object Detection – YOLOv8 (ONNX, CPU)")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="det"
    )

    conf = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    if uploaded is None:
        st.info("Upload an image to run object detection.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", width=450)

    with st.spinner("Running YOLOv8 ONNX inference…"):
        detections = detect_image_pil(img, conf=conf)

    st.subheader("Detection Results")

    if not detections:
        st.warning("No objects detected above the confidence threshold.")
    else:
        for d in detections:
            st.write(
                f"{d['class']} | "
                f"Confidence: {d['confidence']:.2f} | "
                f"Bounding Box: {d['bbox']}"
            )

# -----------------------------------------------------------------------------
# MODEL COMPARISON
# -----------------------------------------------------------------------------
def model_comparison_page():
    st.header("Model Comparison")

    st.markdown("""
    ### Classification Models
    - **VGG16** – deep and accurate, higher compute cost  
    - **ResNet50** – balanced performance and robustness  
    - **MobileNetV2** – lightweight and fast  
    - **EfficientNetB0** – optimal scaling strategy  

    ### Object Detection
    - **YOLOv8 (ONNX Runtime)**  
      - CPU-only inference  
      - No native GUI dependencies  
      - Cloud-safe deployment  
    """)

# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------
def about_page():
    st.header("About SmartVision AI")

    st.markdown("""
    SmartVision AI demonstrates **deployment-grade computer vision pipelines**.

    **Key design decisions:**
    - ONNX Runtime for object detection stability
    - Modular architecture
    - Streamlit Cloud compatibility
    - Separation of inference and UI layers

    This project is suitable for:
    - Research demos  
    - Teaching computer vision  
    - Portfolio and prototype deployments  
    """)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    page = sidebar()

    if page == "Home":
        home_page()
    elif page == "Image Classification":
        classification_page()
    elif page == "Object Detection (YOLOv8 ONNX)":
        detection_page()
    elif page == "Model Comparison":
        model_comparison_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
