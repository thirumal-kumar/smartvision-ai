import streamlit as st
from PIL import Image

from classification.models_loader import predict_topk
from detection.yolo_detect import detect_image_pil

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    layout="wide",
)

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
            "About",
        ],
    )

# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
def home_page():
    st.title("SmartVision AI")
    st.markdown(
        """
        **SmartVision AI** is a deployment-grade computer vision system.

        **Features**
        - Image Classification (ImageNet CNNs)
        - Object Detection (YOLOv8 – ONNX Runtime)
        - CPU-only, Streamlit-Cloud safe
        """
    )

# -----------------------------------------------------------------------------
# IMAGE CLASSIFICATION
# -----------------------------------------------------------------------------
def classification_page():
    st.header("Image Classification")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is None:
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    st.subheader("Top-5 Predictions (ImageNet)")

    models = ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]
    cols = st.columns(2)

    for i, model_name in enumerate(models):
        with cols[i % 2]:
            st.markdown(f"**{model_name}**")
            preds = predict_topk(img, model_name, k=5)
            for label, score in preds:
                st.write(f"{label} — {score:.3f}")

# -----------------------------------------------------------------------------
# OBJECT DETECTION (YOLOv8 ONNX)
# -----------------------------------------------------------------------------
def detection_page():
    st.header("Object Detection – YOLOv8 (ONNX)")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="detector",
    )

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
    )

    if uploaded is None:
        return

    img = Image.open(uploaded).convert("RGB")

    with st.spinner("Running YOLOv8 inference…"):
        detections, vis_img = detect_image_pil(img, conf_thresh)

    st.image(
        vis_img,
        caption="Detection Output",
        use_column_width=True,
    )

    st.subheader("Detection Results")

    if not detections:
        st.warning("No objects detected.")
        return

    for d in detections:
        label = d.get("class", "unknown")
        conf = d.get("confidence", 0.0)
        st.write(f"**{label}** — {conf:.2f}")

# -----------------------------------------------------------------------------
# MODEL COMPARISON
# -----------------------------------------------------------------------------
def model_comparison_page():
    st.header("Model Comparison")
    st.markdown(
        """
        **Classification Models**
        - VGG16
        - ResNet50
        - MobileNetV2
        - EfficientNetB0

        **Detection Model**
        - YOLOv8 (ONNX Runtime, CPU)
        """
    )

# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------
def about_page():
    st.header("About SmartVision AI")
    st.markdown(
        """
        SmartVision AI is designed for **real-world deployment**, not demos.

        - ONNX inference
        - No OpenCV dependency
        - Cloud-safe architecture
        """
    )

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
