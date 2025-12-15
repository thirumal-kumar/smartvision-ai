import streamlit as st
from PIL import Image

# ============================================================
# SAFE IMPORTS (Cloud + Local compatible)
# ============================================================

try:
    # Preferred import (if folder is named 'classification')
    from classification.classifier import classify_image
except ModuleNotFoundError:
    # Fallback import (if folder is named 'classifiers')
    from classifiers.classifier import classify_image

from detection.yolo_detect import detect_image_pil


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Image Classification",
        "Object Detection",
        "Model Comparison",
        "About"
    ]
)

# ============================================================
# HOME
# ============================================================

def home_page():
    st.title("SmartVision AI")

    st.markdown(
        """
        **SmartVision AI** is a **deployment-grade computer vision system**.
        """
    )

    st.markdown("### Features")

    st.markdown(
        """
        • Image Classification (ImageNet CNNs)  
        • Object Detection (YOLOv8 – ONNX Runtime)  
        • CPU-only, Streamlit-Cloud safe  
        """
    )

    st.info(
        "This project focuses on **robust inference pipelines**, "
        "cloud compatibility, and real-world deployment."
    )

# ============================================================
# IMAGE CLASSIFICATION
# ============================================================

def classification_page():
    st.title("Image Classification")

    st.markdown("Upload an image to classify it using ImageNet-trained models.")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input Image", use_container_width=True)

        with st.spinner("Running image classification..."):
            results = classify_image(img)

        st.subheader("Top-5 Predictions (ImageNet)")

        cols = st.columns(len(results))
        for col, (model_name, preds) in zip(cols, results.items()):
            with col:
                st.markdown(f"**{model_name}**")
                for label, score in preds:
                    st.write(f"{label} — {score:.3f}")

# ============================================================
# OBJECT DETECTION
# ============================================================

def detection_page():
    st.title("Object Detection")

    st.markdown(
        "Object detection using **YOLOv8 (ONNX Runtime, CPU-only)**."
    )

    conf_thresh = st.slider(
        "Confidence Threshold",
        0.05,
        0.95,
        0.25,
        0.05
    )

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Running object detection..."):
            detections, vis_img = detect_image_pil(img, conf_thresh)

        st.image(
            vis_img,
            caption="Detection Output",
            use_container_width=True
        )

        st.subheader("Detection Results")

        if not detections:
            st.warning("No objects detected above the confidence threshold.")
        else:
            for d in detections:
                st.write(f"**{d['class']}** — {d['confidence']:.2f}")

# ============================================================
# MODEL COMPARISON
# ============================================================

def model_comparison_page():
    st.title("Model Comparison")

    st.markdown("### Classification Models")
    st.markdown(
        """
        • VGG16  
        • ResNet50  
        • MobileNetV2  
        • EfficientNetB0  
        """
    )

    st.markdown("### Detection Model")
    st.markdown("• YOLOv8 (ONNX Runtime, CPU-only)")

# ============================================================
# ABOUT
# ============================================================

def about_page():
    st.title("About SmartVision AI")

    st.markdown(
        """
        **SmartVision AI** is designed for **real-world deployment**, not demos.

        • ONNX inference  
        • No OpenCV dependency  
        • Cloud-safe architecture  
        • CPU-only execution  
        """
    )

# ============================================================
# ROUTER
# ============================================================

def main():
    if page == "Home":
        home_page()
    elif page == "Image Classification":
        classification_page()
    elif page == "Object Detection":
        detection_page()
    elif page == "Model Comparison":
        model_comparison_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
