import streamlit as st
from PIL import Image

# -------------------- IMPORTS (UNCHANGED) --------------------
from classification.classifier import classify_image
from detection.yolo_detect import detect_image_pil

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🧠",
    layout="wide"
)

# -------------------- SIDEBAR --------------------
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
        **SmartVision AI** is a **deployment-grade computer vision system**  
        designed for real-world inference, not demos.
        """
    )

    st.markdown("### Features")

    st.markdown(
        """
        • **Image Classification** using ImageNet-trained CNNs  
        • **Object Detection** using YOLOv8 (ONNX Runtime)  
        • **CPU-only inference**, fully Streamlit-Cloud safe  
        """
    )

    st.markdown("---")

    st.info(
        "This application focuses on **robust inference pipelines**, "
        "model compatibility, and cloud-safe deployment practices."
    )

# ============================================================
# IMAGE CLASSIFICATION
# ============================================================
def classification_page():
    st.title("Image Classification")

    st.markdown(
        """
        Upload an image to classify it using multiple **ImageNet-trained CNN models**.
        """
    )

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
        """
        Perform **real-time object detection** using **YOLOv8 exported to ONNX**  
        and executed via **ONNX Runtime (CPU-only)**.
        """
    )

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
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
                st.write(
                    f"**{d['class']}** — {d['confidence']:.2f}"
                )

# ============================================================
# MODEL COMPARISON
# ============================================================
def model_comparison_page():
    st.title("Model Comparison")

    st.markdown("### Classification Models")

    st.markdown(
        """
        • **VGG16**  
        • **ResNet50**  
        • **MobileNetV2**  
        • **EfficientNetB0**
        """
    )

    st.markdown("### Detection Model")

    st.markdown(
        """
        • **YOLOv8 (ONNX Runtime, CPU-only)**
        """
    )

# ============================================================
# ABOUT
# ============================================================
def about_page():
    st.title("About SmartVision AI")

    st.markdown(
        """
        **SmartVision AI** is designed for **real-world deployment**, not demos.

        ### Core Principles
        • ONNX-based inference  
        • No OpenCV dependency  
        • Cloud-safe architecture  
        • CPU-only execution  

        The system prioritizes **stability, portability, and reproducibility**
        across local and cloud environments.
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
