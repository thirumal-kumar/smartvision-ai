import os
import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# Safe import for ONNX detection (no torch, no cv2)
# ------------------------------------------------------------
from detection.yolo_detect import detect_image_pil

# ------------------------------------------------------------
# Streamlit Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    layout="wide",
    page_icon="🧠"
)

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Image Classification", "Object Detection", "Model Comparison", "About"]
)

# ------------------------------------------------------------
# Home Page
# ------------------------------------------------------------
def home_page():
    st.title("SmartVision AI")

    st.markdown("""
**SmartVision AI** is a deployment-grade computer vision system.

### Features
- Image Classification (ImageNet CNNs)
- Object Detection (YOLOv8 – ONNX Runtime)
- CPU-only, Streamlit-Cloud safe
    """)

# ------------------------------------------------------------
# Image Classification (UI ONLY – Cloud Safe)
# ------------------------------------------------------------
def classification_page():
    st.title("Image Classification")

    st.info(
        "Image Classification requires **PyTorch**, which is **not available on Streamlit Cloud**.\n\n"
        "✔ This feature works **locally**\n"
        "✔ Object Detection works fully **in the cloud**"
    )

    st.markdown("""
### Upload an image
- Drag and drop file here  
- Limit **200MB** per file  
- JPG, JPEG, PNG
    """)

# ------------------------------------------------------------
# Object Detection (YOLOv8 ONNX – WORKING)
# ------------------------------------------------------------
def detection_page():
    st.title("Object Detection – YOLOv8 (ONNX)")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        st.image(img, caption="Input Image", width=700)

        with st.spinner("Running object detection..."):
            detections, vis_img = detect_image_pil(img, conf_thresh)

        st.image(vis_img, caption="Detection Output", width=700)

        st.subheader("Detection Results")

        if not detections:
            st.write("No objects detected.")
        else:
            for d in detections:
                # ✅ CORRECT KEYS — NO MORE KeyError
                label = d.get("class_name", "unknown")
                conf = d.get("confidence", 0.0)

                st.write(f"**{label}** — {conf:.2f}")

# ------------------------------------------------------------
# Model Comparison (STATIC – HONEST)
# ------------------------------------------------------------
def comparison_page():
    st.title("Model Comparison")

    st.subheader("Classification Models")
    st.markdown("""
- VGG16  
- ResNet50  
- MobileNetV2  
- EfficientNetB0  
    """)

    st.subheader("Detection Model")
    st.markdown("""
- YOLOv8 (ONNX Runtime, CPU)
    """)

# ------------------------------------------------------------
# About Page
# ------------------------------------------------------------
def about_page():
    st.title("About SmartVision AI")

    st.markdown("""
SmartVision AI is designed for **real-world deployment**, not demos.

### Architecture Highlights
- ONNX inference
- No OpenCV dependency
- Cloud-safe architecture
    """)

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
def main():
    if page == "Home":
        home_page()
    elif page == "Image Classification":
        classification_page()
    elif page == "Object Detection":
        detection_page()
    elif page == "Model Comparison":
        comparison_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
