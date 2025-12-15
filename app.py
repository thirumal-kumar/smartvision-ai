import os
import sys
import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# Ensure project root is in PYTHONPATH (Streamlit Cloud safe)
# ------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ------------------------------------------------------------
# Import ONLY cloud-safe modules
# ------------------------------------------------------------
from detection.yolo_detect import detect_image_pil

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    layout="wide",
)

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Image Classification",
        "Object Detection",
        "Model Comparison",
        "About",
    ],
)

# ------------------------------------------------------------
# Home Page
# ------------------------------------------------------------
def home_page():
    st.title("SmartVision AI")

    st.markdown(
        """
**SmartVision AI** is a deployment-grade computer vision system.

### Features
- Image Classification (ImageNet CNNs)
- Object Detection (YOLOv8 – ONNX Runtime)
- CPU-only, Streamlit-Cloud safe
        """
    )

# ------------------------------------------------------------
# Image Classification Page (UI ONLY – no execution)
# ------------------------------------------------------------
def classification_page():
    st.title("Image Classification")

    st.warning(
        """
Image Classification requires **PyTorch**, which is **not available**
in the Streamlit Cloud runtime.

✔ This feature works **locally**  
✔ Object Detection works **fully in the cloud**
        """
    )

    st.markdown(
        """
Upload an image

- Drag and drop file here
- Limit 200MB per file
- JPG, JPEG, PNG
        """
    )

# ------------------------------------------------------------
# Object Detection Page (YOLOv8 ONNX)
# ------------------------------------------------------------
def detection_page():
    st.title("Object Detection – YOLOv8 (ONNX)")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input Image", use_container_width=True)

        with st.spinner("Running object detection..."):
            detections, vis_img = detect_image_pil(img, conf_thresh)

        st.image(vis_img, caption="Detection Output", use_container_width=True)

        st.subheader("Detection Results")
        if not detections:
            st.write("No objects detected.")
        else:
            for d in detections:
                st.write(f"{d['label']} — {d['confidence']:.2f}")

# ------------------------------------------------------------
# Model Comparison Page
# ------------------------------------------------------------
def comparison_page():
    st.title("Model Comparison")

    st.markdown(
        """
### Classification Models
- VGG16
- ResNet50
- MobileNetV2
- EfficientNetB0

### Detection Model
- YOLOv8 (ONNX Runtime, CPU)
        """
    )

# ------------------------------------------------------------
# About Page
# ------------------------------------------------------------
def about_page():
    st.title("About SmartVision AI")

    st.markdown(
        """
SmartVision AI is designed for **real-world deployment**, not demos.

### Architecture Highlights
- ONNX inference
- No OpenCV dependency
- Cloud-safe architecture
        """
    )

# ------------------------------------------------------------
# Main Router
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

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
