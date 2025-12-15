# app.py
import streamlit as st
from PIL import Image
import os
import numpy as np

from classification.models_loader import predict_topk
from detection.yolo_detect import get_model, detect_image_pil
from utils.viz import np_bgr_to_bytes

# Streamlit Page Setup
st.set_page_config(page_title="SmartVision AI", layout="wide")

APP_TITLE = "SmartVision AI – Intelligent Multi-Class Object Recognition System"

COCO_25_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","truck",
    "traffic light","stop sign","bench","dog","cat","horse","bird","cow","elephant",
    "bottle","cup","bowl","pizza","cake","chair","couch","bed","potted plant"
]

# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Classification", "Object Detection", "Model Comparison", "About"]
    )
    return page

# ------------------------------------------------------------------------------
# HOME PAGE (Professional Overview)
# ------------------------------------------------------------------------------
def home_page():
    st.title(APP_TITLE)

    st.markdown("""
    SmartVision AI is a modular computer vision system capable of performing 
    **high-accuracy image classification** and **real-time object detection** using 
    state-of-the-art deep learning models.

    ### Key Capabilities
    - **Image Classification** using VGG16, ResNet50, MobileNetV2, EfficientNetB0  
    - **Object Detection** powered by YOLOv8  
    - **25-Class COCO Subset** supported for detection  
    - **Fully CPU-Compatible**, optimized for cross-platform deployment  
    - **Extensible architecture** suitable for research, prototyping, or integration into larger systems

    SmartVision AI demonstrates a unified workflow combining multiple Deep Learning 
    architectures, designed for clarity, modularity, and real-world applicability.
    """)

    st.subheader("Supported Object Classes (COCO Subset)")
    st.write(COCO_25_CLASSES)

# ------------------------------------------------------------------------------
# CLASSIFICATION PAGE
# ------------------------------------------------------------------------------
def classification_page():
    st.header("Image Classification – Pretrained CNN Architectures")

    uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded is None:
        st.info("Upload an image or use a sample image from the assets folder.")
        if st.button("Use Sample Image"):
            sample_dir = os.path.join("assets", "sample_images")
            if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) > 0:
                sample = os.path.join(sample_dir, os.listdir(sample_dir)[0])
                img = Image.open(sample).convert("RGB")
                run_classification(img)
            else:
                st.error("No sample images available.")
    else:
        img = Image.open(uploaded).convert("RGB")
        run_classification(img)


def run_classification(img_pil):
    st.image(img_pil, caption="Input image", use_column_width=True)

    st.markdown("""
    **Predictions – ImageNet Top-5 (PyTorch Pretrained Models)**  
    The classification module evaluates the input using four widely used CNN architectures.
    """)

    cols = st.columns(2)
    model_names = ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]

    for i, model_name in enumerate(model_names):
        with cols[i % 2]:
            st.subheader(model_name)
            with st.spinner(f"Processing with {model_name}..."):
                try:
                    preds = predict_topk(img_pil, model_name, k=5)

                    for label, score in preds:
                        st.write(f"{label} — {score:.3f}")

                except Exception as e:
                    st.error(f"Error running {model_name}: {e}")

# ------------------------------------------------------------------------------
# OBJECT DETECTION PAGE
# ------------------------------------------------------------------------------
def detection_page():
    st.header("Object Detection – YOLOv8")

    uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="det")

    conf = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.25)
    iou = st.sidebar.slider("IoU threshold", 0.10, 0.70, 0.45)

    if uploaded is None:
        st.info("Upload an image or use one of the sample images.")
        if st.button("Use Sample Image for Detection"):
            sample_dir = os.path.join("assets", "sample_images")
            if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) > 0:
                sample = os.path.join(sample_dir, os.listdir(sample_dir)[0])
                img = Image.open(sample).convert("RGB")
                model = get_model()
                run_detection(img, model, conf, iou)
            else:
                st.error("No sample images available.")
    else:
        img = Image.open(uploaded).convert("RGB")
        model = get_model()
        run_detection(img, model, conf, iou)


def run_detection(img_pil, model, conf, iou):
    st.image(img_pil, caption="Input image", width=450)

    with st.spinner("Running YOLOv8 detection..."):
        img_bgr, detections = detect_image_pil(img_pil, model=model, conf=conf, iou=iou)

    st.subheader("Detection Results")
    st.image(np_bgr_to_bytes(img_bgr))

    if len(detections) == 0:
        st.warning("No objects detected above threshold.")
    else:
        for det in detections:
            st.write(f"{det['class']} — {det['conf']:.2f} — {det['xyxy']}")

# ------------------------------------------------------------------------------
# MODEL COMPARISON (Professional Text)
# ------------------------------------------------------------------------------
def model_comparison_page():
    st.header("Model Comparison")

    st.markdown("""
    This section outlines the characteristics and deployment relevance 
    of the architectures included in SmartVision AI.

    ### **VGG16**
    - Classical deep CNN architecture  
    - ~138M parameters  
    - Strong baseline accuracy, higher computational cost  

    ### **ResNet50**
    - Residual connections allow stable deep training  
    - Excellent generalization  
    - Balanced speed-accuracy tradeoff  

    ### **MobileNetV2**
    - Highly efficient, depthwise separable convolutions  
    - Optimized for mobile and edge hardware  
    - Ideal for low-resource or real-time deployment  

    ### **EfficientNetB0**
    - Compound scaling of depth, width, and resolution  
    - Strong accuracy with compact size  
    - Suitable for scalable production use-cases  

    ### **YOLOv8n**
    - Modern, real-time object detection architecture  
    - Performs well on CPU  
    - Suitable for surveillance, inspection, analytics, and embedded systems  

    These architectures collectively represent a spectrum from classical 
    deep CNNs to optimized lightweight and high-performance modern models.
    """)

# ------------------------------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------------------------------
def about_page():
    st.header("About SmartVision AI")

    st.markdown("""
    SmartVision AI is a unified computer vision platform that integrates 
    image classification and object detection within a modular and 
    extensible deep learning architecture.

    ### System Highlights
    - Modular codebase with clear separation of components  
    - PyTorch-based classification pipeline  
    - YOLOv8-based object detection  
    - Designed for extensibility and deployment  
    - Fully CPU-compatible  

    ### Intended Applications
    - Image analytics  
    - Automated monitoring  
    - Object detection for security and facility management  
    - Research, prototyping, and educational use  

    ### Future Enhancements
    - Model fine-tuning with domain-specific data  
    - Real-time video inference  
    - ONNX / TensorRT deployment for speed optimization  
    - Extended analytics dashboards  
    """)

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
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
