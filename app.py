import streamlit as st
from PIL import Image
from detection.yolo_detect import detect_image_pil

st.set_page_config(
    page_title="SmartVision AI",
    layout="wide",
    page_icon="🧠"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Image Classification", "Object Detection", "Model Comparison", "About"]
)

def home_page():
    st.title("SmartVision AI")
    st.markdown("""
SmartVision AI is a **deployment-grade computer vision system**.

### Features
- Image Classification (ImageNet CNNs)
- Object Detection (YOLOv8 – ONNX Runtime)
- CPU-only, Streamlit-Cloud safe
""")

def classification_page():
    st.title("Image Classification")
    st.info(
        "Image Classification requires PyTorch, which is not available on Streamlit Cloud.\n\n"
        "✔ Works locally\n"
        "✔ Object Detection works fully in the cloud"
    )

def detection_page():
    st.title("Object Detection – YOLOv8 (ONNX)")

    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image", width=700)

        with st.spinner("Running object detection..."):
            detections, vis = detect_image_pil(img, conf)

        st.image(vis, caption="Detection Output", width=700)

        st.subheader("Detection Results")

        if not detections:
            st.write("No objects detected.")
        else:
            for d in detections:
                st.write(f"**{d['class_name']}** — {d['confidence']:.2f}")

def comparison_page():
    st.title("Model Comparison")
    st.markdown("""
### Classification Models
- VGG16
- ResNet50
- MobileNetV2
- EfficientNetB0

### Detection Model
- YOLOv8 (ONNX Runtime, CPU)
""")

def about_page():
    st.title("About SmartVision AI")
    st.markdown("""
SmartVision AI is designed for **real-world deployment**, not demos.

- ONNX inference
- No OpenCV
- Cloud-safe architecture
""")

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
