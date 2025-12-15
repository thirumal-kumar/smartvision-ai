import os
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

from detection.yolo_detect import detect_image_pil

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    layout="wide"
)

# ------------------------------------------------------------
# ImageNet labels
# ------------------------------------------------------------
with open("imagenet_classes.txt") as f:
    IMAGENET_CLASSES = [line.strip() for line in f]

# ------------------------------------------------------------
# Classification models (lazy loaded)
# ------------------------------------------------------------
@st.cache_resource
def load_classification_models():
    return {
        "VGG16": models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).eval(),
        "ResNet50": models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        ).eval(),
        "MobileNetV2": models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        ).eval(),
        "EfficientNetB0": models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        ).eval(),
    }

CLASSIFICATION_MODELS = load_classification_models()

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# ------------------------------------------------------------
# 🔑 THIS FUNCTION WAS MISSING — RESTORED
# ------------------------------------------------------------
def classify_image(pil_image, model_name="ResNet50", topk=5):
    model = CLASSIFICATION_MODELS[model_name]
    img = _transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_idxs = probs.topk(topk)
    return [
        (IMAGENET_CLASSES[idx], float(prob))
        for idx, prob in zip(top_idxs, top_probs)
    ]

# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def home_page():
    st.title("SmartVision AI")
    st.write("SmartVision AI is a deployment-grade computer vision system.")
    st.markdown("""
**Features**
- Image Classification (ImageNet CNNs)
- Object Detection (YOLOv8 – ONNX Runtime)
- CPU-only, Streamlit-Cloud safe
""")

def classification_page():
    st.header("Image Classification")
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image", use_container_width=True)

        model_name = st.selectbox(
            "Select Model",
            list(CLASSIFICATION_MODELS.keys())
        )

        with st.spinner("Running image classification..."):
            results = classify_image(img, model_name)

        st.subheader("Top-5 Predictions (ImageNet)")
        for label, score in results:
            st.write(f"{label} — {score:.3f}")

def detection_page():
    st.header("Object Detection – YOLOv8 (ONNX)")
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    conf = st.slider(
        "Confidence Threshold",
        0.1, 0.9, 0.25
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        detections, vis_img = detect_image_pil(img, conf)
        st.image(vis_img, caption="Detection Output", use_container_width=True)

        st.subheader("Detection Results")
        for d in detections:
            st.write(f"{d['label']} — {d['confidence']:.2f}")

def comparison_page():
    st.header("Model Comparison")
    st.markdown("""
**Classification Models**
- VGG16  
- ResNet50  
- MobileNetV2  
- EfficientNetB0  

**Detection Model**
- YOLOv8 (ONNX Runtime, CPU)
""")

def about_page():
    st.header("About SmartVision AI")
    st.markdown("""
SmartVision AI is designed for real-world deployment, not demos.

- ONNX inference  
- No OpenCV dependency  
- Cloud-safe architecture
""")

# ------------------------------------------------------------
# Main navigation
# ------------------------------------------------------------
def main():
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
