# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from classification.models_loader import predict_topk
from detection.yolo_detect import detect_image_pil

# -----------------------------------------------------------------------------
# COCO LABELS (YOLOv8)
# -----------------------------------------------------------------------------
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SmartVision AI", layout="wide")
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
    **SmartVision AI** demonstrates deployment-grade vision pipelines.

    - Image Classification (CNNs)
    - Object Detection (YOLOv8 – ONNX Runtime)
    - CPU-only, cloud-safe deployment
    """)

# -----------------------------------------------------------------------------
# IMAGE CLASSIFICATION
# -----------------------------------------------------------------------------
def classification_page():
    st.header("Image Classification")

    uploaded = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    st.markdown("### Top-5 Predictions (ImageNet)")
    models = ["vgg16", "resnet50", "mobilenetv2", "efficientnetb0"]
    cols = st.columns(2)

    for i, m in enumerate(models):
        with cols[i % 2]:
            st.subheader(m)
            preds = predict_topk(img, m, k=5)
            for label, score in preds:
                st.write(f"{label} — {score:.3f}")

# -----------------------------------------------------------------------------
# OBJECT DETECTION (YOLOv8 ONNX)
# -----------------------------------------------------------------------------
def detection_page():
    st.header("Object Detection – YOLOv8 (ONNX, CPU)")

    uploaded = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"], key="det")
    conf_thres = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)

    if uploaded is None:
        return

    img = Image.open(uploaded).convert("RGB")
    draw = ImageDraw.Draw(img)

    detections = detect_image_pil(img, conf=conf_thres)

    for d in detections:
        cls_id = int(d.get("class", -1))
        conf = d.get("confidence", d.get("conf", 0.0))
        bbox = d.get("bbox", [])

        if len(bbox) != 4 or cls_id < 0:
            continue

        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
        x1, y1, x2, y2 = map(int, bbox)

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Draw label
        text = f"{label} {conf:.2f}"
        draw.text((x1 + 4, y1 + 4), text, fill="red")

    st.image(img, caption="Detection Output", use_column_width=True)

    st.subheader("Detection Results")
    if not detections:
        st.warning("No objects detected.")
    else:
        for d in detections:
            cls_id = int(d.get("class", -1))
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
            conf = d.get("confidence", d.get("conf", 0.0))
            st.write(f"{label} — {conf:.2f}")

# -----------------------------------------------------------------------------
# MODEL COMPARISON
# -----------------------------------------------------------------------------
def model_comparison_page():
    st.header("Model Comparison")
    st.markdown("""
    **Classification:** VGG16, ResNet50, MobileNetV2, EfficientNetB0  
    **Detection:** YOLOv8 (ONNX Runtime, CPU-only)
    """)

# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------
def about_page():
    st.header("About")
    st.markdown("""
    SmartVision AI is designed for **real cloud deployment**, not demos.

    - ONNX inference
    - No OpenCV dependency
    - Streamlit-Cloud safe
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
