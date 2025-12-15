import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# =============================================================================
# COCO CLASS NAMES (YOLOv8 official order)
# =============================================================================
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

MODEL_PATH = "yolov8n.onnx"
IMG_SIZE = 640

# =============================================================================
# ONNX SESSION (Cloud Safe)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_session():
    return ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

# =============================================================================
# PREPROCESS
# =============================================================================
def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

# =============================================================================
# NMS
# =============================================================================
def nms(boxes, scores, iou_thresh=0.45):
    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * \
                 (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])

        iou = inter / (area_i + area_j - inter + 1e-6)
        idxs = idxs[1:][iou < iou_thresh]

    return keep

# =============================================================================
# MAIN DETECTION
# =============================================================================
def detect_image_pil(img: Image.Image, conf_thresh=0.25):
    session = load_session()
    orig_w, orig_h = img.size

    preds = session.run(None, {"images": preprocess(img)})[0]
    preds = np.squeeze(preds)

    if preds.shape[0] < preds.shape[1]:
        preds = preds.transpose(1, 0)

    boxes, scores, labels = [], [], []

    for p in preds:
        box = p[:4]
        class_scores = p[4:]

        cls_id = int(np.argmax(class_scores))
        score = float(class_scores[cls_id])

        if score < conf_thresh:
            continue

        label = COCO_CLASSES[cls_id]

        x, y, w, h = box
        x1 = (x - w / 2) * orig_w / IMG_SIZE
        y1 = (y - h / 2) * orig_h / IMG_SIZE
        x2 = (x + w / 2) * orig_w / IMG_SIZE
        y2 = (y + h / 2) * orig_h / IMG_SIZE

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        labels.append(label)

    if not boxes:
        return [], img

    boxes = np.array(boxes)
    scores = np.array(scores)
    keep = nms(boxes, scores)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    results = []

    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        label = labels[i]
        score = scores[i]

        text = f"{label} {score:.2f}"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill="lime")
        draw.text((x1 + 3, y1 - th - 4), text, fill="black", font=font)

        results.append({
            "class_name": label,
            "confidence": float(score),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    return results, img
