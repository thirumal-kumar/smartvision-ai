# detection/yolo_detect.py
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_model():
    """
    Lazy-load YOLO only AFTER Streamlit runtime starts.
    This avoids ultralytics -> cv2 import crash.
    """
    from ultralytics import YOLO  # 🔑 MUST BE INSIDE FUNCTION
    return YOLO("yolov8n.pt")


def detect_image_pil(img_pil, model=None, conf=0.25, iou=0.45):
    """
    Run YOLOv8 detection on a PIL image.
    Returns BGR image + detection metadata.
    """
    import cv2  # 🔑 lazy OpenCV import

    if model is None:
        model = get_model()

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    results = model(img, conf=conf, iou=iou, verbose=False)
    r = results[0]

    detections = []

    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id))

            detections.append({
                "xyxy": (x1, y1, x2, y2),
                "conf": score,
                "class": cls_name
            })

            label = f"{cls_name} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1
            )
            cv2.putText(
                img, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

    return img, detections
