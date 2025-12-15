# detection/yolo_detect.py
import os
import numpy as np
import cv2
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')

def get_model():
    # Ultralytics will download yolov8n.pt automatically if not present.
    model = YOLO('yolov8n.pt')
    return model

def detect_image_pil(img_pil, model=None, conf=0.25, iou=0.45):
    """
    img_pil: PIL Image
    model: YOLO model (ultralytics)
    returns: image (BGR, numpy) with boxes drawn, list of detections
    """
    if model is None:
        model = get_model()
    # Convert PIL to numpy BGR
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    results = model(img, conf=conf, iou=iou, verbose=False)
    # take first result
    r = results[0]
    boxes_out = []
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = float(box.conf[0].tolist())
            cls_id = int(box.cls[0].tolist())
            cls_name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
            boxes_out.append({'xyxy': (x1,y1,x2,y2), 'conf': conf_score, 'class': cls_name})
            # draw
            label = f"{cls_name} {conf_score:.2f}"
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0,255,0), -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img, boxes_out
