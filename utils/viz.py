# utils/viz.py
import io
from PIL import Image
import numpy as np
import cv2

def pil_to_bytes(img_pil, fmt='PNG'):
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    buf.seek(0)
    return buf

def np_bgr_to_bytes(img_bgr, fmt='PNG'):
    # convert BGR (OpenCV) to RGB PIL then bytes
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return pil_to_bytes(pil, fmt=fmt)
