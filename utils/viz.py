# utils/viz.py

import io
import numpy as np
from PIL import Image

def np_bgr_to_bytes(img_bgr: np.ndarray) -> bytes:
    """
    Convert a BGR NumPy image to PNG bytes for Streamlit display.
    Safe for Streamlit Cloud (lazy cv2 import).
    """
    import cv2  # 🔑 LAZY IMPORT — DO NOT MOVE

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()
