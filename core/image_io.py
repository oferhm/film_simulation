"""
core/image_io.py - Unicode-safe image loading and saving.
"""

import os
import cv2
import numpy as np
from typing import Optional


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk, handling Unicode paths correctly.
    Returns BGR numpy array or None on failure.
    """
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        if img_array.size == 0:
            return None
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[image_io] load error: {e}")
        return None


def save_image(file_path: str, image: np.ndarray) -> bool:
    """
    Save a BGR numpy array to disk, handling Unicode paths correctly.
    Returns True on success.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower() or ".jpg"
        is_success, buffer = cv2.imencode(ext, image)
        if not is_success:
            return False
        with open(file_path, "wb") as f:
            f.write(buffer.tobytes())
        return True
    except Exception as e:
        print(f"[image_io] save error: {e}")
        return False


def next_available_filename(folder: str, base: str, ext: str) -> str:
    """
    Return the next non-existing filename: base_01.ext, base_02.ext, …
    Falls back to a timestamp suffix after 999 attempts.
    """
    for i in range(1, 1000):
        name = f"{base}_{i:02d}{ext}"
        if not os.path.exists(os.path.join(folder, name)):
            return name

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}{ext}"
