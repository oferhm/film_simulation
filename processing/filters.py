"""
processing/filters.py

Film-stock filter implementations.
All functions: (image: np.ndarray) -> np.ndarray  (float32 0-1 BGR in/out)
Dispatch is done via apply_filter_by_path().
"""

import os
import cv2
import numpy as np


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _s_curve(image: np.ndarray, strength: float = 1.1) -> np.ndarray:
    """Apply an S-curve for contrast enhancement (float [0-1])."""
    curve = np.linspace(0, 1, 256)
    curve = np.clip((curve - 0.5) * strength + 0.5, 0, 1)
    if image.ndim == 2:
        return np.interp(image, np.linspace(0, 1, 256), curve)
    result = image.copy()
    for i in range(min(3, image.shape[2])):
        result[:, :, i] = np.interp(result[:, :, i], np.linspace(0, 1, 256), curve)
    return result


def _ensure_bgr_float(image: np.ndarray) -> np.ndarray:
    """Guarantee the image is float32 BGR with shape (H, W, 3)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image.astype(np.float32)


def _vignette(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    max_d = np.sqrt(cx ** 2 + cy ** 2)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = 1 - (dist / max_d) * strength
    return image * mask[:, :, np.newaxis]


# ──────────────────────────────────────────────
# Film stocks
# ──────────────────────────────────────────────

def kodak_portra(image: np.ndarray) -> np.ndarray:
    img = _ensure_bgr_float(image)
    img = _s_curve(img, 1.1)
    img[:, :, 0] *= 0.95   # blue down
    img[:, :, 2] *= 1.05   # red up
    hsv = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] *= 1.1
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(img, 0, 1)


def ektar_100(image: np.ndarray) -> np.ndarray:
    img = _ensure_bgr_float(image)
    hsv = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] *= 1.3
    hsv[:, :, 2] *= 1.1
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = _s_curve(img, 1.3)
    return np.clip(img, 0, 1)


def vintage(image: np.ndarray) -> np.ndarray:
    img = _ensure_bgr_float(image)
    img[:, :, 0] *= 0.8
    img[:, :, 1] *= 0.95
    img[:, :, 2] *= 1.1
    img = _vignette(img, 0.3)
    return np.clip(img, 0, 1)


def ilford_hp5(image: np.ndarray) -> np.ndarray:
    img = _ensure_bgr_float(image)
    gray = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
    gray = _s_curve(gray.reshape(*gray.shape, 1), 1.2)[:, :, 0]
    result = np.dstack([gray, gray, gray])
    return np.clip(result, 0, 1)


def colorplus_200(image: np.ndarray) -> np.ndarray:
    img = _ensure_bgr_float(image)
    hsv = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] *= 0.9
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img[:, :, 0] *= 0.9
    img[:, :, 2] *= 1.05
    return np.clip(img, 0, 1)


def colorplus_bright(image: np.ndarray) -> np.ndarray:
    img = colorplus_200(image)
    return np.clip(img + 0.1, 0, 1)


def kodak_gold(image: np.ndarray) -> np.ndarray:
    img = _ensure_bgr_float(image)
    img[:, :, 0] *= 0.85
    img[:, :, 1] *= 1.05
    img[:, :, 2] *= 1.15
    return np.clip(img, 0, 1)


def expired_film(image: np.ndarray) -> np.ndarray:
    img = _ensure_bgr_float(image)
    img += 0.03
    img[:, :, 0] *= 0.98
    img[:, :, 1] *= 1.02
    img = _vignette(img, 0.1)
    return np.clip(img, 0, 1)


def generic(image: np.ndarray) -> np.ndarray:
    img = _ensure_bgr_float(image)
    return np.clip(_s_curve(img, 1.05), 0, 1)


# ──────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────

_FILTER_MAP: dict[str, callable] = {
    "kodak_portra": kodak_portra,
    "ektar":        ektar_100,
    "vintage":      vintage,
    "ilford":       ilford_hp5,
    "colorplus":    colorplus_200,   # checked before "bright" variant
    "gold":         kodak_gold,
    "expired":      expired_film,
}


def apply_filter_by_path(image: np.ndarray, filter_path: str) -> np.ndarray:
    """
    Convert a uint8 BGR image, run the matching filter, return uint8 BGR.
    Falls back to generic() when no keyword matches.
    """
    name = os.path.basename(filter_path).lower()

    # Float conversion
    img_f = image.astype(np.float32) / 255.0
    if img_f.ndim == 2:
        img_f = cv2.cvtColor(img_f, cv2.COLOR_GRAY2BGR)

    # Special-case: bright variant must be checked before plain colorplus
    if "colorplus" in name and "bright" in name:
        result = colorplus_bright(img_f)
    else:
        fn = next(
            (fn for key, fn in _FILTER_MAP.items() if key in name),
            generic,
        )
        result = fn(img_f)

    return np.clip(result * 255, 0, 255).astype(np.uint8)
