"""
processing/adjustments.py

Pure functions: (image: np.ndarray, value: int) -> np.ndarray
No Qt, no side effects. All values are on a -100 … +100 scale.
"""

import cv2
import numpy as np
from core.state import EditState


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _luminance(img: np.ndarray) -> np.ndarray:
    """Return a 2-D float luminance map from a float BGR image [0-1]."""
    return 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]


# ──────────────────────────────────────────────
# Individual adjustments
# ──────────────────────────────────────────────

def apply_temp(image: np.ndarray, value: int) -> np.ndarray:
    """Warm (+) / cool (–) colour temperature."""
    if value == 0:
        return image
    img = image.astype(np.float32)
    factor = value / 100.0
    if factor > 0:
        img[:, :, 2] *= (1 + factor * 0.3)   # red up
        img[:, :, 0] *= (1 - factor * 0.2)   # blue down
    else:
        img[:, :, 0] *= (1 - factor * 0.3)   # blue up
        img[:, :, 2] *= (1 + factor * 0.2)   # red down
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_tint(image: np.ndarray, value: int) -> np.ndarray:
    """Magenta (+) / green (–) tint."""
    if value == 0:
        return image
    img = image.astype(np.float32)
    factor = value / 100.0
    if factor > 0:
        img[:, :, 1] *= (1 - factor * 0.2)   # green down
        img[:, :, 2] *= (1 + factor * 0.15)  # red up
        img[:, :, 0] *= (1 + factor * 0.15)  # blue up
    else:
        img[:, :, 1] *= (1 - factor * 0.2)   # green up
        img[:, :, 2] *= (1 + factor * 0.1)
        img[:, :, 0] *= (1 + factor * 0.1)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_exposure(image: np.ndarray, value: int) -> np.ndarray:
    """Global brightness via linear gain."""
    if value == 0:
        return image
    factor = 1.0 + value / 100.0
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_contrast(image: np.ndarray, value: int) -> np.ndarray:
    """S-curve style contrast around the midpoint."""
    if value == 0:
        return image
    img = image.astype(np.float32) / 255.0
    factor = 1.0 + value / 100.0
    img = 0.5 + factor * (img - 0.5)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def apply_highlights(image: np.ndarray, value: int) -> np.ndarray:
    """Brighten or darken the highlight range."""
    if value == 0:
        return image
    img = image.astype(np.float32) / 255.0
    lum = _luminance(img)
    mask = np.clip(lum ** 0.5, 0, 1)
    factor = value / 100.0
    for ch in range(3):
        img[:, :, ch] = np.clip(img[:, :, ch] + factor * mask, 0, 1)
    return (img * 255).astype(np.uint8)


def apply_shadows(image: np.ndarray, value: int) -> np.ndarray:
    """Lift or crush the shadow range."""
    if value == 0:
        return image
    img = image.astype(np.float32) / 255.0
    lum = _luminance(img)
    mask = np.clip((1.0 - lum) ** 2, 0, 1)
    factor = value / 100.0
    for ch in range(3):
        img[:, :, ch] = np.clip(img[:, :, ch] + factor * mask, 0, 1)
    return (img * 255).astype(np.uint8)


def apply_whites(image: np.ndarray, value: int) -> np.ndarray:
    """Adjust the white point (top 30 % of luminance)."""
    if value == 0:
        return image
    img = image.astype(np.float32) / 255.0
    lum = _luminance(img)
    mask = np.clip((lum - 0.7) / 0.3, 0, 1)
    factor = value / 100.0
    for ch in range(3):
        img[:, :, ch] = np.clip(img[:, :, ch] + factor * mask * 0.5, 0, 1)
    return (img * 255).astype(np.uint8)


def apply_blacks(image: np.ndarray, value: int) -> np.ndarray:
    """Adjust the black point (bottom 30 % of luminance)."""
    if value == 0:
        return image
    img = image.astype(np.float32) / 255.0
    lum = _luminance(img)
    mask = np.clip((0.3 - lum) / 0.3, 0, 1)
    factor = value / 100.0
    for ch in range(3):
        img[:, :, ch] = np.clip(img[:, :, ch] + factor * mask * 0.3, 0, 1)
    return (img * 255).astype(np.uint8)


def apply_texture(image: np.ndarray, value: int) -> np.ndarray:
    """
    Fine-detail sharpening (+) / smoothing (–).
    Uses a small-radius unsharp mask so only micro-texture is affected.
    """
    if value == 0:
        return image
    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
    detail = img - blurred
    factor = value / 100.0
    result = img + factor * detail * 0.6
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clarity(image: np.ndarray, value: int) -> np.ndarray:
    """Mid-tone contrast via medium-radius unsharp mask."""
    if value == 0:
        return image
    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    detail = img - blurred
    factor = value / 100.0
    result = img + factor * detail * 0.3
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_dehaze(image: np.ndarray, value: int) -> np.ndarray:
    """Boost local contrast in low-contrast (hazy) areas."""
    if value == 0:
        return image
    img = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(gray, (15, 15), 5.0)
    local_contrast = np.abs(gray - blurred)
    haze_mask = 1.0 - np.clip(local_contrast * 3, 0, 1)
    factor = value / 100.0
    for ch in range(3):
        img[:, :, ch] = np.clip(
            0.5 + (img[:, :, ch] - 0.5) * (1 + factor * haze_mask * 0.5), 0, 1
        )
    return (img * 255).astype(np.uint8)


def apply_vibrance(image: np.ndarray, value: int) -> np.ndarray:
    """Selectively saturate less-saturated colours."""
    if value == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    mask = 1.0 - sat / 255.0
    factor = value / 100.0
    hsv[:, :, 1] = np.clip(sat + factor * mask * 50, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_saturation(image: np.ndarray, value: int) -> np.ndarray:
    """Global saturation multiplier."""
    if value == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + value / 100.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

# Maps EditState field name → adjustment function
_PIPELINE: list[tuple[str, callable]] = [
    ("temp",       apply_temp),
    ("tint",       apply_tint),
    ("exposure",   apply_exposure),
    ("contrast",   apply_contrast),
    ("highlights", apply_highlights),
    ("shadows",    apply_shadows),
    ("whites",     apply_whites),
    ("blacks",     apply_blacks),
    ("texture",    apply_texture),
    ("clarity",    apply_clarity),
    ("dehaze",     apply_dehaze),
    ("vibrance",   apply_vibrance),
    ("saturation", apply_saturation),
]


def apply_all(image: np.ndarray, state: EditState) -> np.ndarray:
    """Apply every non-zero adjustment in pipeline order."""
    result = image.copy()
    for field, fn in _PIPELINE:
        value = getattr(state, field)
        if value != 0:
            result = fn(result, value)
    return result
