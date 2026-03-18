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


def kodak_portra_400_2(image: np.ndarray) -> np.ndarray:
    """Kodak Portra 400 variant 2 - focused on saturation and grain."""
    img = _ensure_bgr_float(image)
    
    # ---------- Saturation ----------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 0.95, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # ---------- Shadow lift ----------
    shadow_lift = 0.05
    img = np.clip(img + shadow_lift * (1 - img), 0, 1)
    
    # ---------- Highlight compression ----------
    img = np.power(img, 0.95)
    
    # ---------- Film grain ----------
    grain_strength = 0.03
    noise = np.random.normal(0, grain_strength, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    return np.clip(img, 0, 1)


def ektar_100(image: np.ndarray) -> np.ndarray:
    """Kodak Ektar 100 film emulation with detailed processing."""
    
    def ektar_curve(img):
        """Moderate Ektar contrast curve."""
        curve = np.linspace(0, 1, 256)
        # softer S-curve than before
        curve = curve + 0.15 * (curve - 0.5)
        curve = np.clip(curve, 0, 1)
        curve = (curve * 255).astype(np.uint8)
        table = np.array(curve)
        result = cv2.LUT((img * 255).astype(np.uint8), table)
        return result.astype(np.float32) / 255.0
    
    img = _ensure_bgr_float(image)
    
    # ---------- Moderate Ektar contrast curve ----------
    img = ektar_curve(img)
    
    # ---------- Slight color shift ----------
    b, g, r = cv2.split(img)
    b = np.clip(b * 1.05, 0, 1)  # subtle blue boost
    g = np.clip(g * 1.01, 0, 1)
    r = np.clip(r * 1.05, 0, 1)  # subtle red boost
    img = cv2.merge([b, g, r])
    
    # ---------- Mild saturation boost ----------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.12, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # ---------- Slight shadow depth ----------
    img = np.power(img, 1.1)
    
    # ---------- Very light highlight bloom ----------
    blur = cv2.GaussianBlur(img, (0, 0), 5)
    mask = img > 0.85
    img[mask] = img[mask] * 0.92 + blur[mask] * 0.08
    
    # ---------- Very fine grain (Ektar is clean) ----------
    grain_strength = 0.017
    noise = np.random.normal(0, grain_strength, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    # ---------- Shadow lift ----------
    shadow_lift = 0.07
    img = np.clip(img + shadow_lift * (1 - img), 0, 1)
    
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


def film_0(image: np.ndarray) -> np.ndarray:
    """Clean digital film look with subtle warmth."""
    
    def soft_curve(img):
        curve = np.linspace(0, 1, 256)
        curve = curve + 0.1 * (curve - 0.5)
        curve = np.clip(curve, 0, 1)
        curve = (curve * 255).astype(np.uint8)
        table = np.array(curve)
        result = cv2.LUT((img * 255).astype(np.uint8), table)
        return result.astype(np.float32) / 255.0
    
    img = _ensure_bgr_float(image)
    
    # Soft contrast curve
    img = soft_curve(img)
    
    # Warm color shift
    b, g, r = cv2.split(img)
    b = np.clip(b * 0.98, 0, 1)
    g = np.clip(g * 1.02, 0, 1)
    r = np.clip(r * 1.08, 0, 1)
    img = cv2.merge([b, g, r])
    
    # Gentle saturation boost
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.08, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return np.clip(img, 0, 1)


def film_1(image: np.ndarray) -> np.ndarray:
    """Vintage film with slight desaturation and warmth."""
    
    def vintage_curve(img):
        curve = np.linspace(0, 1, 256)
        curve = curve + 0.12 * (curve - 0.5)
        curve = np.clip(curve, 0, 1)
        curve = (curve * 255).astype(np.uint8)
        table = np.array(curve)
        result = cv2.LUT((img * 255).astype(np.uint8), table)
        return result.astype(np.float32) / 255.0
    
    img = _ensure_bgr_float(image)
    
    # Vintage contrast
    img = vintage_curve(img)
    
    # Warm vintage color
    b, g, r = cv2.split(img)
    b = np.clip(b * 0.92, 0, 1)
    g = np.clip(g * 1.05, 0, 1) 
    r = np.clip(r * 1.15, 0, 1)
    img = cv2.merge([b, g, r])
    
    # Reduce saturation slightly
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 0.92, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Light vignette
    img = _vignette(img, 0.15)
    
    return np.clip(img, 0, 1)


def film_2(image: np.ndarray) -> np.ndarray:
    """Cool toned film with enhanced shadows."""
    
    def cool_curve(img):
        curve = np.linspace(0, 1, 256)
        curve = curve + 0.18 * (curve - 0.5)
        curve = np.clip(curve, 0, 1)
        curve = (curve * 255).astype(np.uint8)
        table = np.array(curve)
        result = cv2.LUT((img * 255).astype(np.uint8), table)
        return result.astype(np.float32) / 255.0
    
    img = _ensure_bgr_float(image)
    
    # Strong contrast curve
    img = cool_curve(img)
    
    # Cool color shift
    b, g, r = cv2.split(img)
    b = np.clip(b * 1.08, 0, 1)
    g = np.clip(g * 0.98, 0, 1)
    r = np.clip(r * 0.95, 0, 1)
    img = cv2.merge([b, g, r])
    
    # Enhanced saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.15, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Shadow enhancement
    img = np.power(img, 1.2)
    
    return np.clip(img, 0, 1)


def film_3_grain(image: np.ndarray) -> np.ndarray:
    """Classic film with noticeable grain texture."""
    
    def classic_curve(img):
        curve = np.linspace(0, 1, 256)
        curve = curve + 0.14 * (curve - 0.5)
        curve = np.clip(curve, 0, 1)
        curve = (curve * 255).astype(np.uint8)
        table = np.array(curve)
        result = cv2.LUT((img * 255).astype(np.uint8), table)
        return result.astype(np.float32) / 255.0
    
    img = _ensure_bgr_float(image)
    
    # Classic contrast
    img = classic_curve(img)
    
    # Balanced color
    b, g, r = cv2.split(img)
    b = np.clip(b * 1.02, 0, 1)
    g = np.clip(g * 1.01, 0, 1)
    r = np.clip(r * 1.03, 0, 1)
    img = cv2.merge([b, g, r])
    
    # Saturation boost
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.1, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Prominent grain
    grain_strength = 0.025
    noise = np.random.normal(0, grain_strength, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    return np.clip(img, 0, 1)


def film_4_grain(image: np.ndarray) -> np.ndarray:
    """High contrast film with heavy grain and lifted blacks."""
    
    def high_contrast_curve(img):
        curve = np.linspace(0, 1, 256)
        curve = curve + 0.25 * (curve - 0.5)
        curve = np.clip(curve, 0, 1)
        curve = (curve * 255).astype(np.uint8)
        table = np.array(curve)
        result = cv2.LUT((img * 255).astype(np.uint8), table)
        return result.astype(np.float32) / 255.0
    
    img = _ensure_bgr_float(image)
    
    # High contrast curve
    img = high_contrast_curve(img)
    
    # Dramatic color shift
    b, g, r = cv2.split(img)
    b = np.clip(b * 0.95, 0, 1)
    g = np.clip(g * 1.08, 0, 1)
    r = np.clip(r * 1.12, 0, 1)
    img = cv2.merge([b, g, r])
    
    # Strong saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.2, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Heavy grain
    grain_strength = 0.035
    noise = np.random.normal(0, grain_strength, img.shape)
    img = np.clip(img + noise, 0, 1) 
    
    # Black lift
    shadow_lift = 0.12
    img = np.clip(img + shadow_lift * (1 - img), 0, 1)
    
    return np.clip(img, 0, 1)


def film_5(image: np.ndarray) -> np.ndarray:
    """Dreamy film with highlight bloom and soft contrast."""
    
    def dreamy_curve(img):
        curve = np.linspace(0, 1, 256)
        curve = curve + 0.08 * (curve - 0.5)
        curve = np.clip(curve, 0, 1)
        curve = (curve * 255).astype(np.uint8)
        table = np.array(curve)
        result = cv2.LUT((img * 255).astype(np.uint8), table)
        return result.astype(np.float32) / 255.0
    
    img = _ensure_bgr_float(image)
    
    # Soft contrast
    img = dreamy_curve(img)
    
    # Warm dreamy tones
    b, g, r = cv2.split(img)
    b = np.clip(b * 0.96, 0, 1)
    g = np.clip(g * 1.03, 0, 1)
    r = np.clip(r * 1.10, 0, 1)
    img = cv2.merge([b, g, r])
    
    # Gentle saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.05, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Dreamy highlight bloom
    blur = cv2.GaussianBlur(img, (0, 0), 8)
    mask = img > 0.75
    img[mask] = img[mask] * 0.85 + blur[mask] * 0.15
    
    # Light shadow lift
    shadow_lift = 0.05
    img = np.clip(img + shadow_lift * (1 - img), 0, 1)
    
    return np.clip(img, 0, 1)


def film_6(image: np.ndarray) -> np.ndarray:
    """Punchy modern film with enhanced mid-tones."""
    
    def punchy_curve(img):
        curve = np.linspace(0, 1, 256)
        curve = curve + 0.16 * (curve - 0.5)
        curve = np.clip(curve, 0, 1)
        curve = (curve * 255).astype(np.uint8)
        table = np.array(curve)
        result = cv2.LUT((img * 255).astype(np.uint8), table)
        return result.astype(np.float32) / 255.0
    
    img = _ensure_bgr_float(image)
    
    # Punchy contrast
    img = punchy_curve(img)
    
    # Modern color grading
    b, g, r = cv2.split(img)
    b = np.clip(b * 1.05, 0, 1)
    g = np.clip(g * 1.02, 0, 1) 
    r = np.clip(r * 1.06, 0, 1)
    img = cv2.merge([b, g, r])
    
    # Vibrant saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.18, 0, 1)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Mid-tone enhancement
    img = np.power(img, 0.95)
    
    # Subtle grain
    grain_strength = 0.012
    noise = np.random.normal(0, grain_strength, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    return np.clip(img, 0, 1)


# ──────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────

_FILTER_MAP: dict[str, callable] = {
    "kodak_portra": kodak_portra,
    "portra_400_2": kodak_portra_400_2,
    "ektar":        ektar_100,
    "vintage":      vintage,
    "ilford":       ilford_hp5,
    "colorplus":    colorplus_200,   # checked before "bright" variant
    "kodak_gold":   kodak_gold,
    "gold":         kodak_gold,
    "expired":      expired_film,
    "film_0":       film_0,
    "film_1":       film_1, 
    "film_2":       film_2,
    "film_3":       film_3_grain,
    "film_4":       film_4_grain,
    "film_5":       film_5,
    "film_6":       film_6,
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
