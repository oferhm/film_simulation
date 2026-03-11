import cv2
import numpy as np
import sys
import os

# Add parent directories to Python path to find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from config import input, kodak_portra_output, kodak_portra_output_1

# load image
img = cv2.imread(input)

# Check if image was loaded successfully
if img is None:
    print(f"❌ Error: Could not load '{input}'")
    exit(1)

img = img.astype(np.float32) / 255.0


# ---------- Stronger film S-curve ----------
def film_curve(image):

    curve = np.linspace(0, 1, 256)

    # stronger S curve
    curve = curve + 0.15 * (curve - 0.5)
    curve = np.clip(curve, 0, 1)

    curve = (curve * 255).astype(np.uint8)
    table = np.array(curve)

    result = cv2.LUT((image * 255).astype(np.uint8), table)
    return result.astype(np.float32) / 255.0


img = film_curve(img)


# ---------- Warm film color balance ----------
b, g, r = cv2.split(img)

r = np.clip(r * 1.12, 0, 1)
g = np.clip(g * 1.02, 0, 1)
b = np.clip(b * 0.90, 0, 1)

img = cv2.merge([b, g, r])


# ---------- Slight fade blacks ----------
img = img * 0.92 + 0.04


# ---------- Highlight compression ----------
img = np.power(img, 0.9)


# ---------- Saturation control ----------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

s = np.clip(s * 0.9, 0, 1)

hsv = cv2.merge([h, s, v])
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------- Portra green shift ----------
# reduce green saturation slightly
img[:, :, 1] *= 0.93


# ---------- Halation (film glow) ----------
blur = cv2.GaussianBlur(img, (0, 0), 8)

mask = img > 0.75
img[mask] = img[mask] * 0.85 + blur[mask] * 0.15


# ---------- Film grain ----------
grain_strength = 0.05
noise = np.random.normal(0, grain_strength, img.shape)

img = np.clip(img + noise, 0, 1)

# ---------- Save ----------
img = (img * 255).astype(np.uint8)

# Save to ouput path
cv2.imwrite(kodak_portra_output_1, img)
