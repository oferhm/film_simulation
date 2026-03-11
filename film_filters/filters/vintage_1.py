import cv2
import numpy as np
import sys
import os

# Add parent directories to Python path to find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from config import input, vintage_1_output

# load image
img = cv2.imread(input)

# Check if image was loaded successfully
if img is None:
    print(f"❌ Error: Could not load '{input}'")
    exit(1)

img = img.astype(np.float32) / 255.0


# ---------- Moderate Ektar contrast curve ----------
def ektar_curve(image):

    curve = np.linspace(0, 1, 256)

    # softer S-curve than before
    curve = curve + 0.15 * (curve - 0.5)

    curve = np.clip(curve, 0, 1)

    curve = (curve * 255).astype(np.uint8)
    table = np.array(curve)

    result = cv2.LUT((image * 255).astype(np.uint8), table)

    return result.astype(np.float32) / 255.0


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
img = np.power(img, 1.02)


# ---------- Very light highlight bloom ----------
blur = cv2.GaussianBlur(img, (0, 0), 5)

mask = img > 0.85
img[mask] = img[mask] * 0.92 + blur[mask] * 0.08


# ---------- Very fine grain (Ektar is clean) ----------
grain_strength = 0.008
noise = np.random.normal(0, grain_strength, img.shape)

img = np.clip(img + noise, 0, 1)


# ---------- Save ----------
img = (img * 255).astype(np.uint8)

# Save to ouput path
cv2.imwrite(vintage_1_output, img)
