import cv2
import numpy as np
import sys
import os

# Add parent directories to Python path to find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from config import input, kodak_portra_output

# load image
img = cv2.imread(input)

# Check if image was loaded successfully
if img is None:
    print(f"❌ Error: Could not load '{input}'")
    exit(1)

img = img.astype(np.float32) / 255.0


# ---------- Tone curve (Portra style) ----------
def apply_s_curve(image):
    # gentle S curve
    curve = np.linspace(0, 1, 256)
    curve = np.clip((curve - 0.5) * 1.1 + 0.5, 0, 1)

    curve = (curve * 255).astype(np.uint8)
    table = np.array(curve)

    result = cv2.LUT((image * 255).astype(np.uint8), table)
    return result.astype(np.float32) / 255.0


img = apply_s_curve(img)


# ---------- Temperature (warm tone) ----------
# increase red slightly
b, g, r = cv2.split(img)

r = np.clip(r * 1.08, 0, 1)
b = np.clip(b * 0.95, 0, 1)

img = cv2.merge([b, g, r])


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
highlight = img ** 0.95
img = np.clip(highlight, 0, 1)


# ---------- Film grain ----------
grain_strength = 0.03
noise = np.random.normal(0, grain_strength, img.shape)

img = np.clip(img + noise, 0, 1)


# ---------- Save ----------
# Ensure image is properly normalized and converted
img = np.clip(img, 0.0, 1.0)  # Ensure 0-1 range
img = (img * 255.0).astype(np.uint8)  # Convert to 0-255 uint8

# Ensure image has correct format (3 channels, uint8)
if len(img.shape) == 3 and img.shape[2] == 3:
    # Standard BGR format, ready to save
    pass
elif len(img.shape) == 3 and img.shape[2] == 4:
    # RGBA format, convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
else:
    # Grayscale, convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Final validation
img = np.clip(img, 0, 255).astype(np.uint8)

# Save to ouput path
cv2.imwrite(kodak_portra_output, img)
print(f"✅ Saved: {kodak_portra_output}")
