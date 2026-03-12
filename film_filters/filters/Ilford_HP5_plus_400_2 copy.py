import cv2
import numpy as np
import sys
import os

# Add parent directories to Python path to find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from config import input, Ilford_HP5_plus_400_2_output

# load image
img = cv2.imread(input)

# Check if image was loaded successfully
if img is None:
    print(f"❌ Error: Could not load '{input}'")
    exit(1)

img = img.astype(np.float32) / 255.0


# ---------- Convert to black and white ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------- HP5 contrast curve ----------
gray = np.clip((gray - 0.5) * 1.25 + 0.5, 0, 1)


# ---------- Slight lifted blacks ----------
gray = gray * 0.95 + 0.02


# ---------- HP5 grain ----------
grain_strength = 0.055
noise = np.random.normal(0, grain_strength, gray.shape)

gray = np.clip(gray + noise, 0, 1)


# ---------- Slight blur (film grain texture) ----------
gray = cv2.GaussianBlur(gray, (3,3), 0)


# ---------- Vignette ----------
rows, cols = gray.shape

X = cv2.getGaussianKernel(cols, cols/1.7)
Y = cv2.getGaussianKernel(rows, rows/1.7)

kernel = Y * X.T
mask = kernel / kernel.max()

gray = gray * mask


# ---------- Convert back to 3 channels ----------
result = cv2.merge([gray, gray, gray])

# ---------- Add exposure using EV ----------
ev = 0.15
img = np.clip(result * (2 ** ev), 0, 1)


# ---------- Save ----------
img = (img * 255).astype(np.uint8)

# Save to ouput path
cv2.imwrite(Ilford_HP5_plus_400_2_output, img)
