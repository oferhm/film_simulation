import cv2
import numpy as np
import sys
import os

# Add parent directories to Python path to find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from config import input, kodak_gold_200_output

# load image
img = cv2.imread(input)

# Check if image was loaded successfully
if img is None:
    print(f"❌ Error: Could not load '{input}'")
    exit(1)

img = img.astype(np.float32) / 255.0


# ---------- Slight exposure boost ----------
img = np.clip(img * 1.05, 0, 1)


# ---------- Medium contrast (S-curve) ----------
img = np.clip((img - 0.5) * 1.15 + 0.5, 0, 1)


# ---------- Kodak warm tone ----------
b, g, r = cv2.split(img)

r = np.clip(r * 1.12, 0, 1)   # stronger reds
g = np.clip(g * 1.03, 0, 1)
b = np.clip(b * 0.90, 0, 1)   # reduce blue

img = cv2.merge([b, g, r])


# ---------- Slight cyan shadows ----------
shadow_mask = img < 0.35

img[:,:,0][shadow_mask[:,:,0]] *= 1.08
img[:,:,1][shadow_mask[:,:,1]] *= 1.05


# ---------- Saturation boost ----------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)
s = np.clip(s * 1.15, 0, 1)

hsv = cv2.merge([h, s, v])
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------- Muted greens (Kodak palette) ----------
img[:,:,1] *= 0.95


# ---------- Slight faded blacks ----------
img = img * 0.96 + 0.02


# ---------- Grain ----------
grain_strength = 0.04
noise = np.random.normal(0, grain_strength, img.shape)

img = np.clip(img + noise, 0, 1)


# ---------- Soft vignette ----------
rows, cols = img.shape[:2]

X = cv2.getGaussianKernel(cols, cols/1.2)
Y = cv2.getGaussianKernel(rows, rows/1.2)

kernel = Y * X.T
mask = kernel / kernel.max()

for i in range(3):
    img[:,:,i] *= mask


# ---------- Save ----------
img = (img * 255).astype(np.uint8)

# Save to ouput path
cv2.imwrite(kodak_gold_200_output, img)
