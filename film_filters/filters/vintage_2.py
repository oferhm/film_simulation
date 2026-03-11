import cv2
import numpy as np
import sys
import os

# Add parent directories to Python path to find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from config import input, vintage_2_output


# load image
img = cv2.imread(input)

# Check if image was loaded successfully
if img is None:
    print(f"❌ Error: Could not load '{input}'")
    exit(1)

img = img.astype(np.float32) / 255.0


# ---------- Strong contrast ----------
img = np.clip((img - 0.5) * 1.25 + 0.5, 0, 1)


# ---------- Teal shadows ----------
shadow_mask = img < 0.4

img[:,:,0][shadow_mask[:,:,0]] *= 1.15   # boost blue
img[:,:,1][shadow_mask[:,:,1]] *= 1.08   # boost green
img[:,:,2][shadow_mask[:,:,2]] *= 0.90   # reduce red


# ---------- Warm highlights ----------
highlight_mask = img > 0.6

img[:,:,2][highlight_mask[:,:,2]] *= 1.10
img[:,:,1][highlight_mask[:,:,1]] *= 1.05


# ---------- Saturation boost ----------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)

s = np.clip(s * 1.18, 0, 1)

hsv = cv2.merge([h,s,v])
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------- Slight shadow crush ----------
img = np.power(img, 1.05)


# ---------- Film grain ----------
grain_strength = 0.045
noise = np.random.normal(0, grain_strength, img.shape)

img = np.clip(img + noise, 0, 1)


# ---------- Slight vignette ----------
rows, cols = img.shape[:2]

X = cv2.getGaussianKernel(cols, cols/1.7)
Y = cv2.getGaussianKernel(rows, rows/1.7)

kernel = Y * X.T
mask = kernel / kernel.max()

for i in range(3):
    img[:,:,i] *= mask





# Save to ouput path

img = np.clip(img, 0, 1)          # ensure valid range
img = (img * 255.0).round().astype(np.uint8)

cv2.imwrite(vintage_2_output, img)
