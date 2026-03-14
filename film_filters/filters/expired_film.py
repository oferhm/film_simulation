import cv2
import numpy as np
import sys
import os

# Add parent directories to Python path to find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from config import input, expired_film_output

# load image
img = cv2.imread(input)

# Check if image was loaded successfully
if img is None:
    print(f"❌ Error: Could not load '{input}'")
    exit(1)

rows, cols = img.shape[:2]

# --- 1. ULTRA-SOFT COLOR SHIFT (FILM TINT) ---
# Instead of strong tints, we create a very faint 'wash'
tint_layer = img.copy().astype(np.float32)
# Very subtle: 0.98 Blue (slight yellow) / 1.02 Green (slight Fuji tint)
tint_layer[:, :, 0] *= 0.98  
tint_layer[:, :, 1] *= 1.02  
tint_layer = np.clip(tint_layer, 0, 255).astype(np.uint8)

# Use a very low alpha (0.1 to 0.2) for a "whisper" of color
img = cv2.addWeighted(tint_layer, 0.15, img, 0.85, 0)

# --- 2. FADED BLACKS (THE "EXPIRED" Haze) ---
# This lifts the shadows so blacks become dark gray
img = img.astype(np.float32)
img = img + 8.0  # Add constant to lift shadows
img = np.clip(img, 0, 255).astype(np.uint8)

# --- 3. ULTRA-SOFT VIGNETTE ---
# Use a massive Sigma (3x the image size) for an almost invisible falloff
rows, cols = img.shape[:2]
sigma = max(rows, cols) * 1.5  

kernel_x = cv2.getGaussianKernel(cols, int(sigma))
kernel_y = cv2.getGaussianKernel(rows, int(sigma))
kernel = kernel_y * kernel_x.T
mask = kernel / kernel.max()

# Blend the vignette layer at 30% strength 
# This ensures the edges only darken by a tiny fraction
vignette_img = (img.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
img = cv2.addWeighted(vignette_img, 0.3, img, 0.7, 0)


# ---------- Save ----------
# Save to output path (img is already uint8 in 0-255 range)
cv2.imwrite(expired_film_output, img)
