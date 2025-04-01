import cv2
import numpy as np
import os

# Path to thermal image (grayscale)
input_path = "000096_0000012062.png"
output_path = "thermal_normalized_colored2.png"

# Load thermal image in grayscale
thermal = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

if thermal is None:
    raise FileNotFoundError(f"Image not found: {input_path}")

# Normalize to 0–255
norm_thermal = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX)
norm_thermal = norm_thermal.astype(np.uint8)

# Apply colormap (you can try other ones like COLORMAP_HOT or COLORMAP_MAGMA)
colored = cv2.applyColorMap(norm_thermal, cv2.COLORMAP_JET)

# Save the colorized image
cv2.imwrite(output_path, colored)
print(f"✅ Saved normalized thermal image with colormap: {output_path}")
