import cv2
import numpy as np
import os

# Path to thermal image (grayscale)
input_path = "assets/thermal2.png"
output_path = "assets/thermal_normalized.png"

# Load thermal image in grayscale
thermal16 = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)

if thermal16 is None:
    raise FileNotFoundError(f"Image not found: {input_path}")


thermal8 = np.zeros((480, 640), dtype=np.uint8)
thermal8 = cv2.normalize(thermal16, thermal8, 0, 255, cv2.NORM_MINMAX)
thermal8 = np.uint8(thermal8)




# Apply colormap (you can try other ones like COLORMAP_HOT or COLORMAP_MAGMA)
colored = cv2.applyColorMap(thermal8, cv2.COLORMAP_INFERNO)

# Save the colorized image
cv2.imwrite(output_path, colored)
print(f"Saved normalized thermal image with colormap: {output_path}")
