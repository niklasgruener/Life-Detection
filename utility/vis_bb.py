import numpy as np
import cv2
import matplotlib.pyplot as plt

# CONFIGURE THIS
mask_path = "../masks/val/0_000000_0000000448.npy"  # Change to your mask path
image_shape = (480, 640)               # (height, width)

# Load the mask
mask = np.load(mask_path)

# If the mask has extra dimensions (e.g., HxWx1), squeeze it
if mask.ndim == 3:
    mask = mask.squeeze()

h, w = image_shape
vis_image = np.zeros((h, w, 3), dtype=np.uint8)

# Assign colors to each object ID
colors = {}

object_ids = np.unique(mask)
object_ids = object_ids[object_ids != 0]  # Ignore background

for obj_id in object_ids:
    obj_mask = (mask == obj_id).astype(np.uint8)
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Random color per object
    color = colors.get(obj_id)
    if color is None:
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        colors[obj_id] = color

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + bw, y + bh), color, 2)

        # Label with ID
        cv2.putText(vis_image, f"ID:{obj_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Display
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
plt.title("Bounding Boxes on Segmentation Mask")
plt.axis("off")
plt.show()
