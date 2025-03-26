import numpy as np
import os
import cv2

# Paths
mask_dir = "masks/val"     # Folder with .npy mask files
output_dir = "labels/val"  # Where to save .txt annotation files
image_shape = (480, 640)        # Replace with (height, width) of original images

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(mask_dir):
    if not filename.endswith(".npy"):
        continue

    # Load mask
    mask_path = os.path.join(mask_dir, filename)
    mask = np.load(mask_path)

    h, w = image_shape

    # Prepare lines for YOLO annotation
    lines = []

    # Get object IDs (excluding background = 0)
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids != 0]

    for obj_id in object_ids:
        # Create binary mask for this object
        obj_mask = (mask == obj_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Convert to YOLO format (normalized center x/y, width, height)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h

            # class_id is assumed to be 0 for all objects (change if needed)
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")

    # Write YOLO annotation file
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {txt_path}")

