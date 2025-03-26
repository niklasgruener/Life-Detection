import numpy as np
import os
import cv2

# Root directory containing all the subfolders (train/0/mask, train/1/mask, etc.)
root_mask_dir = "tristar/train"  # 🔁 Replace with your actual path

# Output folder to save all .txt annotation files
output_dir = "dataset_thermal/labels"
os.makedirs(output_dir, exist_ok=True)

# Image size (adjust if different)
image_shape = (480, 640)  # (height, width)
h, w = image_shape

# Walk through subdirectories
for subdir, _, files in os.walk(root_mask_dir):
    if not subdir.endswith("mask"):
        continue

    for filename in files:
        if not filename.endswith(".npy"):
            continue

        # Load mask
        mask_path = os.path.join(subdir, filename)
        mask = np.load(mask_path)

        # Prepare YOLO annotation lines
        lines = []
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]  # Skip background

        for obj_id in object_ids:
            obj_mask = (mask == obj_id).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)

                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h

                lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")

        # Save to output directory using original filename (without extension)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Saved: {txt_path}")
