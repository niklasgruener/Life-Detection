import os
import shutil

# Root folder where your train subfolders are
root_dir = "tristar/train"  # 👈 change this
source_folder_name = "depth"
output_folder = "dataset_depth/images"  # 👈 where all images go

os.makedirs(output_folder, exist_ok=True)

for subdir, _, files in os.walk(root_dir):
    if not subdir.endswith(source_folder_name):
        continue

    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            continue

        src_path = os.path.join(subdir, file)
        dst_path = os.path.join(output_folder, file)

        # Optional: rename if duplicates possible
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(file)
            counter = 1
            while os.path.exists(dst_path):
                new_name = f"{base}_{counter}{ext}"
                dst_path = os.path.join(output_folder, new_name)
                counter += 1

        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} → {dst_path}")
