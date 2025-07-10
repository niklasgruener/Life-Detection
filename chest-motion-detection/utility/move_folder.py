import os
import shutil

# Path to your folder with all images
source_folder = '/home/uni/Documents/Uni/Thesis/scenarios/chest/chest-side/sweater'  # CHANGE THIS to your actual folder path

# Target subfolders (will be created if they do not exist)
target_folders = {
    'rgb_': os.path.join(source_folder, 'rgb'),
    'depth_': os.path.join(source_folder, 'depth'),
    'thermal_': os.path.join(source_folder, 'thermal')
}

# Create target folders if they don't exist
for folder in target_folders.values():
    os.makedirs(folder, exist_ok=True)

# Move files based on prefix
for filename in os.listdir(source_folder):
    for prefix, target_folder in target_folders.items():
        if filename.startswith(prefix):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)
            shutil.move(source_path, target_path)
            print(f"Moved {filename} → {target_folder}")

print("Done moving files.")

