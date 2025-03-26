import os
import shutil

# Set the source and destination directories
source_root = 'tristar/val'
destination_dir = 'masks/val'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop through each numbered folder in the source
for folder in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder)
    if os.path.isdir(folder_path):
        # Look for the 'depth' subfolder
        depth_folder = os.path.join(folder_path, 'mask')
        if os.path.isdir(depth_folder):
            # Iterate over each file in the 'depth' folder
            for filename in os.listdir(depth_folder):
                file_path = os.path.join(depth_folder, filename)
                if os.path.isfile(file_path):
                    # Prepend the folder name to the file name
                    new_filename = f"{folder}_{filename}"
                    destination_file = os.path.join(destination_dir, new_filename)
                    shutil.copy(file_path, destination_file)
                    print(f"Copied {file_path} to {destination_file}")
