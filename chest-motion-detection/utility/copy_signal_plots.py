import os
import shutil

# Config:
source_root = 'chest-floor-cleaned'  # Change this to your base folder (contains jacket1, polo1, ...)
target_dir = './'  # Change this to your desired target directory

# Make sure target dir exists
os.makedirs(target_dir, exist_ok=True)

# Clothing types (top level folders)
clothing_types = ['jacket1', 'polo1', 'shirt1', 'sweather1']
modalities = ['rgb', 'thermal', 'depth']

# Walk through the structure and copy files
for clothing in clothing_types:
    for modality in modalities:
        # Build source file path
        source_file = os.path.join(source_root, clothing, 'out', modality, 'signal_plot.png')
        
        if os.path.exists(source_file):
            # Build target filename
            target_filename = f"lean-{clothing}_{modality}_signal_plot.png"
            target_path = os.path.join(target_dir, target_filename)
            
            # Copy file
            shutil.copyfile(source_file, target_path)
            print(f"Copied: {source_file} → {target_path}")
        else:
            print(f"WARNING: File not found → {source_file}")

print("Done.")

