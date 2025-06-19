import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to your images and the bounding box txt file
depth_image_path = 'assets/dataset_vis_depth2.png'
rgb_image_path = 'assets/dataset_vis_rgb2.png'
thermal_image_path = 'assets/dataset_vis_thermal2.png'
bbox_txt_path = 'assets/dataset_vis2.txt'

# Boolean flag: if True, write figure to disk; if False, display the figure.
write_to_disk = True  # Set to False to display instead of saving

# Read the YOLO-format bounding box from the text file
with open(bbox_txt_path, 'r') as f:
    line = f.readline().strip()
parts = line.split()
if len(parts) == 5:
    _, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts)
else:
    raise ValueError("Bounding box file format not recognized. Expected 4 or 5 values.")

# Function to convert YOLO normalized bbox to pixel coordinates and draw it on the image.
def draw_bbox(image, x_center_norm, y_center_norm, width_norm, height_norm):
    h, w = image.shape[:2]
    x_center = x_center_norm * w
    y_center = y_center_norm * h
    box_w = width_norm * w
    box_h = height_norm * h
    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)
    
    # Draw the bounding box (red color, thickness=2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    return image

# --- Load and process the Depth image ---
depth_img = cv2.imread(depth_image_path)
depth_copy = depth_img.copy()
depth_copy = cv2.normalize(depth_copy, None, 0, 255, cv2.NORM_MINMAX)
depth_copy = depth_copy.astype(np.uint8)
depth_copy = cv2.applyColorMap(depth_copy, cv2.COLORMAP_VIRIDIS)
depth_with_box = draw_bbox(depth_copy, x_center_norm, y_center_norm, width_norm, height_norm)

# --- Load and process the RGB image ---
rgb_img = cv2.imread(rgb_image_path)
rgb_copy = rgb_img.copy()
rgb_with_box = draw_bbox(rgb_copy, x_center_norm, y_center_norm, width_norm, height_norm)

# --- Load and process the Thermal image ---
thermal_img = cv2.imread(thermal_image_path, cv2.IMREAD_ANYDEPTH)
thermal_copy = thermal_img.copy()
thermal_copy = cv2.normalize(thermal_img, None, 0, 255, cv2.NORM_MINMAX)
thermal_copy = thermal_copy.astype(np.uint8)
thermal_copy = cv2.applyColorMap(thermal_copy, cv2.COLORMAP_INFERNO)
thermal_with_box = draw_bbox(thermal_copy, x_center_norm, y_center_norm, width_norm, height_norm)

# Convert images from BGR to RGB for proper display in Matplotlib
depth_disp = cv2.cvtColor(depth_with_box, cv2.COLOR_BGR2RGB)
rgb_disp = cv2.cvtColor(rgb_with_box, cv2.COLOR_BGR2RGB)
thermal_disp = cv2.cvtColor(thermal_with_box, cv2.COLOR_BGR2RGB)

# Create a figure with 3 subplots for side-by-side display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(depth_disp)
axes[0].set_title("Depth Image")
axes[0].axis('off')

axes[1].imshow(rgb_disp)
axes[1].set_title("RGB Image")
axes[1].axis('off')

axes[2].imshow(thermal_disp)
axes[2].set_title("Thermal Image")
axes[2].axis('off')

plt.tight_layout()

# Either save the figure to disk or display it
if write_to_disk:
    output_path = 'assets/output_figure2.png'
    plt.savefig(output_path)
    print(f"Figure saved to '{output_path}'")
else:
    plt.show()
