from ultralytics import YOLO
import cv2
import numpy as np

# --- Calibration parameters ---
# Assume raw values are in hundredths of Kelvin, so:
#   0°C == 273.15 K  --> raw value = 27315
#   1°C change == 100 raw units.
THERMAL_RAW_OFFSET = 27315  # raw value corresponding to 0°C
THERMAL_RAW_SCALE  = 100     # raw units per C

# --- Load your trained model ---
model = YOLO("/home/ngruener/thesis/life_detection/NEW/yolo/thermal/runs/detect/train4/weights/best.pt")

# --- Load images ---
img_path = "NEW/yolo/thermal/data/images/test/image_t_1000.png"
# Load a conventional BGR image for inference & plotting.
color_img = cv2.imread(img_path)
if color_img is None:
    raise FileNotFoundError("Could not load image (color).")

# Load the thermal image as a 16-bit image for temperature calibration.
thermal_raw = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
if thermal_raw is None:
    raise FileNotFoundError("Could not load image (16-bit thermal).")

# Convert the raw thermal image to Celsius.
# Using float32 to maintain decimals.
celsius_image = (thermal_raw.astype(np.float32) - THERMAL_RAW_OFFSET) / THERMAL_RAW_SCALE

# --- Run YOLO Inference ---
# (We assume your detection model is trained on this thermal imagery.)
results = model.predict(source=color_img, save=False)

# --- Get the mapping from class indices to names ---
# This dictionary is provided by the model. For example, the 'person' class is usually index 0.
names = model.model.names

# --- Process the detection results ---
for r in results:
    # Use the built-in plot routine to create an image with boxes drawn.
    plotted_img = r.plot()
    
    # Iterate over detections.
    for box in r.boxes:
        # Retrieve class index.
        cls_id = int(box.cls[0]) if hasattr(box.cls, '__iter__') else int(box.cls)
        
        # Check if the detected object is a person.
        if names[cls_id] == 'person':
            # Retrieve bounding box coordinates (xyxy format).
            x1, y1, x2, y2 = box.xyxy[0]
            # Convert coordinates to integer values.
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract the region from the Celsius image.
            person_region = celsius_image[y1:y2, x1:x2]
            if person_region.size == 0:
                continue

            # Calculate the median temperature within the bounding box.
            median_temp = np.median(person_region)
            
            # Prepare the label text without the ° symbol.
            label = f"{median_temp:.1f} C"
            
            # Calculate text size to center it.
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Calculate the center position of the bounding box.
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2
            
            # Compute top-left coordinate of the text such that it is centered.
            text_x = center_x - text_w // 2
            text_y = center_y + text_h // 2
            
            # Draw the text label.
            cv2.putText(plotted_img, label, (text_x, text_y),
                        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # --- Save the Annotated Image ---
    output_path = "pred_image.png"
    cv2.imwrite(output_path, plotted_img)
    print(f"Saved: {output_path}")
