from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("rgb/runs/detect/train3/weights/best.pt")

# Load image
img_path = "test1/rgb/rgb_00071.png" 
img = cv2.imread(img_path)

# Run inference
results = model.predict(source=img, save=False)

# Plot bounding boxes on the image
for r in results:
    plotted_img = r.plot()  

# Save the result
output_path = "pred_image.png"
cv2.imwrite(output_path, plotted_img)
print(f"Saved: {output_path}")
