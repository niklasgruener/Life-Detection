from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("/home/ngruener/thesis/life_detection/NEW/yolo/thermal/runs/detect/train2/weights/best.pt")

# Load image
img_path = "/home/ngruener/thesis/life_detection/test2/thermal_000114_0000013465.png" 
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
