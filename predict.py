from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/depth/weights/best.pt")

# Load image
img_path = "000012_0000001854.png"  # 🔁 replace with your image path
img = cv2.imread(img_path)

# Run inference
results = model.predict(source=img, save=False)

# Plot bounding boxes on the image
for r in results:
    plotted_img = r.plot()  # returns image with bounding boxes (NumPy array)

# Save the result
output_path = "predicted_image6.jpg"
cv2.imwrite(output_path, plotted_img)
print(f"✅ Saved: {output_path}")
