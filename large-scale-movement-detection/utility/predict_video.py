from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("/home/ngruener/thesis/life_detection/NEW/yolo/depth/runs/detect/train2/weights/best.pt")

# Load video
input_video_path = "input_video_NEW_depth.mp4"     
output_video_path = "output_video_NEW_depth.mp4"  

cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(source=frame, save=False, device=0)

    # Draw bounding boxes
    for r in results:
        plotted_frame = r.plot()

    # Write to output video
    out.write(plotted_frame)
    frame_count += 1

    print(f"Processed frame {frame_count}", end="\r")

# Cleanup
cap.release()
out.release()
print(f"\nDone! Saved to: {output_video_path}")
