from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("runs/detect/depth/weights/best.pt")

# Load video
input_video_path = "video.mp4"         # 🔁 Your input video
output_video_path = "output_predicted.mp4"   # 🔁 Output path

cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert grayscale to RGB if needed
    # if len(frame.shape) == 2 or frame.shape[2] == 1:
    #    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Run inference
    results = model.predict(source=frame, save=False, device=0)  # use 'cpu' if needed

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
print(f"\n✅ Done! Saved to: {output_video_path}")
