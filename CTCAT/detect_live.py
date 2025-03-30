from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("runs/detect/depth/weights/best.pt")

# Load video
input_video_path = "video.mp4"  # 🔁 Input video file
cap = cv2.VideoCapture(input_video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(source=frame, save=False, device=0)  # use 'cpu' if needed

    # Draw bounding boxes
    for r in results:
        plotted_frame = r.plot()

    # Show frame
    cv2.imshow("Live YOLOv8 Predictions", plotted_frame)
    frame_count += 1
    print(f"Processed frame {frame_count}", end="\r")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✅ Display finished.")
