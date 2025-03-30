import cv2
import os

# Set the path to your frames and output video
frames_folder = 'tristar/train/0/thermal'
output_video_path = 'input_video.mp4'
frame_rate = 7  # frames per second

# Get a sorted list of image files
image_files = sorted([
    f for f in os.listdir(frames_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# Read the first frame to get dimensions
first_frame = cv2.imread(os.path.join(frames_folder, image_files[0]))
height, width, layers = first_frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

# Write each frame to the video
for file_name in image_files:
    frame = cv2.imread(os.path.join(frames_folder, file_name))
    video_writer.write(frame)

video_writer.release()
print(f"Video saved to {output_video_path}")
