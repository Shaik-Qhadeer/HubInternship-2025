import cv2
import os

video_path = "video/input_video.mp4"
output_folder = "extracted_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
frame_interval = int(fps)  # extract 1 frame every second

frame_number = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_number += 1

cap.release()
print(f"Extracted {saved_frame_count} frames (1 per second).")
