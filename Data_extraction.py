import cv2
import os

# Path to the video file
video_path = 'E:/snowbotix_edit.mp4'

# Directory where frames will be saved
output_dir = 'E:/Snowbotix/Data'

# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Construct the output file path
    output_file = os.path.join(output_dir, f'frame_{frame_count:05d}.png')
    
    # Save the frame as an image
    cv2.imwrite(output_file, frame)
    
    frame_count += 1

# Release the video capture object
cap.release()

print(f'Saved {frame_count} frames to {output_dir}')
