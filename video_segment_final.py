import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from U_Net import UNet  # Ensure this is the correct import path
from Dataset_class import SegmentationDataset  # Ensure this is the correct import path
from tqdm import tqdm
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('model_epoch_100.pth', map_location=device))
model = model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict(model, frame, device):
    model.eval()
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL image
    pil_image = Image.fromarray(frame_rgb)
    
    # Apply transformations
    input_image = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_image)
        
    output = torch.sigmoid(output).cpu().numpy()[0, 0]
    output = (output > 0.5).astype(np.uint8)
    
    return output

def process_video(input_video_path, output_video_path, model, device):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()

    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict the segmentation mask
            pred_mask = predict(model, frame, device)
            
            # Resize pred_mask to match the frame size
            pred_mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))

            # Create an overlay image
            overlay = frame.copy()
            overlay[pred_mask_resized == 1] = [0, 255, 0]  # Green color for the mask

            # Blend the original image and the overlay
            blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Write the frame to the output video
            out.write(blended)

            pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Video processing completed in {total_time:.2f} seconds.")

    # Release everything if job is finished
    cap.release()
    out.release()

input_video_path = "E:/snowbotix_edit.mp4"  # Path to the input video
output_video_path = 'video_hopefully_big.mp4'  # Path to save the output video

process_video(input_video_path, output_video_path, model, device)
