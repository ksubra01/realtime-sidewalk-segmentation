import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from U_Net import UNet  # Ensure this is the correct import path
#from Dataset_class import SegmentationDataset  # Ensure this is the correct import path
import math

# Set environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Constants
PI = 3.1415926
frameWidth = 720
frameHeight = 720

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('model_epoch_100.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict(model, frame, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_image = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_image)
        
    output = torch.sigmoid(output).cpu().numpy()[0, 0]
    output = (output > 0.5).astype(np.uint8)
    return output

def update_perspective(alpha, beta, gamma, focalLength, dist, roi):
    alpha = (alpha - 90) * PI / 180
    beta = (beta - 90) * PI / 180
    gamma = (gamma - 90) * PI / 180

    image_size = (roi.shape[1], roi.shape[0])
    w, h = image_size

    A1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 0],
                   [0, 0, 1]], dtype=np.float32)

    RX = np.array([[1, 0, 0, 0],
                   [0, math.cos(alpha), -math.sin(alpha), 0],
                   [0, math.sin(alpha), math.cos(alpha), 0],
                   [0, 0, 0, 1]], dtype=np.float32)

    RY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                   [0, 1, 0, 0],
                   [math.sin(beta), 0, math.cos(beta), 0],
                   [0, 0, 0, 1]], dtype=np.float32)

    RZ = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                   [math.sin(gamma), math.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)

    R = np.dot(np.dot(RX, RY), RZ)

    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dist],
                  [0, 0, 0, 1]], dtype=np.float32)

    K = np.array([[focalLength, 0, w / 2, 0],
                  [0, focalLength, h / 2, 0],
                  [0, 0, 1, 0]], dtype=np.float32)

    transformationMat = np.dot(np.dot(np.dot(K, T), R), A1)

    roi_transformed = cv2.warpPerspective(roi, transformationMat, image_size, flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    return roi_transformed

def update_occupancy_grid(grid, transformed_roi, alpha=0.9):
    occupied_indices = np.where(transformed_roi > 0)
    grid[occupied_indices] = grid[occupied_indices] * alpha + (1 - alpha)
    free_indices = np.where(transformed_roi == 0)
    grid[free_indices] = grid[free_indices] * alpha
    return grid

def draw_occupancy_grid(grid):
    occupancy_grid_img = (grid * 255).astype(np.uint8)
    occupancy_grid_img = cv2.applyColorMap(occupancy_grid_img, cv2.COLORMAP_JET)
    return occupancy_grid_img

# Video file path
video_path = "E:/snowbotix_small_2.mp4"
cap = cv2.VideoCapture(video_path)

# Define the ROI using the provided coordinates
roi_coords = (128, 360, 574, 481)
roi_x, roi_y, roi_x2, roi_y2 = roi_coords

# Parameters for perspective transformation
alpha = 90
beta = 90
gamma = 90
focalLength = 500
dist = 500

# Initialize the occupancy grid
occupancy_grid = np.zeros((roi_y2 - roi_y, roi_x2 - roi_x))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Generate the segmentation mask
    pred_mask = predict(model, frame, device)
    
    # Extract the region of interest from the segmentation mask
    roi = pred_mask[roi_y:roi_y2, roi_x:roi_x2]
    
    # Apply perspective transformation to the ROI
    roi_transformed = update_perspective(alpha, beta, gamma, focalLength, dist, roi)
    
    # Update the occupancy grid
    occupancy_grid = update_occupancy_grid(occupancy_grid, roi_transformed)
    
    # Generate the occupancy grid map image
    occupancy_grid_img = draw_occupancy_grid(occupancy_grid)
    
    # Display the results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Segmented Mask", pred_mask * 255)
    cv2.imshow("Bird's Eye View", roi_transformed)
    cv2.imshow("Occupancy Grid Map", occupancy_grid_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
