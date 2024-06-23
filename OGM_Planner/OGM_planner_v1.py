import os
import cv2
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import imutils
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from tqdm import tqdm
import time
import heapq

# Unet Architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.decoder4 = self.up_conv_block(1024, 512)
        self.decoder3 = self.up_conv_block(512, 256)
        self.decoder2 = self.up_conv_block(256, 128)
        self.decoder1 = self.up_conv_block(128, 64)
        self.decoder0 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        
        # Final convolutional layer
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block
    
    def up_conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder
        dec4 = self.decoder4(bottleneck)
        dec3 = self.decoder3(dec4 + enc4)
        dec2 = self.decoder2(dec3 + enc3)
        dec1 = self.decoder1(dec2 + enc2)
        dec0 = self.decoder0(dec1 + enc1)
        
        # Final convolutional layer
        out = self.final_conv(dec0)
        
        return out




# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('model_epoch_16_200.pth', map_location=device))
model = model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

execution_times = {
    'predict': [],
    'get_contour': [],
    'goal_from_mask': [],
    'occupancy_grid_route': [],
    'get_commands': [],
    'draw_segmentation_and_path': [],
    'gradient_planner': []
}

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_times[func.__name__].append(end_time - start_time)
        return result
    return wrapper

@measure_time
def get_contour(binary_mask):
    thresh = np.array(255 * (1 - binary_mask), dtype=np.uint8)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None 
    contour = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # Keep only the largest contour
    return contour

@measure_time
def goal_from_mask(binary_mask, i):
    drivable_indexes = np.asarray(np.where(binary_mask == 0))
    if drivable_indexes.size == 0:
        print("Frame number is:", i)
        plt.imshow(binary_mask)
        plt.title("binary mask when d index is zero")
        plt.show()
        goal = [0, 0]
        return goal
    farest_y = np.min(drivable_indexes[0, :])  # Changed to min
    farest_id = np.where(drivable_indexes[0, :] == farest_y)[0][0]
    farest_x = drivable_indexes[1, farest_id]

    goal = [farest_x, farest_y]
    return goal

@measure_time
def get_commands(binary_mask, route):
    _, w = binary_mask.shape
    route_int = np.asarray(route, dtype=np.int)
    m = route_int[:, 0]
    h = np.max(route_int[:, 1]) - np.min(route_int[:, 1])

    beta = 0.01
    omega = beta * np.mean(m - w / 2)

    alpha = 0.02
    vel = alpha * h - np.abs(omega)

    return vel, omega

@measure_time
def draw_segmentation_and_path(frame, mask, route, goal):
    overlay = frame.copy()
    overlay[mask == 1] = [0, 255, 0]  # Green color for the mask
    if route is not None:
        for point in route:
            cv2.circle(overlay, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)  # Blue color for the route
        cv2.circle(overlay, (int(goal[0]), int(goal[1])), 5, (0, 255, 255), -1)  # Red color for the goal
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    return blended

@measure_time
def predict(model, frame, device):
    model.eval()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_image = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)

    output = torch.sigmoid(output).cpu().numpy()[0, 0]
    output = (output > 0.5).astype(np.uint8)

    return output

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

@measure_time
def occupancy_grid_route(binary_mask, goal):
    start = (int(np.mean(np.where(binary_mask[0] == 0))), 0)
    goal = (goal[1], goal[0])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    nrows, ncols = binary_mask.shape
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            route = []
            while current in came_from:
                route.append((current[1], current[0]))
                current = came_from[current]
            route.append((start[1], start[0]))
            return route[::-1]

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < nrows and 0 <= neighbor[1] < ncols and binary_mask[neighbor] == 0:
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def process_video(input_video_path, output_video_path, roi_coords):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        pred_mask = predict(model, frame, device)
        pred_mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))
        roi_mask = pred_mask_resized[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
        contour = get_contour(roi_mask)

        if contour is not None:
            contour_mask = np.zeros_like(roi_mask)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
            goal = goal_from_mask(contour_mask, i)
            route = occupancy_grid_route(contour_mask, goal)
            if route is not None:
                route = np.array(route) + np.array([roi_coords[0], roi_coords[1]])
        else:
            route = None
            goal = None

        result_frame = draw_segmentation_and_path(frame, pred_mask_resized, route, goal)
        
        # Add frame number to the frame
        cv2.putText(result_frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(result_frame)

    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")

# Example usage
input_video_path = "E:/snowbotix_edit.mp4"
output_video_path = 'APF_Planner_final.mp4'
roi_coords = (128, 360, 574, 481)  # (x1, y1, x2, y2)

def plot_execution_times(execution_times):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for func_name, times in execution_times.items():
        ax.plot(times, label=func_name)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time per Function per Frame')
    ax.legend()
    plt.show()

process_video(input_video_path, output_video_path, roi_coords)
plot_execution_times(execution_times)
