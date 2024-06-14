import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from U_Net import UNet  # Ensure this is the correct import path
import matplotlib.pyplot as plt
import imutils
from tqdm import tqdm

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

i = 0
# def get_binary(pr_mask):
#     global i
#     mask = pr_mask.squeeze()
#     blurred = cv2.GaussianBlur(mask, (5, 5), 0)
#     mask255 = np.array(255 * (blurred / blurred.max()), dtype=np.uint8)  # Ensure correct normalization
#     _, thresh255 = cv2.threshold(mask255, nps.mean(mask255), np.max(mask255), cv2.THRESH_BINARY)
#     binary_mask = np.array(1 - (thresh255 / 255), dtype=np.int)

#     cv2.imwrite(f"Binary_image_{i}.png", binary_mask * 255)  # Save binary mask for debugging
#     i += 1
#     return binary_mask

def get_contour(binary_mask):
    thresh = np.array(255 * (1 - binary_mask), dtype=np.uint8)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    contour = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # Keep only the largest contour
    return contour

def get_middle_route(binary_mask):
    drivable_indexes = np.asarray(np.where(binary_mask == 0))

    if drivable_indexes.size == 0:
        return None

    farest_y = np.min(drivable_indexes[0, :])

    h = binary_mask.shape[0]
    middle_route = []
    for i in range(farest_y, h-2):
        road_horizontal_slice = np.asarray(np.where(binary_mask[i, :] == 0)).squeeze()
        try:  # road_horizontal_slice could be empty or single number
            left_x = np.min(road_horizontal_slice)
            right_x = np.max(road_horizontal_slice)
            m_i = np.mean([left_x, right_x])
            middle_route.append([m_i, i])
        except:
            pass
    middle_route = np.array(middle_route)

    return middle_route

def draw_segmentation_and_path(frame, mask, route):
    overlay = frame.copy()
    overlay[mask == 1] = [0, 255, 0]  # Green color for the mask
    try:
        for point in route:
            cv2.circle(overlay, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)  # Blue color for the route
    except:
        pass
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    return blended

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

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        pred_mask = predict(model, frame, device)
        cv2.imshow("Pred_mask is", pred_mask)
        cv2.waitKey(0)
        pred_mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))
        roi_mask = pred_mask_resized[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
        
        # binary_mask = get_binary(roi_mask)
        contour = get_contour(roi_mask)
        contour_mask = np.zeros_like(roi_mask)
        cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
        route = get_middle_route(contour_mask)
        try:
            if len(route):
                route = route + np.array([roi_coords[0], roi_coords[1]])
        except:
            pass

        result_frame = draw_segmentation_and_path(frame, pred_mask_resized, route)
        out.write(result_frame)
    
    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")

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

# Example usage
input_video_path = "E:/snowbotix_edit.mp4"
output_video_path = 'SAI_RAM.mp4'
roi_coords = (130, 360, 566, 481)  # (x1, y1, x2, y2)

process_video(input_video_path, output_video_path, roi_coords)
