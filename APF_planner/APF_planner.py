import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from U_Net import UNet  # Ensure this is the correct import path
import matplotlib.pyplot as plt
import imutils
from scipy.ndimage.morphology import distance_transform_edt as bwdist
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


execution_times = {
    'predict': [],
    'get_contour': [],
    'goal_from_mask': [],
    'APF_route': [],
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

i = 0
# def get_binary(pr_mask):
#     global i
#     mask = pr_mask.squeeze()
#     blurred = cv2.GaussianBlur(mask, (5, 5), 0)
#     mask255 = np.array(255 * (blurred / blurred.max()), dtype=np.uint8)  # Ensure correct normalization
#     _, thresh255 = cv2.threshold(mask255, np.mean(mask255), np.max(mask255), cv2.THRESH_BINARY)
#     binary_mask = np.array(1 - (thresh255 / 255), dtype=np.int)

#     cv2.imwrite(f"Binary_image_{i}.png", binary_mask * 255)  # Save binary mask for debugging
#     i += 1
#     return binary_mask
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
def gradient_planner(f, start_coords, end_coords, max_its, binary_mask):
    [gy, gx] = np.gradient(-f)

    route = np.vstack([np.array(start_coords), np.array(start_coords)])

    for i in range(max_its):
        current_point = route[-1, :]
        if sum(abs(current_point - end_coords)) < 5.0:
            print('Reached the goal !')
            break
        ix = int(round(current_point[1]))
        iy = int(round(current_point[0]))
        ix = min(max(ix, 0), gx.shape[0] - 1)
        iy = min(max(iy, 0), gx.shape[1] - 1)
        vx = gx[ix, iy]
        vy = gy[ix, iy]
        dt = 1 / np.linalg.norm([vx, vy])
        next_point = current_point + dt * np.array([vx, vy])
        next_point[1] = min(max(next_point[1], 0), binary_mask.shape[0] - 1)
        next_point[0] = min(max(next_point[0], 0), binary_mask.shape[1] - 1)
        midpoint_x = np.mean(np.where(binary_mask[int(next_point[1]), :] == 0))
        next_point[0] = midpoint_x
        route = np.vstack([route, next_point])
    route = route[1:, :]

    return route

@measure_time
def APF_route(binary_mask, goal,i):
    d = bwdist(binary_mask == 0)  # The binary mask must be that of obstacles. The bianry mask s correct.
    d2 = (d / 100.) + 1  # Rescale and transform distances
    d0 = 1.3 #1.2            # threshold value for repulsive force  - Increase this to increase the repulsive forces in the boundary
    nu = 500            # Scaling factor - How much the robot is pushed from the obstacles - Increase this to increase the repulsive forces overall
    repulsive = nu * ((1. / d2 - 1 / d0) ** 2)
    repulsive[d2 > d0] = 0
    # print("repulsive is", repulsive)
    drivable_indexes = np.asarray(np.where(binary_mask == 0))
    if drivable_indexes.size !=0:
        
        start_y = np.max(drivable_indexes[0, :]) ##3 Changed to max
        start_x = np.mean(drivable_indexes[1, np.where(drivable_indexes[0, :] == start_y)])
    else:
        start_x, start_y = 0,0
    if start_x > 210:
        start_x = 210
    start = [start_x, start_y]
    nrows, ncols = binary_mask.shape
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xi = 1 / 700.   # Changed from 700
    attractive = xi * ((x - goal[0]) ** 2 + (y - goal[1]) ** 2)

    f = attractive + repulsive

    # if i%500 == 0:


    #     print("Start is", start)
    #     print("Goal is", goal)
    #     plt.figure()
    #     plt.imshow(binary_mask)
    #     plt.title("Binary mask")
    #     plt.show()

    #     plt.figure()
    #     plt.imshow(f, cmap='hot')
    #     plt.title("Potential Field")
    #     plt.colorbar()
    #     plt.show()



    route = gradient_planner(f, start, goal, 700, binary_mask)
    return route

@measure_time
def goal_from_mask(binary_mask,i):
    drivable_indexes = np.asarray(np.where(binary_mask == 0))

    if drivable_indexes.size == 0:
        print("Frame number is:", i)
        plt.imshow(binary_mask)
        plt.title("binary mask when d index is zero")
        plt.show()
        goal = [0,0]
        return goal
    # if drivable_indexes.size != 0:
    farest_y = np.min(drivable_indexes[0, :]) ### Changed to min
    farest_id = np.where(drivable_indexes[0, :] == farest_y)[0][0]
    farest_x = drivable_indexes[1, farest_id]

    goal = [farest_x, farest_y]
    #else:
    #goal = [0, 0]  # Default goal if no drivable area is found

    # print("Goal coordinates", goal)

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
        # plt.figure()
        # plt.imshow(pred_mask)
        # plt.show()

        pred_mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))
        # plt.figure()
        # plt.imshow(pred_mask_resized)
        # plt.show()
       
        roi_mask = pred_mask_resized[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]

        # plt.figure()
        # plt.imshow(roi_mask)

        # plt.show()

        # cv2.imshow("ROI Mask", roi_mask)
        # cv2.waitKey(0)

        #binary_mask = get_binary(roi_mask)
        contour = get_contour(roi_mask)


        if contour is not None:
            contour_mask = np.zeros_like(roi_mask)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
            # plt.figure()
            # plt.imshow(contour_mask)
            # plt.title("the contour map is")

            # plt.show()


            goal = goal_from_mask(contour_mask,i)
            route = APF_route(contour_mask, goal,i)
            if route is not None:
                route = route + np.array([roi_coords[0], roi_coords[1]])
        else:
            route = None
            goal = None


        result_frame = draw_segmentation_and_path(frame, pred_mask_resized, route, goal)
        # if i%500 ==0 :
        #     plt.figure()
        #     plt.imshow(result_frame)
        #     plt.title("Frame")
        #     plt.axis('off')
        #     plt.show()
        out.write(result_frame)

        # if i == 100:
        #     break

    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")

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
