import cv2
import numpy as np
import os

# Global variables
seeds = []

# Mouse callback function to select seed points for flood fill
def select_seed(event, x, y, flags, param):
    global seeds, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        seeds.append((x, y))
        flood_fill(img_copy, x, y)

# Function to perform flood fill
def flood_fill(img, x, y):
    global img_copy
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    connectivity = 4
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE
    flags |= (255 << 8)
    lo_diff = (15, 15, 15)
    up_diff = (15, 15, 15)
    _, _, _, rect = cv2.floodFill(img, mask, (x, y), (0, 255, 0), lo_diff, up_diff, flags)
    cv2.imshow('Flood Fill', img_copy)

# Function to create a mask from the flood filled regions
def create_mask_from_seeds(img, seeds):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for seed in seeds:
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        connectivity = 4
        flags = connectivity
        flags |= cv2.FLOODFILL_FIXED_RANGE
        flags |= (255 << 8)
        lo_diff = (15, 15, 15)
        up_diff = (15, 15, 15)
        _, _, _, rect = cv2.floodFill(img, flood_mask, seed, (255), lo_diff, up_diff, flags)
        mask |= flood_mask[1:-1, 1:-1]
    return mask

# Path to the folders
source_folder = 'E:/Snowbotix/Data'
destination_folder = 'E:/Snowbotix/Masks'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Load the images from the source folder
image_paths = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img_path in image_paths:
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image {img_path}")
        continue

    img_copy = img.copy()
    seeds = []

    # Create a window and set a mouse callback function
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_seed)

    while True:
        cv2.imshow('Image', img_copy)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit the current image
            break
        elif key == ord('c'):  # Press 'c' to clear the seeds and image
            seeds = []
            img_copy = img.copy()
        elif key == ord('m'):  # Press 'm' to create the mask
            if seeds:
                mask = create_mask_from_seeds(img, seeds)
                cv2.imshow('Mask', mask)
                mask_rgb_values = cv2.bitwise_and(img, img, mask=mask)
                cv2.imshow('RGB Values in Contours', mask_rgb_values)

                # Save the mask to the destination folder with the new name
                base_name = os.path.basename(img_path)
                frame_number = base_name.split('_')[1].split('.')[0]  # Extract the frame number
                mask_name = f'mask_{frame_number}.png'  # Create the new mask name
                mask_path = os.path.join(destination_folder, mask_name)
                cv2.imwrite(mask_path, mask)
                break
    
    cv2.destroyAllWindows()
