# import cv2
# import numpy as np

# # Global variables to store the points of the contour
# points = []

# # Mouse callback function to draw the contour
# def draw_contour(event, x, y, flags, param):
#     global points, img_copy
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         points = []
#         img_copy = img.copy()

# # Function to create a mask from the drawn contour
# def create_mask(img, points):
#     mask = np.zeros(img.shape[:2], dtype=np.uint8)
#     if len(points) > 2:
#         cv2.fillPoly(mask, np.array([points], dtype=np.int32), 255)
#     return mask

# # Path to the image file
# img_path = 'E:/Snowbotix/Data/frame_00605.png'

# # Load the image
# img = cv2.imread(img_path)
# if img is None:
#     print("Error loading image")
#     exit()

# img_copy = img.copy()

# # Create a window and set a mouse callback function
# cv2.namedWindow('Image')
# cv2.setMouseCallback('Image', draw_contour)

# while True:
#     cv2.imshow('Image', img_copy)
    
#     if len(points) > 1:
#         cv2.polylines(img_copy, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
    
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # Press 'q' to quit
#         break
#     elif key == ord('c'):  # Press 'c' to clear the contour
#         points = []
#         img_copy = img.copy()
#     elif key == ord('m'):  # Press 'm' to create the mask
#         mask = create_mask(img, points)
#         cv2.imshow('Mask', mask)
#         mask_rgb_values = cv2.bitwise_and(img, img, mask=mask)
#         cv2.imshow('RGB Values in Contour', mask_rgb_values)
        
# cv2.destroyAllWindows()



import cv2
import numpy as np

# Global variables
seeds = []

# Mouse callback function to select seed points for flood fill
def select_seed(event, x, y, flags, param):
    global seeds, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        seeds.append((x, y))
        flood_fill(img, x, y)

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
    _, _, _, rect = cv2.floodFill(img_copy, mask, (x, y), (0, 255, 0), lo_diff, up_diff, flags)
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

# Path to the image file
img_path = 'E:/Snowbotix/Data/frame_00024.png'

# Load the image
img = cv2.imread(img_path)
if img is None:
    print("Error loading image")
    exit()

img_copy = img.copy()

# Create a window and set a mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_seed)

while True:
    cv2.imshow('Image', img_copy)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('c'):  # Press 'c' to clear the seeds and image
        seeds = []
        img_copy = img.copy()
    elif key == ord('m'):  # Press 'm' to create the mask
        mask = create_mask_from_seeds(img, seeds)
        cv2.imshow('Mask', mask)
        mask_rgb_values = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('RGB Values in Contours', mask_rgb_values)
        
cv2.destroyAllWindows()

