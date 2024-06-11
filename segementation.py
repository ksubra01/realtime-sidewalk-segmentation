import cv2
import numpy as np

def segment_sidewalk_and_obstacles(frame):
    # Convert to HSV color space for color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range of sidewalk color in HSV
    lower_sidewalk = np.array([0, 0, 100])
    upper_sidewalk = np.array([180, 50, 255])
    
    # Threshold the HSV image to get only sidewalk colors
    mask = cv2.inRange(hsv, lower_sidewalk, upper_sidewalk)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Use morphology to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the edges
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask for the sidewalk and obstacles
    sidewalk_mask = np.zeros_like(frame)
    obstacle_mask = np.zeros_like(frame)
    
    for contour in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(contour)
        
        # Filter based on area size
        if area > 500:  # Adjust area threshold as needed
            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check the aspect ratio and size to ensure it's likely a sidewalk
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2:  # Sidewalks are usually long and narrow
                cv2.drawContours(sidewalk_mask, [contour], -1, (0, 255, 0), -1)  # Green for sidewalk
            else:
                cv2.drawContours(obstacle_mask, [contour], -1, (0, 0, 255), -1)  # Red for obstacles
        else:
            cv2.drawContours(obstacle_mask, [contour], -1, (0, 0, 255), -1)  # Red for obstacles

    cv2.imshow("sidwalk_mask", sidewalk_mask)
    
    return sidewalk_mask, obstacle_mask

def calculate_centroid(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    M = cv2.moments(gray)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def predict_future_trajectory(past_centroids, num_future_points=10):
    if len(past_centroids) < 2:
        return []
    
    # Calculate velocity (dx, dy) based on the last two points
    dx = past_centroids[-1][0] - past_centroids[-2][0]
    dy = past_centroids[-1][1] - past_centroids[-2][1]
    
    # Predict future points
    future_points = []
    last_point = past_centroids[-1]
    for i in range(num_future_points):
        next_point = (last_point[0] + dx, last_point[1] + dy)
        future_points.append(next_point)
        last_point = next_point
    
    return future_points

# Store past centroids
past_centroids = []

# Capture video from file
cap = cv2.VideoCapture("D:/2024-03-28_09.29.51.mkv")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Split the frame into left and right views
    h, w, _ = frame.shape
    left_frame = frame[:, :h//2]
    right_frame = frame[:, h//2:]
    
    # Segment sidewalks and obstacles
    left_sidewalk_mask, left_obstacle_mask = segment_sidewalk_and_obstacles(left_frame)
    right_sidewalk_mask, right_obstacle_mask = segment_sidewalk_and_obstacles(right_frame)
    
    # Calculate and store the centroid
    left_centroid = calculate_centroid(left_sidewalk_mask)
    right_centroid = calculate_centroid(right_sidewalk_mask)
    if left_centroid and right_centroid:
        past_centroids.append((left_centroid, right_centroid))
    
    # Draw the past trajectories
    for left_pt, right_pt in past_centroids:
        cv2.circle(left_frame, left_pt, 5, (0, 255, 0), -1)
        cv2.circle(right_frame, right_pt, 5, (0, 255, 0), -1)
    
    # Predict and draw future trajectories
    left_future_points = predict_future_trajectory([pt[0] for pt in past_centroids])
    right_future_points = predict_future_trajectory([pt[1] for pt in past_centroids])
    for pt in left_future_points:
        cv2.circle(left_frame, pt, 5, (0, 0, 255), -1)
    for pt in right_future_points:
        cv2.circle(right_frame, pt, 5, (0, 0, 255), -1)
    
    # Combine the left and right frames back
    combined_frame = np.hstack((left_frame, right_frame))
    
    # Display the combined frame
    cv2.imshow('Frame with Trajectories', combined_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
