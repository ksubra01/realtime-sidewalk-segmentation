import cv2

# Path to the image file
img_path = 'E:/Snowbotix/Data/frame_00024.png'

# Load the image
img = cv2.imread(img_path)

# Check if the image was loaded successfully
if img is None:
    print("Error loading image")
    exit()

# Display the image
cv2.imshow("Image", img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()