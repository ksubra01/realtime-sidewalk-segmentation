import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
import torchvision.models.segmentation
import numpy as np
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model.load_state_dict(torch.load('model_epoch_5.pth'))
model = model.to(device)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=512, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict(model, frame, device):
    model.eval()
    augmented = transform(image=frame)
    input_image = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_image)['out']
        
    output = torch.sigmoid(output).cpu().numpy()[0, 0]
    output = (output > 0.5).astype(np.uint8)
    
    return output

video_path = 'E:/snowbotix_edit.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output_segmented_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    pred_mask = predict(model, frame, device)
    
    # Resize pred_mask to match the frame size
    pred_mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))
    
    # Overlay the mask on the frame
    overlay = frame.copy()
    overlay[pred_mask_resized == 1] = [0, 255, 0]  # Green color for the mask
    
    # Blend the frame and the overlay
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    print("yes")
    
    out.write(blended)

cap.release()
out.release()
cv2.destroyAllWindows()
