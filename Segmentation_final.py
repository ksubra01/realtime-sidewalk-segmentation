import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
from Dataset_class import SegmentationDataset
import torchvision.models.segmentation


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=512, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


image_dir = 'E:/Snowbotix/imgs'
mask_dir = 'E:/Snowbotix/Masks'

# Create dataset
dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)



model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device is:", device)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()  # Add channel dimension
        
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()  # Add channel dimension
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save model after each epoch
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")


# Save the model
torch.save(model.state_dict(), 'deeplabv3_model_new.pth')

# Load the model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
model.load_state_dict(torch.load('deeplabv3_model.pth'))
model = model.to(device)