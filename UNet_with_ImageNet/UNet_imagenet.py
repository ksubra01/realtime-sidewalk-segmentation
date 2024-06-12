import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

class OxfordPetsDataset(Dataset):
    def __init__(self, root, split='train', transform=None, resize=None):
        self.dataset = OxfordIIITPet(root=root, split=split, download=True)
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        mask_path = str(self.dataset._images[idx])[:-4] + '.jpg'
        mask = Image.open(mask_path).convert("L")
        
        # Resize images and masks
        if self.resize is not None:
            image = image.resize(self.resize, Image.BILINEAR)
            mask = mask.resize(self.resize, Image.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

class ToTensor:
    def __call__(self, image, mask):
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        return image, mask

transform = ToTensor()
resize = (128, 128)  # Define the target size for resizing

train_dataset = OxfordPetsDataset(root='data', split='trainval', transform=transform, resize=resize)

# train_dataset = OxfordPetsDataset(root='data', split='trainval', transform=transform)
test_dataset = OxfordPetsDataset(root='data', split='test', transform=transform, resize=resize)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
# Define the model using a pre-trained ResNet34 encoder
model = smp.Unet(
    encoder_name="resnet34",        # Use ResNet34 as the encoder
    encoder_weights="imagenet",     # Use ImageNet pre-trained weights
    in_channels=3,                  # Input channels (RGB images)
    classes=1                       # Output channels (number of classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.sigmoid(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.sigmoid(output)
            test_loss += criterion(output, target).item()
    
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

def visualize_predictions(model, device, test_loader,  num_images=5, save_path=None):
    model.eval()
    data_iter = iter(test_loader)
    with torch.no_grad():
        for i in range(num_images):
            data, target = next(data_iter)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output > 0.5  # Apply threshold to get binary mask

            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(data[0].cpu().permute(1, 2, 0))
            plt.title('Input Image')

            plt.subplot(1, 3, 2)
            plt.imshow(target[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('True Mask')

            plt.subplot(1, 3, 3)
            plt.imshow(output[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Predicted Mask')
        
        # Save the figure if save_path is provided
            if save_path:
                plt.savefig(f"{save_path}/image_{i+1}.png")

torch.save(model.state_dict(), 'UNet_Imagenet.pth')
evaluate(model, device, test_loader)
visualize_predictions(model, device, test_loader, num_images=5, save_path='saved_images_imagenet')
