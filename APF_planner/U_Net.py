import torch
import torch.nn as nn
from torchsummary import summary


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


if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=1)
    summary(model, (3, 256, 256))
