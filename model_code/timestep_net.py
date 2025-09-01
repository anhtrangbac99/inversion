import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model_code.unet import UNetModel
from model_code.abcfusenet import ABSFusenet

class ConvEncoder(nn.Module):
    def __init__(self, in_ch=3, out_dim=256):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3), nn.ReLU(inplace=True),  # 64x64 -> 32x32
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),  # 32x32 -> 16x16
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16x16 -> 8x8
            nn.AdaptiveAvgPool2d(1)  # -> [B,256,1,1]
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):                      # x: [B,3,64,64]
        h = self.feat(x).flatten(1)            # [B,256]
        z = self.proj(h)                       # [B,out_dim]
        return F.normalize(z, dim=1)

class TSNet(nn.Module):
    def __init__(
        self,
        config):
        super().__init__()

        self.unet = UNetModel(config)
        self.fusenet = ABSFusenet()
        self.encoder = ConvEncoder()

    def forward(self, x, timesteps=None, unet=True,encoder=False,y=None):
        if unet:
            return self.unet(x,timesteps,y)
        elif encoder:
            return self.encoder(x)
        else:
            return self.fusenet(x)
