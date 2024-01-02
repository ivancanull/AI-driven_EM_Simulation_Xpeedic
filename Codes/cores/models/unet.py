import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)
    
    
class up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, out_shape):
        super().__init__()
        self.out_shape = out_shape
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.up(x)
        diff = self.out_shape - x.size()[-1]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x = self.conv(x)
        return x
    
class outconv(nn.Module):
    def __init__(self, in_channels, out_channels, out_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            # nn.ReLU(inplace=True),
            # nn.Linear(out_shape, out_shape)
        )
    def forward(self, x):
        return self.conv(x)  

class unet(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        assert len(mid_channels) == 6

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = math.floor(self.out_channels ** 1 / 3)
        self.transconv3_shape = self.out_channels // 2
        self.transconv2_shape = self.transconv3_shape // 2
        self.transconv1_shape = self.transconv2_shape // 2
        
        self.backbone = nn.Sequential(
            nn.Linear(self.in_channels, mid_channels[0]),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[0]),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[0]),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[0]),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[1]),
        )
        
        self.unflatten = nn.Unflatten(-1, (-1, 1))
        # self.upsample = nn.Upsample(scale_factor=self.transconv1_shape)
        self.upsample = nn.ConvTranspose1d(
            in_channels=mid_channels[1], out_channels=mid_channels[2], kernel_size=self.transconv1_shape, stride=2)
        
        self.up1 = up(mid_channels[2], mid_channels[3], self.transconv2_shape)
        self.up2 = up(mid_channels[3], mid_channels[4], self.transconv3_shape)
        self.up3 = up(mid_channels[4], mid_channels[5], self.out_channels)
        self.out = outconv(mid_channels[5], 1, self.out_channels)
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.upsample(self.unflatten(x))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        x = x.squeeze(dim=1)
        return x
    
class unet_pulse(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        assert len(mid_channels) == 6

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = math.floor(self.out_channels ** 1 / 3)
        self.transconv3_shape = self.out_channels // 2
        self.transconv2_shape = self.transconv3_shape // 2
        self.transconv1_shape = self.transconv2_shape // 2
        
        self.backbone = nn.Sequential(
            nn.Linear(self.in_channels, mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[1]),
            nn.BatchNorm1d(mid_channels[1]),
        )
        
        self.unflatten = nn.Unflatten(-1, (-1, 1))
        # self.upsample = nn.Upsample(scale_factor=self.transconv1_shape)
        self.upsample = nn.ConvTranspose1d(
            in_channels=mid_channels[1], out_channels=mid_channels[2], kernel_size=self.transconv1_shape, stride=2)
        
        self.up1 = up(mid_channels[2], mid_channels[3], self.transconv2_shape)
        self.up2 = up(mid_channels[3], mid_channels[4], self.transconv3_shape)
        self.up3 = up(mid_channels[4], mid_channels[5], self.out_channels)
        self.out = outconv(mid_channels[5], 1, self.out_channels)
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.upsample(self.unflatten(x))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        x = x.squeeze(dim=1)
        return x

class unet_RLGC(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        assert len(mid_channels) == 6

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = math.floor(self.out_channels ** 1 / 3)
        self.transconv3_shape = self.out_channels // 2
        self.transconv2_shape = self.transconv3_shape // 2
        self.transconv1_shape = self.transconv2_shape // 2
        
        self.backbone = nn.Sequential(
            nn.Linear(self.in_channels, mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[1]),
            nn.BatchNorm1d(mid_channels[1]),
        )
        
        self.unflatten = nn.Unflatten(-1, (-1, 1))
        # self.upsample = nn.Upsample(scale_factor=self.transconv1_shape)
        self.upsample = nn.ConvTranspose1d(
            in_channels=mid_channels[1], out_channels=mid_channels[2], kernel_size=self.transconv1_shape, stride=2)
        
        self.up1 = up(mid_channels[2], mid_channels[3], self.transconv2_shape)
        self.up2 = up(mid_channels[3], mid_channels[4], self.transconv3_shape)
        self.up3 = up(mid_channels[4], mid_channels[5], self.out_channels)
        self.out = outconv(mid_channels[5], 1, self.out_channels)
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.upsample(self.unflatten(x))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x