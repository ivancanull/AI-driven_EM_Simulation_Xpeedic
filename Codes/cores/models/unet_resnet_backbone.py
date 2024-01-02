import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dense.resnet import residual
__all__ = ["unet_resnet_backbone", 
           "unet_resnet_connect",
           "unet_fourier"]
class double_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
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

class unet_resnet_backbone(nn.Module):
    
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
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
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
        self.out = outconv(mid_channels[5], 2, self.out_channels)
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.upsample(self.unflatten(x))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

class unet_resnet_connect(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        assert len(mid_channels) == 6

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.kernel_size = math.floor(self.out_channels ** 1 / 3)
        self.transconv3_shape = self.out_channels // 2
        self.transconv2_shape = self.transconv3_shape // 2
        self.transconv1_shape = self.transconv2_shape // 2
        
        self.backbone = nn.Sequential(
            nn.Linear(self.in_channels, mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[1]*(1+2+4+8)),
            nn.BatchNorm1d(mid_channels[1]*(1+2+4+8)),
        )
        
        self.unflatten = nn.Unflatten(-1, (-1, 1))
        # self.upsample = nn.Upsample(scale_factor=self.transconv1_shape)
        self.upsample_1 = nn.ConvTranspose1d(
            in_channels=mid_channels[1]*8, out_channels=mid_channels[2], kernel_size=self.transconv1_shape, stride=2)
        self.upsample_2 = nn.ConvTranspose1d(
            in_channels=mid_channels[1]*4, out_channels=mid_channels[3] // 2, kernel_size=self.transconv2_shape, stride=2)
        self.upsample_3 = nn.ConvTranspose1d(
            in_channels=mid_channels[1]*2, out_channels=mid_channels[4] // 2, kernel_size=self.transconv3_shape, stride=2)
        self.upsample_4 = nn.ConvTranspose1d(
            in_channels=mid_channels[1], out_channels=mid_channels[5] // 2, kernel_size=self.out_channels, stride=2)
        

        self.up1 = up(mid_channels[2], mid_channels[3] // 2, self.transconv2_shape)
        self.up2 = up(mid_channels[3], mid_channels[4] // 2, self.transconv3_shape)
        self.up3 = up(mid_channels[4], mid_channels[5] // 2, self.out_channels)
        self.out = outconv(mid_channels[5], 2, self.out_channels)
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.unflatten(x)
        x1 = x[:, 0:self.mid_channels[1]*8, :]
        x2 = x[:, self.mid_channels[1]*8:self.mid_channels[1]*12, :]
        x3 = x[:, self.mid_channels[1]*12:self.mid_channels[1]*14, :]
        x4 = x[:, self.mid_channels[1]*14:self.mid_channels[1]*15, :]

        x1 = self.upsample_1(x1)
        x2 = self.upsample_2(x2)
        x2 = torch.cat((self.up1(x1), x2), dim=1)
        x3 = self.upsample_3(x3)
        x3 = torch.cat((self.up2(x2), x3), dim=1)
        x4 = self.upsample_4(x4)
        x4 = torch.cat((self.up3(x3), x4), dim=1)
        x4 = self.out(x4)
        return x4
    

class up_fourier(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, out_shape, kernel_size):
        super().__init__()
        self.out_shape = out_shape
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)
        self.conv = double_conv(out_channels, out_channels)
    
    def forward(self, x):
        x = self.up(x)
        diff = self.out_shape - x.size()[-1]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        return self.conv(x)
    
class unet_fourier(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_sizes):
        super().__init__()

        assert len(mid_channels) == 7

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = math.floor(self.out_channels ** 1 / 3)
        self.transconv4_shape = self.out_channels // kernel_sizes[0]
        self.transconv3_shape = self.transconv4_shape // kernel_sizes[1]
        self.transconv2_shape = self.transconv3_shape // kernel_sizes[2]
        self.transconv1_shape = self.transconv2_shape // kernel_sizes[3]
        
        self.backbone = nn.Sequential(
            nn.Linear(self.in_channels, mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            residual(mid_channels[0], mid_channels[0]),
            nn.BatchNorm1d(mid_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels[0], mid_channels[1]),
        )
        
        self.unflatten = nn.Unflatten(-1, (-1, 1))
        # self.upsample = nn.Upsample(scale_factor=self.transconv1_shape)
        self.upsample = nn.ConvTranspose1d(
            in_channels=mid_channels[1], out_channels=mid_channels[2], kernel_size=self.transconv1_shape, stride=2)
        
        self.up1 = up_fourier(mid_channels[2], mid_channels[3], self.transconv2_shape, kernel_sizes[3])
        self.up2 = up_fourier(mid_channels[3], mid_channels[4], self.transconv3_shape, kernel_sizes[2])
        self.up3 = up_fourier(mid_channels[4], mid_channels[5], self.transconv4_shape, kernel_sizes[1])
        self.up4 = up_fourier(mid_channels[5], mid_channels[6], self.out_channels, kernel_sizes[0])
        self.out_conv = outconv(mid_channels[6], 1, self.out_channels)
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.upsample(self.unflatten(x))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.out_conv(x)
        x = torch.squeeze(x, dim=1)
        # x = torch.fft.rfft(x)
        # x = torch.cat([torch.real(x), torch.imag(x)], dim=1)
        return x
