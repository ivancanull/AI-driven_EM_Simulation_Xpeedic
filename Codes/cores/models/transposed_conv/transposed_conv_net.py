from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dense.resnet import ResNet

class TransposedConvNetUpsample(nn.Module):

    def __init__(self, in_features, layers, out_features, out_channels):
        
        super().__init__()

        assert len(layers) == 3
        assert in_features % 32 == 0

        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.reshape = nn.Unflatten(-1, (-1, 32))
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.in_features // 32, 
                out_channels=layers[0], 
                kernel_size=10, 
                stride=3, 
                output_padding=0
            ),
            nn.SiLU(),
            nn.ConvTranspose1d(
                in_channels=layers[0], 
                out_channels=layers[1], 
                kernel_size=32, 
                stride=2, 
                output_padding=0
            ),
            nn.SiLU(),
            nn.ConvTranspose1d(
                in_channels=layers[1], 
                out_channels=layers[2], 
                kernel_size=32, 
                stride=2, 
                output_padding=0
            ),
            nn.Conv1d(layers[2], self.out_channels, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.reshape(x)
        x = self.upsample(x)
        x = x[..., :self.out_features]
        return x

# layers definition: [backbone_out_features, backbone_hidden_features, upsample_channel_0, upsample_channel_1, upsample_channel_2, upsample_channel_3]
# for example: [1024, 1024, 512, 256, 128, 64]

class TransposedConvNet(nn.Module):
    
    def __init__(self, in_features, layers, out_features, out_channels, multiport=False):
        
        super().__init__()

        assert len(layers) == 5

        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.multiport = multiport

        self.backbone = ResNet(
            in_features=self.in_features,
            out_features=layers[1],
            layers=[layers[0] for _ in range(3)]
        )
        if self.multiport:
            # multiply every item in layers[2:] by self.out_channels
            for i in range(2, len(layers)):
                layers[i] *= self.out_channels

        self.transposed_conv1d = TransposedConvNetUpsample(
            in_features=layers[1],
            layers=layers[2:],
            out_features=self.out_features,
            out_channels=self.out_channels * 2 if self.multiport else self.out_channels
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.transposed_conv1d(x)
        if not self.multiport:
            x = torch.flatten(x, start_dim=1, end_dim=-1)
        else:
            x = torch.reshape(x, (x.shape[0], self.out_channels, self.out_features * 2))
        return x
