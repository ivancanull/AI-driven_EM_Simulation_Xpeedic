from typing import Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
# from torchvision.models.resnet import BasicBlock, Bottleneck
from torch import Tensor

class residual(nn.Module):  
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.SiLU(),
            nn.Linear(mid_channels, in_channels),
        )

    def forward(self, x):
        return self.layers(x) + x

class ResNet(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 layers: List[int]):
        
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        layers_list = [nn.Linear(self.in_features, layers[0])]

        for i in range(len(layers) - 1):
            layers_list += [
                residual(layers[i], layers[i]),
                nn.SiLU()
            ]
        self.fc = nn.Sequential(*layers_list)
        self.out_layer = nn.Linear(layers[-1], out_features)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.out_layer(x)
        return x

class ResNet_SRSI(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 trunk_layers: List[int],
                 head_layers: List[int],

                 ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # trunk layers
        
        self.trunk = ResNet(in_features, trunk_layers[-1], trunk_layers[:-1])
        self.head_sr = ResNet(trunk_layers[-1], out_features, head_layers)
        self.head_si = ResNet(trunk_layers[-1], out_features, head_layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.trunk(x)
        y1 = self.head_sr(x)
        y2 = self.head_si(x)
        return torch.concat([y1, y2], dim=1)