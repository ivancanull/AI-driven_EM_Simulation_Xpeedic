from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn

from torch import Tensor

class MLP(nn.Module):
    """
    This implements a simple MLP with Swish activation function.

    Attributes:
        in_channels: int
        out_channels: int
        layers: List[int] for example, [2048, 2048, 2048]
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 layers: List[int]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        layers = [self.in_features] + layers

        layers_list = []
        for i in range(len(layers) - 1):
            layers_list += [
                nn.Linear(layers[i], layers[i+1]),
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

class MLP_SRSI(nn.Module):
    """
    This implements a two-channel MLP with Swish activation function.

    Attributes:
        in_channels: int
        out_channels: int
        layers: List[int] for example, [2048, 2048, 2048]
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 trunk_layers: List[int],
                 leaf_layers: List[int]):
        

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # trunk layers
        
        self.trunk = MLP(in_features, trunk_layers[-1], trunk_layers[:-1])
        self.leaf_sr = MLP(trunk_layers[-1], out_features, leaf_layers)
        self.leaf_si = MLP(trunk_layers[-1], out_features, leaf_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.trunk(x)
        y1 = self.leaf_sr(x)
        y2 = self.leaf_si(x)
        return torch.concat([y1, y2], dim=1)