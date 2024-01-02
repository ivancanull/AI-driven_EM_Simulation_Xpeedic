import torch
import torch.nn as nn

from prettytable import PrettyTable
__all__ = [
    "initialize_weights",
    "count_parameters",
    "calculate_Lout"
]

def _is_layer(m):
    if isinstance(m, nn.Linear):
        return True
    elif isinstance(m, nn.ConvTranspose1d):
        return True
    elif isinstance(m, nn.Conv1d):
        return True
    else:
        return False

def initialize_weights(self):
    for m in self.modules():
        if _is_layer(m):
            torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)

def count_parameters(model):
    
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def calculate_Lout(
    Lin,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    output_padding=0
):
    Lout = (Lin - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return Lout