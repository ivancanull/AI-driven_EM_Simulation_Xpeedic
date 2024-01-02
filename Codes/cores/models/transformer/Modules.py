import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        output = torch.matmul(attn, v)

        return output, attn

if __name__ == "__main__":
    # generate tensor example
    batch_size = 32
    in_channels = 10
    out_channels = 500

    device = torch.device("cuda:0")

    X = torch.rand([batch_size, in_channels]).to(device)

    lstm = ScaledDotProductAttention(in_channels, [32, 32], out_channels, proj_size=1).to(device)
