
import torch
import torch.nn as nn

from utils import *
from cores import *

class LSTM_proj(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.nF = out_channels
        self.hidden_size = mid_channels
        self.proj_size = 1
        self.num_layers = 3
        self.lstm = nn.LSTM(input_size=in_channels+1, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True,
                            proj_size=self.proj_size)

    
    def forward(self, x):
        #
        batch_num = x.shape[0]
        x = x.unsqueeze(1).repeat(1, self.nF, 1)
        f = torch.linspace(0, 1, self.nF).reshape(1, self.nF, 1).repeat(batch_num, 1, 1).to(x.device)
        x = torch.cat([f, x], dim=-1)
        h0 = torch.zeros(self.num_layers, batch_num, self.proj_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_num, self.hidden_size).to(x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return torch.unsqueeze(torch.squeeze(output, dim=-1), dim=1)
    
device = f"cuda:{0}"
x = torch.randn(50, 9).to(torch.device(device))
lstm = LSTM_proj(9, 64, 501).to(torch.device(device))

y = lstm(x)
print(y.shape)