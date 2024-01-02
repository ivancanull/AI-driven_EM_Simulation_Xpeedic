import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, max_len=1000):
        super().__init__()
        # Create a long enough P
        self.P = torch.zeros((max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        return self.P[:, :X.shape[1], :].to(X.device)
    
class BaseLSTM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, proj_size=1):
        super().__init__()
        self.nF = out_channels
        self.hidden_size = mid_channels[0]
        if proj_size != 1:
            raise NotImplementedError
        self.proj_size = proj_size
        self.num_layers = 3
        self.lstm = nn.LSTM(input_size=in_channels+mid_channels[1], hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True,
                            proj_size=self.proj_size)
        
        self.f = torch.zeros((self.nF, mid_channels[1]))
        X = torch.arange(self.nF, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(self.nF, torch.arange(
            0, mid_channels[1], 2, dtype=torch.float32) / mid_channels[1])
        
        self.f[:, 0::2] = torch.sin(X)
        self.f[:, 1::2] = torch.cos(X)

    
    def forward(self, x):
        #
        batch_num = x.shape[0]
        x = x.unsqueeze(1).repeat(1, self.nF, 1)
        f = self.f.repeat(batch_num, 1, 1).to(x.device)
        # TODO: try different position encoding methods
        x = torch.cat([f, x], dim=-1)
        h0 = torch.zeros(self.num_layers, batch_num, self.proj_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_num, self.hidden_size).to(x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # return torch.swapaxes(output, -2, -1)
        return torch.squeeze(output, -1)

if __name__ == "__main__":

    # generate tensor example
    batch_size = 32
    in_channels = 10
    out_channels = 500

    device = torch.device("cuda:0")

    X = torch.rand([batch_size, in_channels]).to(device)
    lstm = BaseLSTM(in_channels, [32, 32], out_channels, proj_size=1).to(device)

    y = lstm(X)
    assert y.shape == torch.Size([batch_size, out_channels])