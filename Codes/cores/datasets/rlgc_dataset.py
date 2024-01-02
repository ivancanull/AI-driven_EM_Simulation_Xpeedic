
import json
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
__all__ = ["RLGCDataset", "to_rlgc_dataframe"]

class RLGCDataset(Dataset):
    def __init__(
        self, 
        df, 
        input_cols, 
        output_cols, 
        nF,
        indices, 
        config_file,
        mode,
        device
    ):
        """
        :param df: dataframe that contains the data
        :param input_cols: columns names of input features
        :param output_cols: columns of output features
        :param nF: number of frequency points
        :param indices: define rows to extract from the dataframe
        :param devive: device to locate the tensors
        :param out_features: output feature num, equals to frequency points
        """
        if indices is not None:
            dfp = df.loc[indices]
        else:
            dfp = df
        self.input_cols = input_cols

        parameters_shape = (-1, len(input_cols))
        col_num = len(output_cols)
        output_shape = (-1, col_num, nF)
        
        cols = []
        # expand output cols
        for i in range(col_num):
            for j in range(nF):
                cols.append(output_cols[i]+'_%d'%j)

        parameters = dfp.loc[:, input_cols]
        # get parameters
        with open(config_file) as f:
            config = json.load(f)
        
        config_tensor_min = []
        config_tensor_max = []

        for input_col in input_cols:
            parameters.loc[:, input_col] = (parameters[input_col] - config[input_col]['min']) / (config[input_col]['max'] - config[input_col]['min'])
            config_tensor_min.append(config[input_col]['min'])
            config_tensor_max.append(config[input_col]['max'])

        output = dfp.loc[: , cols]
        
        self.config_tensor_min = torch.Tensor(config_tensor_min)
        self.config_tensor_max = torch.Tensor(config_tensor_max)

        if not mode in ['R', 'L', 'G', 'C']:
            raise ValueError('Mode must be one of R, L , G, C!')
        
        self.parameters = torch.Tensor(parameters.to_numpy().reshape(parameters_shape)).type(torch.float32).to(device)
        if mode == 'L':
            self.output = torch.Tensor(output.to_numpy().reshape(output_shape)[..., 1:]).type(torch.float32).to(device)
        else:
            self.output = torch.Tensor(output.to_numpy().reshape(output_shape)).type(torch.float32).to(device)

        if mode == 'C':
            self.output = torch.mean(self.output, dim=-1, keepdim=True)
        elif mode == 'G':
            self.output = self.output[..., -1:]
        self.mode = mode
        self.device = device
        return

    def __len__(self):
        return self.parameters.shape[0]

    def __getitem__(self, idx):
        return self.parameters[idx, ...], self.output[idx, ...]

    def get_max_min(self):
        self.max = torch.max(self.output)
        self.min = torch.min(self.output)
        return self.max, self.min


    def normalize_output(self):
        self.output = (self.output - self.min) / (self.max - self.min)
        return
    
    def denormalize_output(self, output):
        output = (self.max - self.min) * output + self.min
        return output
    
    def denormalize_features(self, features):
        feature_max = self.config_tensor_max.reshape(1, -1)
        feature_min = self.config_tensor_min.reshape(1, -1)
        return features * (feature_max - feature_min) + feature_min

    def set_max_min(self, max_tensor: torch.Tensor, min_tensor: torch.Tensor = None):
        self.max = torch.clone(max_tensor.cpu()).to(self.device)
        self.min = torch.clone(min_tensor.cpu()).to(self.device)
        return

def to_rlgc_dataframe(
    cols,
    rlgc_tensor,
    df,
):
    
    print(rlgc_tensor)
    df.loc[:, cols] = torch.transpose(rlgc_tensor, 0, 1).numpy()
    return df