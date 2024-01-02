import numpy as np
import pandas as pd

from utils import *
from .input_features import *
from .base_dataset import BaseDataset
import json
import os
import torch

class FDataset(BaseDataset):
    """
    This is a dataset which puts f as an input.

    Attributes:
        configs: Config
        df: pd.DataFrame of dataset
        input_features: InputFeatures
        output_col: output column name, such as SR(1,1)
        indices: indices of input dataframe 
    
    Returns:
        parameters: Tensors (N, Cin)
        output: Tensors (N, Cout = nF)
    """

    def __init__(self, 
                 configs: Config, 
                 df: pd.DataFrame, 
                 input_features: InputFeatures, 
                 output_col: str, 
                 device: torch.device, 
                 indices=None) -> None:
        
        self.input_features = input_features

        if indices is not None:
            dfp = df.loc[indices]
        else:
            dfp = df

        # Define input and output shape
        dfp_num = dfp.shape[0]
        parameters_shape = (dfp_num * self.input_features.nF, self.input_features.variable_num + 1)
        output_shape = (dfp_num * self.input_features.nF, 1)
        
        parameters = dfp.loc[:, self.input_features.sampled_variables]

        with open(os.path.join(os.getcwd(), configs.config.dir)) as f:
            config = json.load(f)

        config_tensor_min = []
        config_tensor_max = []

        # Normalize method: Min-Max
        for input_col in self.input_features.sampled_variables:
            parameters.loc[:, input_col] = (parameters[input_col] - config[input_col]['min']) / (config[input_col]['max'] - config[input_col]['min'])
            config_tensor_min.append(config[input_col]['min'])
            config_tensor_max.append(config[input_col]['max'])

        parameters_np = np.zeros(shape=parameters_shape, dtype=float)
        parameters_np[:, 0:self.input_features.variable_num] = np.tile(parameters.to_numpy(), (self.input_features.nF, 1)).reshape(dfp_num * self.input_features.nF, -1)
        parameters_np[:, -1:] = np.tile(self.input_features.frequency_np.reshape(-1, 1), (dfp_num, 1))

        fmin = self.input_features.frequency_np[0]
        fmax = self.input_features.frequency_np[-1]
        config_tensor_min.append(fmin)
        config_tensor_max.append(fmax)
        parameters_np[:, -1:] = (parameters_np[:, -1:] - fmin) / (fmax - fmin)

        cols = []
        # Expand output cols
        for i in range(self.input_features.nF):
            cols.append(output_col+'_%d'%i)
        
        # Extract parameters
        output = dfp.loc[: , cols]
        output_np = output.to_numpy().reshape(-1, 1)
        
        self.config_tensor_min = torch.Tensor(config_tensor_min)
        self.config_tensor_max = torch.Tensor(config_tensor_max)

        self.parameters = torch.Tensor(parameters_np).type(torch.float32).to(device)
        self.output = torch.Tensor(output_np).type(torch.float32).to(device)

        self.device = device
    