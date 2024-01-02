import numpy as np
import pandas as pd

from utils import *
from .input_features import *
import json
import os
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    This is a base class for samplers.

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
                 indices = None) -> None:
        super().__init__()

        self.input_features = input_features

        if indices is not None:
            dfp = df.loc[indices]
        else:
            dfp = df

        # Define input and output shape
        parameters_shape = (-1, self.input_features.variable_num)
        output_shape = (-1, self.input_features.nF)
        cols = []
        
        # Expand output cols
        for i in range(self.input_features.nF):
            cols.append(output_col+'_%d'%i)
        parameters = dfp.loc[:, self.input_features.sampled_variables]
        
        # Extract parameters
        with open(os.path.join(os.getcwd(), configs.config.dir)) as f:
            config = json.load(f)
        
        config_tensor_min = []
        config_tensor_max = []

        # Normalize method: Min-Max
        for input_col in self.input_features.sampled_variables:
            parameters.loc[:, input_col] = (parameters[input_col] - config[input_col]['min']) / (config[input_col]['max'] - config[input_col]['min'])
            config_tensor_min.append(config[input_col]['min'])
            config_tensor_max.append(config[input_col]['max'])

        output = dfp.loc[: , cols]
        
        self.config_tensor_min = torch.Tensor(config_tensor_min)
        self.config_tensor_max = torch.Tensor(config_tensor_max)

        self.parameters = torch.Tensor(parameters.to_numpy().reshape(parameters_shape)).type(torch.float32).to(device)
        self.output = torch.Tensor(output.to_numpy().reshape(output_shape)).type(torch.float32).to(device)

        self.device = device
    
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
        return output * (self.max - self.min) * output + self.min
    
    def denormalize_features(self, features):
        feature_max = self.config_tensor_max.reshape(1, -1)
        feature_min = self.config_tensor_min.reshape(1, -1)
        return features * (feature_max - feature_min) + feature_min

    def set_max_min(self, max_tensor: torch.Tensor, min_tensor: torch.Tensor):
        self.max = torch.clone(max_tensor.cpu()).to(self.device)
        self.min = torch.clone(min_tensor.cpu()).to(self.device)
        return
    
    def get_original_parameters(
        self,
        X,
        device: torch.device
    ):
        config_tensor_max = self.config_tensor_max.to(device)
        config_tensor_min = self.config_tensor_min.to(device)
        return (X * (config_tensor_max - config_tensor_min) + config_tensor_min)

    def upsample_input(
        self,
        upsample_ratio: int,
        expand_dim: bool = False,
    ):
        """
        Upsample the input to get better results.
        
        
        """
        # upsample_ratio must be an odd number
        assert (upsample_ratio % 2 == 1)
        if expand_dim == False:
            upsampled_input_tensor = torch.zeros((self.__len__(), len(self.input_cols) * upsample_ratio)).to(self.device)
            upsampled_input_tensor[:, 0:len(self.input_cols)] = self.parameters
        else:
            upsampled_input_tensor = torch.zeros((self.__len__(), len(self.input_cols), upsample_ratio)).to(self.device)
            upsampled_input_tensor[:, 0] = self.parameters
        
        # for i in range(0, upsample_ratio // 2):
        #     if expand_dim == False:
        #         upsampled_input_tensor[:, (i*2+1) * len(self.input_cols): (i*2+2) * len(self.input_cols)] = torch.cos(self.parameters * (i+1))
        #         upsampled_input_tensor[:, (i*2+2) * len(self.input_cols): (i*2+3) * len(self.input_cols)] = torch.sin(self.parameters * (i+1))
        #     else:
        #         upsampled_input_tensor[:, :, i*2+1] = torch.cos(self.parameters * (i+1))
        #         upsampled_input_tensor[:, :, i*2+2] = torch.sin(self.parameters * (i+1))
        
        input_num = len(self.input_cols)

        for c in range(input_num):
            if expand_dim == False:
                upsampled_input_tensor[:, c*(upsample_ratio)] = self.parameters[:, c]
                for i in range(0, upsample_ratio // 2):
                    upsampled_input_tensor[:, c*(upsample_ratio) + i*2 + 1] = torch.cos(self.parameters[:, c])
                    upsampled_input_tensor[:, c*(upsample_ratio) + i*2 + 2] = torch.sin(self.parameters[:, c])

            else:
                raise NotImplementedError
        self.parameters = upsampled_input_tensor
        return

    def downsample_input(
        self, 
        variable_num, 
        X,
        expand_dim: bool = False
    ):

        if expand_dim == False:
            return X[:, 0: variable_num]
        else:
            return X[:, :, 0]
