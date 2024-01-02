
import json
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
__all__ = ["CustomDataset",
           "NonlinearDataset"]




# This dataset will read the dataframe with only SR, SI
class CustomDataset(Dataset):
    def __init__(
        self, 
        df, 
        input_cols, 
        output_cols, 
        nF,
        indices, 
        config_file,
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
        output_shape = (-1, 2, nF)
        
        cols = []
        # expand output cols
        for i in range(2):
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

        self.parameters = torch.Tensor(parameters.to_numpy().reshape(parameters_shape)).type(torch.float32).to(device)
        self.output = torch.Tensor(output.to_numpy().reshape(output_shape)).type(torch.float32).to(device)

        self.device = device
        self.max = torch.Tensor
        self.min = torch.Tensor
        return
    
    def __len__(self):
        return self.parameters.shape[0]

    def __getitem__(self, idx):
        return self.parameters[idx, ...], self.output[idx, ...]

    def get_max_min(self):
        self.max, _ = torch.max(self.output, dim=0, keepdim=True)
        self.max, _ = torch.max(self.max, dim=2, keepdim=True)
        self.min, _ = torch.min(self.output, dim=0, keepdim=True)
        self.min, _ = torch.min(self.min, dim=2, keepdim=True)

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

    def move_nonlinear_features_first(
        self,
        nonlinear_params,
        indices,
        df,
    ):
        # Locate the nonlinear parameters
        nonlinear_ids = []
        rest_ids = []

        if indices is not None:
            dfp = df.loc[indices]
        else:
            dfp = df

        for col in self.input_cols:
            loc = dfp.loc[:, self.input_cols].columns.get_loc(col)
            if col in nonlinear_params:
                nonlinear_ids.append(loc)
            else:
                rest_ids.append(loc)

        self.parameters = torch.cat((self.parameters[:, nonlinear_ids], self.parameters[:, rest_ids]), dim=1)

    def upsample_input(
        self,
        upsample_ratio: int,
        expand_dim: bool = False,
    ):
        # upsample_ratio must be an odd number
        assert (upsample_ratio % 2 == 1)
        if expand_dim == False:
            upsampled_input_tensor = torch.zeros((self.__len__(), len(self.input_cols) * upsample_ratio)).to(self.device)
            upsampled_input_tensor[:, 0:len(self.input_cols)] = self.parameters
        else:
            upsampled_input_tensor = torch.zeros((self.__len__(), len(self.input_cols), upsample_ratio)).to(self.device)
            upsampled_input_tensor[:, :, 0] = self.parameters
        
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

    def downsample_input(self, 
        variable_num, 
        X,
        expand_dim: bool = False
    ):

        if expand_dim == False:
            return X[:, 0: variable_num]
        else:
            return X[:, :, 0]

class NonlinearDataset(CustomDataset):

    def __init__(self, df, input_cols, output_cols, nF, indices, config_file, device):
        super().__init__(df, input_cols, output_cols, nF, indices, config_file, device)

        # Locate the nonlinear parameters
        nonlinear_params = ['H1', 'H2']
        nonlinear_ids = []
        rest_ids = []

        if indices is not None:
            dfp = df.loc[indices]
        else:
            dfp = df

        for col in self.input_cols:
            loc = dfp.loc[:, input_cols].columns.get_loc(col)
            if col in nonlinear_params:
                nonlinear_ids.append(loc)
            else:
                rest_ids.append(loc)

        self.parameters = torch.cat((self.parameters[:, nonlinear_ids], self.parameters[:, rest_ids]), dim=1)
        
        return