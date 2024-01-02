# multilayer_dataset
# read input features pattern and location
# model: 
#    layer 1 - 2
#    layer 1 - 3
#    layer 1 - 4
#    layer 1 - 5
#    ...


from builtins import ValueError
from unittest import case
import numpy as np
import pandas as pd

from utils import *
from .multilayer_input_features import *
import json
import os
import torch
from torch.utils.data import Dataset
from typing import List
from ..samplers.sample_optimizer import *
supported_metis_variables = ["W", "S", "H", "T"]

class MultiLayerDataset(Dataset):

    def __init__(self,
                 configs: Config,
                 df_list: List[pd.DataFrame],
                 input_features: MultiLayerInputFeatures,
                 output_col: str,
                 stackups_list: List[List[Stackup]],
                 device: torch.device,
                 unmasked_col: List[int] = None,
                 indices = None,
                 ) -> None:
                
        self.input_features = input_features
        # support multiports
        for i in range(len(stackups_list)):
            if len(stackups_list[i]) != df_list[i].shape[0]:
                raise ValueError('The number of stackups does not match the number of data points')
        # initialize parameters tensor
        nF = self.input_features.nF
        self.parameters = torch.Tensor()
        self.output = torch.Tensor()
        for dataset_idx, stackups in enumerate(stackups_list):
            parameters = torch.zeros([len(stackups), len(input_features.sampled_variables)], dtype=torch.float32)
            
            df = df_list[dataset_idx]
            for i in range(df.shape[0]):
                df_idx = df.index[i]
                # calculate stackup index from dataframe idx (e.g. 0_0, 0_1, 0_2, 1_0, 1_1, 1_2)
                # old version of stack index defination: 
                # # stack_idx = int(idx.split('_')[-2]) * configs.dataset_generation.batch_num + int(idx.split('_')[-1]) - 1
                # new version of stack index defination:
                
                stack_idx = int(df_idx.split('_')[-1]) - 1
                for v_i, v in enumerate(input_features.sampled_variables):
                    if v == "W":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].w_list[0]
                    elif v == "S":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].spaces[0]
                    elif v == "H":
                        parameters[i, v_i] = stackups[stack_idx].layers[4].thickness
                    elif v == "T":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].thickness
                    elif v == "Er":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].er
                    elif v == "Loss":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].loss
                    elif v.split('_')[0] == "W":
                        # support differentline mode: same_column, center_aligned
                        if hasattr(configs, 'differentline'):
                            layer_index = int(v.split('_')[-1])
                            # if specific column, use the column of the port related w and mask the other columns
                            if hasattr(configs.datasets, 'column_specific'):
                                # the layer index of the port related w should not be masked
                                if unmasked_col is None:
                                    raise ValueError("Unmasked column list cannot be empty")
                                if configs.datasets.column_specific and layer_index not in unmasked_col:
                                    # mask the parameter
                                    parameters[i, v_i] = 0
                                else:
                                    parameters[i, v_i] = stackups[stack_idx].layers[3].w_list[layer_index * 5 - 3]
                        else:
                            layer_index = int(v.split('_')[-1])
                            parameters[i, v_i] = stackups[stack_idx].layers[1 + layer_index * 2].w_list[0]
                    elif v.split('_')[0] == "H":
                        layer_index = int(v.split('_')[-1])
                        parameters[i, v_i] = stackups[stack_idx].layers[layer_index * 2].thickness
                    elif v.split('_')[0] == "T":
                        layer_index = int(v.split('_')[-1])
                        parameters[i, v_i] = stackups[stack_idx].layers[1 + layer_index * 2].thickness
                    elif v.split('_')[0] == "Er":
                        layer_index = int(v.split('_')[-1])
                        parameters[i, v_i] = stackups[stack_idx].layers[1 + layer_index].er
                    elif v.split('_')[0] == "Loss":
                        layer_index = int(v.split('_')[-1])
                        parameters[i, v_i] = stackups[stack_idx].layers[1 + layer_index].loss
                    else:
                        raise ValueError(f"Unsupported variable type {v}")

            self.parameters = torch.cat([self.parameters, parameters], dim=0)
            self.output = torch.cat([self.output, torch.Tensor(df.to_numpy()).type(torch.float32)])
            
        self.parameters = self.parameters.to(device)
        self.output = self.output.to(device)
        if 'SR' in output_col:
            self.output = self.output[:, 0:nF] 
        elif 'SI' in output_col:
            self.output = self.output[:, nF:2*nF] 
        
        # define min and max for normalization
        config_tensor_min = torch.zeros([1, len(input_features.sampled_variables)])
        config_tensor_max = torch.zeros([1, len(input_features.sampled_variables)])
        # normalize method: Min-Max
        for i, v in enumerate(self.input_features.sampled_variables):
            if getattr(self.input_features, v).is_list():
                config_tensor_min[0, i] = min(getattr(self.input_features, v).min)
                config_tensor_max[0, i] = max(getattr(self.input_features, v).max)
            else:
                config_tensor_min[0, i] = getattr(self.input_features, v).min
                config_tensor_max[0, i] = getattr(self.input_features, v).max

        self.config_tensor_min = torch.Tensor(config_tensor_min).to(device)
        self.config_tensor_max = torch.Tensor(config_tensor_max).to(device)

        self.parameters = (self.parameters - self.config_tensor_min) / (self.config_tensor_max - self.config_tensor_min)
        self.device = device

    def sample_split(self, sample_num, trial_num = 20):

        if sample_num > self.__len__():
            raise ValueError("Sample number is larger than the dataset size.")

        # get original features from normalized features
        original_features_np = self.denormalize_features(self.parameters).cpu().detach().numpy()
        
        sample_optimizer = SampleOptimizer(self.input_features)
        train_ids = None
        max_uniformity = 0
        for i in range(trial_num):
            indices = np.random.choice(self.__len__(), sample_num, replace=False)
            sampled_parameters = original_features_np[indices, :]
            range_uniformity = sample_optimizer.range_uniformity(sampled_parameters)
            if range_uniformity > max_uniformity:
                max_uniformity = range_uniformity
                train_ids = indices

        val_ids = np.setdiff1d(np.arange(self.__len__()), train_ids)
        return train_ids, val_ids

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
        return output * (self.max - self.min) + self.min
    
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

class MultiLayerCombinedColumnDataset(MultiLayerDataset):
    def __init__(self,
                 configs: Config,
                 df_list: List[pd.DataFrame],
                 input_features: MultiLayerInputFeatures,
                 output_col: str,
                 stackups_list: List[List[Stackup]],
                 device: torch.device,
                 unmasked_cols: List[List[int]] = None,
                 port_per_layer: int = 10,
                 indices = None,
                 ) -> None:
                
        self.input_features = input_features
        # support multiports
        for i in range(len(stackups_list)):
            if len(stackups_list[i]) != df_list[i].shape[0]:
                raise ValueError('The number of stackups does not match the number of data points')
        # initialize parameters tensor
        nF = self.input_features.nF
        self.parameters = torch.Tensor()
        self.output = torch.Tensor()
        
        W_list = [f"W_{i+1}" for i in range(port_per_layer)]
        if all(W_i in input_features.sampled_variables for W_i in W_list):
            num_variables = len(input_features.sampled_variables) - port_per_layer + 2
        else:
            raise ValueError("The input features must contain W_1, W_2, ..., W_10")
        
        for dataset_idx, stackups in enumerate(stackups_list): 

            parameters = torch.zeros([len(stackups), num_variables], dtype=torch.float32)
            df = df_list[dataset_idx]
            unmasked_col = unmasked_cols[dataset_idx]

            for i in range(df.shape[0]):
                df_idx = df.index[i]
                stack_idx = int(df_idx.split('_')[-1]) - 1
                # first load the port related w
                parameters[i, 0] = stackups[stack_idx].layers[3].w_list[unmasked_col[0] * 5 - 3]
                parameters[i, 1] = stackups[stack_idx].layers[3].w_list[unmasked_col[1] * 5 - 3]
                v_i = 2
                for v in input_features.sampled_variables:
                    if v == "W":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].w_list[0]
                    elif v == "S":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].spaces[0]
                    elif v == "H":
                        parameters[i, v_i] = stackups[stack_idx].layers[4].thickness
                    elif v == "T":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].thickness
                    elif v == "Er":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].er
                    elif v == "Loss":
                        parameters[i, v_i] = stackups[stack_idx].layers[3].loss
                    elif v.split('_')[0] == "W":
                        # support differentline mode: same_column, center_aligned
                        if hasattr(configs, 'differentline'):
                            if configs.differentline != 'center_aligned':
                                raise ValueError("Only support center_aligned mode for now.")
                        else:
                            raise ValueError("Only support differentline mode for now.")
                    elif v.split('_')[0] == "H":
                        layer_index = int(v.split('_')[-1])
                        parameters[i, v_i] = stackups[stack_idx].layers[layer_index * 2].thickness
                    elif v.split('_')[0] == "T":
                        layer_index = int(v.split('_')[-1])
                        parameters[i, v_i] = stackups[stack_idx].layers[1 + layer_index * 2].thickness
                    elif v.split('_')[0] == "Er":
                        layer_index = int(v.split('_')[-1])
                        parameters[i, v_i] = stackups[stack_idx].layers[1 + layer_index].er
                    elif v.split('_')[0] == "Loss":
                        layer_index = int(v.split('_')[-1])
                        parameters[i, v_i] = stackups[stack_idx].layers[1 + layer_index].loss
                    else:
                        raise ValueError(f"Unsupported variable type {v}")
                    
                    if not (v.split('_')[0] == "W" and v != "W"):
                        v_i += 1

            self.parameters = torch.cat([self.parameters, parameters], dim=0)
            self.output = torch.cat([self.output, torch.Tensor(df.to_numpy()).type(torch.float32)])
            
        self.parameters = self.parameters.to(device)
        self.output = self.output.to(device)
        
        # define min and max for normalization
        config_tensor_min = torch.zeros([1, num_variables])
        config_tensor_max = torch.zeros([1, num_variables])

        # normalize method: Min-Max
        v_i = 2
        min_W = 10000
        max_W = 0
        for W in W_list:
            if getattr(self.input_features, W).is_list():
                min_W = min(min_W, min(getattr(self.input_features, W).min))
                max_W = max(max_W, max(getattr(self.input_features, W).max))
            else:
                min_W = min(min_W, getattr(self.input_features, W).min)
                max_W = max(max_W, getattr(self.input_features, W).max)
        config_tensor_min[0, 0] = min_W
        config_tensor_max[0, 0] = max_W
        config_tensor_min[0, 1] = min_W
        config_tensor_max[0, 1] = max_W

        for v in self.input_features.sampled_variables:
            if not (v.split('_')[0] == "W" and v != "W"):
                if getattr(self.input_features, v).is_list():
                    config_tensor_min[0, v_i] = min(getattr(self.input_features, v).min)
                    config_tensor_max[0, v_i] = max(getattr(self.input_features, v).max)
                else:
                    config_tensor_min[0, v_i] = getattr(self.input_features, v).min
                    config_tensor_max[0, v_i] = getattr(self.input_features, v).max
                v_i += 1

        self.config_tensor_min = torch.Tensor(config_tensor_min).to(device)
        self.config_tensor_max = torch.Tensor(config_tensor_max).to(device)

        self.parameters = (self.parameters - self.config_tensor_min) / (self.config_tensor_max - self.config_tensor_min)
        self.device = device
        
    def sample_split(self, sample_num, trial_num = 20):

        if sample_num > self.__len__():
            raise ValueError("Sample number is larger than the dataset size.")

        # get original features from normalized features
        original_features_np = self.denormalize_features(self.parameters).cpu().detach().numpy()
        
        sample_optimizer = SampleOptimizer(self.input_features)
        train_ids = None
        max_uniformity = 0
        for i in range(trial_num):
            indices = np.random.choice(self.__len__(), sample_num, replace=False)
            sampled_parameters = original_features_np[indices, :]
            range_uniformity = sample_optimizer.range_uniformity(sampled_parameters)
            if range_uniformity > max_uniformity:
                max_uniformity = range_uniformity
                train_ids = indices

        val_ids = np.setdiff1d(np.arange(self.__len__()), train_ids)
        return train_ids, val_ids

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
