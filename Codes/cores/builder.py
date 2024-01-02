import torch
import torch.nn as nn
import pandas as pd

from typing import Tuple, List
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
from sklearn.model_selection import train_test_split

from .datasets import *
from .models import *
from .samplers import *
from utils import Config

import numpy as np
from .samplers import UniformSampler
from .datasets.multilayer_input_features import MultiLayerInputFeatures

def make_samples(
    input_features: MultiLayerInputFeatures,
    method: str, 
    sampling_num: int = 0,
    step_ratio: float = 0.5,
    trial_num: int = 50,
) -> np.ndarray:
    """
    Generate samples for the given input features using the specified sampling method.

    Args:
        input_features (MultiLayerInputFeatures): The input features to generate samples for.
        method (str): The sampling method to use. Currently only 'uniform' is supported.
        sampling_num (int): The number of samples to generate.

    Returns:
        np.ndarray: An array of shape (sampling_num, input_features.total_dim) containing the generated samples.
    """
    if method == 'uniform':
        sampler = UniformSampler(input_features)
        samples_np = sampler.sample(sample_num=sampling_num)
    elif method == 'guided_uniform':
        sampler = GuidedUniformSampler(input_features)
        samples_np = sampler.sample(sample_num=sampling_num, trial_num=trial_num)
    elif method == 'sweep':
        sampler = SweepSampler(input_features)
        samples_np = sampler.sample()
    elif method == 'ratio_sweep':
        sampler = RatioSweepSampler(input_features)
        samples_np = sampler.sample()
    elif method == 'test_sweep':
        sampler = TestSweepSampler(input_features, step_ratio)
        samples_np, X, Y, wn, sn = sampler.sample()
        return samples_np, X, Y, wn, sn
    else:
        raise NotImplementedError
    
    return samples_np

def make_sampler(
    input_features: MultiLayerInputFeatures,
    method: str, 
    step_ratio: float = 0.5,
):
    """
    Generate samples for the given input features using the specified sampling method.

    Args:
        input_features (MultiLayerInputFeatures): The input features to generate samples for.
        method (str): The sampling method to use. Currently only 'uniform' is supported.
        sampling_num (int): The number of samples to generate.

    Returns:
        np.ndarray: An array of shape (sampling_num, input_features.total_dim) containing the generated samples.
    """
    if method == 'uniform':
        sampler = UniformSampler(input_features)
    elif method == 'guided_uniform':
        sampler = GuidedUniformSampler(input_features)
    elif method == 'sweep':
        sampler = SweepSampler(input_features)
    elif method == 'ratio_sweep':
        sampler = RatioSweepSampler(input_features)
    elif method == 'test_sweep':
        sampler = TestSweepSampler(input_features, step_ratio)
    else:
        raise NotImplementedError
    
    return sampler

# def make_inference_datasets(
#     configs: Config,  
#     inference_input_features,
#     input_featuers,
#     output_col: str,
#     device: torch.device,
#     multiport: bool = False) -> BaseDataset:
    
#     # stackups must be provided when dataset_mode is final
#     if input_features.sampled_varaiables != inference_input_features.sampled_variables:
#         raise ValueError('inference parameters must be the same as input features')
#     for v_i, v in enumerate(input_features.sampled_variables):
        
#                 return dataset, None
#             else:
#                 # output cols are (1,1), (1,2), ...
#                 # df_list's shape is (col num, ds num)
#                 for col_idx, load_file_col in enumerate(output_col):
#                     df_list.append([])
#                     for ds in configs.datasets.datasets:
#                         df, stackups = _load_data(configs, ds, load_file_col)
#                         df_list[col_idx].append(df)
#                         if ds == configs.datasets.datasets[0]:
#                             stackups_list.append(stackups)

#         else:
#             # TODO: add multiport features
#             if multiport:
#                 raise NotImplementedError
                
#             # load train / test datasets
#             train_dfs = []
#             test_dfs = []
#             train_stackups = []
#             test_stackups = []
#             if 'SR' in output_col or 'SI' in output_col:
#                 load_file_col = output_col[2:]
#             else:
#                 load_file_col = output_col

#             # if masked column
#             cols = load_file_col[1:-1].split('_')
#             # count input_features.pattern has how many S
#             port_per_layer = input_features.pattern.count('S')
#             # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
#             unmasked_col = [int(cols[0]) % port_per_layer, int(cols[1]) % port_per_layer]
            
#             for i, ds in enumerate(configs.datasets.train_datasets): 
#                 train_df, train_stackup = _load_data(configs, ds, load_file_col)
#                 train_stackups.append(train_stackup)
#                 train_dfs.append(train_df)

#             for i, ds in enumerate(configs.datasets.test_datasets):
#                 test_df, test_stackup = _load_data(configs, ds, load_file_col)
#                 test_stackups.append(test_stackup)
#                 test_dfs.append(test_df)
            
#             train_dataset = MultiLayerDataset(
#                 configs=configs,
#                 df_list=train_dfs,
#                 input_features=input_features,
#                 output_col=output_col,
#                 stackups_list=train_stackups,
#                 unmasked_col=unmasked_col,
#                 device=device,
#             )

#             test_dataset = MultiLayerDataset(
#                 configs=configs,
#                 df_list=test_dfs,
#                 input_features=input_features,
#                 output_col=output_col,
#                 stackups_list=test_stackups,
#                 unmasked_col=unmasked_col,
#                 device=device,
#             )
#             return train_dataset, test_dataset
#     else:
#         raise NotImplementedError


def make_datasets(
    configs: Config,
    input_features,
    output_col: str,
    device: torch.device,
    multiport: bool = False,
    data_cols: List = None) -> BaseDataset:
    """
    This function creates the most basic datasets where input are (N, Cin) and output are (N, Cout = nF).
    Dataset mode contains:
        - normal: load in all data
        - freq: (not recommended) load frequency as an input feature
        - port: load in only one port

    Args:
        configs: Config
        output: str SR(1,1) | SI(1,1) | ...

    Returns:
        Dataset: torch.dataset
    """
    def _load_data(configs: Config, ds: str, output_col: str):
        stackup_writer = StackupWriter.load_pickle(configs, ds)
        print(f'Loading {ds}_{output_col}_concat.zip')
        df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}/{ds}_{output_col}_concat.zip'), compression='zip')
        df.index = df.index.astype(str, copy = False)
        return df, stackup_writer.stackups
    
    case = configs.case

    # dataset_mode: normal | freq | port
    dataset_mode = configs.datasets.mode
    
    # dataset mode is normal or freq
    
    if dataset_mode in ['normal', 'freq']:

        if multiport:
            raise ValueError('multiport is not supported in normal or freq mode')

        train_dfs = []
        test_dfs = []
        indices = {}
        indices['train_idx'] = []
        indices['test_idx'] = []

        for i, ds in enumerate(configs.datasets.train_datasets):
            train_df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}.zip'), compression='zip')
            train_df.index = train_df.index + ds
            train_dfs.append(train_df)

        for i, ds in enumerate(configs.datasets.test_datasets):
            test_df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}.zip'), compression='zip')
            test_df.index = test_df.index + ds
            test_dfs.append(test_df)

    elif dataset_mode == 'port':

        if multiport:
            raise ValueError('multiport is not supported in normal or freq mode')
        
        train_dfs = []
        test_dfs = []
        indices = {}
        indices['train_idx'] = []
        indices['test_idx'] = []

        for i, ds in enumerate(configs.datasets.train_datasets): 
            train_df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}/{ds}_{output_col[2:]}_concat.zip'), compression='zip')
            train_df.index = train_df.index + ds
            train_dfs.append(train_df)

        for i, ds in enumerate(configs.datasets.test_datasets):
            test_df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}/{ds}_{output_col[2:]}_concat.zip'), compression='zip') 
            test_df.index = test_df.index + ds
            test_dfs.append(test_df)

    elif dataset_mode == 'final':
        # stackups must be provided when dataset_mode is final
        if hasattr(configs.datasets, 'datasets'):
            df_list = []
            stackups_list = []
            if not multiport:
                if 'SR' in output_col or 'SI' in output_col:
                    load_file_col = output_col[2:]
                else:
                    load_file_col = output_col
                for ds in configs.datasets.datasets:
                    df, stackups = _load_data(configs, ds, load_file_col)
                    df_list.append(df)
                    stackups_list.append(stackups)
                # if masked column
                cols = load_file_col[1:-1].split(',')
                # count input_features.pattern has how many S
                port_per_layer = input_features.pattern.count('S')
                # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
                unmasked_col = [int(cols[0]) % port_per_layer, int(cols[1]) % port_per_layer]
                dataset = MultiLayerDataset(
                    configs=configs,
                    df_list=df_list,
                    input_features=input_features,
                    output_col=output_col,
                    stackups_list=stackups_list,
                    unmasked_col=unmasked_col,
                    device=device,
                )
                return dataset, None
            else:
                # output cols are (1,1), (1,2), ...
                # df_list's shape is (col num, ds num)
                raise NotImplementedError
                for col_idx, load_file_col in enumerate(output_col):
                    df_list.append([])
                    for ds in configs.datasets.datasets:
                        df, stackups = _load_data(configs, ds, load_file_col)
                        df_list[col_idx].append(df)
                        if ds == configs.datasets.datasets[0]:
                            stackups_list.append(stackups)            
        else:
            if multiport:
                raise NotImplementedError
            # load train / test datasets
            train_dfs = []
            test_dfs = []
            train_stackups = []
            test_stackups = []
            if 'SR' in output_col or 'SI' in output_col:
                load_file_col = output_col[2:]
            else:
                load_file_col = output_col

            # if masked column
            cols = load_file_col[1:-1].split('_')
            # count input_features.pattern has how many S
            port_per_layer = input_features.pattern.count('S')
            # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
            unmasked_col = [int(cols[0]) % port_per_layer, int(cols[1]) % port_per_layer]
            
            for i, ds in enumerate(configs.datasets.train_datasets): 
                train_df, train_stackup = _load_data(configs, ds, load_file_col)
                train_stackups.append(train_stackup)
                train_dfs.append(train_df)

            for i, ds in enumerate(configs.datasets.test_datasets):
                test_df, test_stackup = _load_data(configs, ds, load_file_col)
                test_stackups.append(test_stackup)
                test_dfs.append(test_df)
            
            train_dataset = MultiLayerDataset(
                configs=configs,
                df_list=train_dfs,
                input_features=input_features,
                output_col=output_col,
                stackups_list=train_stackups,
                unmasked_col=unmasked_col,
                device=device,
            )

            test_dataset = MultiLayerDataset(
                configs=configs,
                df_list=test_dfs,
                input_features=input_features,
                output_col=output_col,
                stackups_list=test_stackups,
                unmasked_col=unmasked_col,
                device=device,
            )
            return train_dataset, test_dataset
    
    elif dataset_mode == 'final_multi_column':
        if hasattr(configs.datasets, 'datasets'):
            if multiport:
                dfs = []
                stackups = []
                port_per_layer = input_features.pattern.count('S')
                unmasked_cols = []
                for i, ds in enumerate(configs.datasets.datasets):
                    for col_idx, col in enumerate(output_col):
                        for data_col in data_cols[col_idx]:
                            df, stackup = _load_data(configs, ds, data_col)
                            dfs.append(df.sort_index())
                            if col_idx == 0:
                                stackups.append(stackup)
                                col_indices = data_col[1:-1].split(',')
                                # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
                                unmasked_col = [int(col_indices[0]) % port_per_layer, int(col_indices[1]) % port_per_layer]
                                unmasked_cols.append(unmasked_col)
                dataset = MultiLayerCombinedColumnDataset(
                    configs=configs,
                    df_list=dfs,
                    input_features=input_features,
                    output_col=output_col,
                    stackups_list=stackups,
                    unmasked_cols=unmasked_cols,
                    port_per_layer=port_per_layer,
                    device=device,
                )
            else:
                dfs = []
                stackups = []
                # count input_features.pattern has how many S
                port_per_layer = input_features.pattern.count('S')
                unmasked_cols = []
                for i, ds in enumerate(configs.datasets.datasets): 
                    for data_col in data_cols:
                        df, stackup = _load_data(configs, ds, data_col)
                        stackups.append(stackup)
                        dfs.append(df)
                        col_indices = data_col[1:-1].split(',')
                        # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
                        unmasked_col = [int(col_indices[0]) % port_per_layer, int(col_indices[1]) % port_per_layer]
                        unmasked_cols.append(unmasked_col)

                dataset = MultiLayerCombinedColumnDataset(
                    configs=configs,
                    df_list=dfs,
                    input_features=input_features,
                    output_col=output_col,
                    stackups_list=stackups,
                    unmasked_cols=unmasked_cols,
                    port_per_layer=port_per_layer,
                    device=device,
                )

            return dataset, None
        else:
            if multiport:
                train_dfs = []
                train_stackups = []
                port_per_layer = input_features.pattern.count('S')
                train_unmasked_cols = []
                for i, ds in enumerate(configs.datasets.train_datasets):
                    for col_idx, col in enumerate(output_col):
                        for data_col in data_cols[col_idx]:
                            df, stackup = _load_data(configs, ds, data_col)
                            train_dfs.append(df.sort_index())
                            if col_idx == 0:
                                train_stackups.append(stackup)
                                col_indices = data_col[1:-1].split(',')
                                # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
                                unmasked_col = [int(col_indices[0]) % port_per_layer, int(col_indices[1]) % port_per_layer]
                                train_unmasked_cols.append(unmasked_col)
                train_dataset = MultiLayerCombinedColumnDataset(
                    configs=configs,
                    df_list=train_dfs,
                    input_features=input_features,
                    output_col=output_col,
                    stackups_list=train_stackups,
                    unmasked_cols=train_unmasked_cols,
                    port_per_layer=port_per_layer,
                    device=device,
                )
                test_dfs = []
                test_stackups = []
                port_per_layer = input_features.pattern.count('S')
                test_unmasked_cols = []
                for i, ds in enumerate(configs.datasets.test_datasets):
                    for col_idx, col in enumerate(output_col):
                        for data_col in data_cols[col_idx]:
                            df, stackup = _load_data(configs, ds, data_col)
                            test_dfs.append(df.sort_index())
                            if col_idx == 0:
                                test_stackups.append(stackup)
                                col_indices = data_col[1:-1].split(',')
                                # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
                                unmasked_col = [int(col_indices[0]) % port_per_layer, int(col_indices[1]) % port_per_layer]
                                test_unmasked_cols.append(unmasked_col)
                test_dataset = MultiLayerCombinedColumnDataset(
                    configs=configs,
                    df_list=test_dfs,
                    input_features=input_features,
                    output_col=output_col,
                    stackups_list=test_stackups,
                    unmasked_cols=test_unmasked_cols,
                    port_per_layer=port_per_layer,
                    device=device,
                )
            else:
                # load train / test datasets
                train_dfs = []
                test_dfs = []
                train_stackups = []
                test_stackups = []
                # count input_features.pattern has how many S
                port_per_layer = input_features.pattern.count('S')
                train_unmasked_cols = []
                test_unmasked_cols = []
                for i, ds in enumerate(configs.datasets.train_datasets): 
                    for data_col in data_cols:
                        train_df, train_stackup = _load_data(configs, ds, data_col)
                        train_stackups.append(train_stackup)
                        train_dfs.append(train_df)
                        col_indices = data_col[1:-1].split(',')
                        # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
                        unmasked_col = [int(col_indices[0]) % port_per_layer, int(col_indices[1]) % port_per_layer]
                        train_unmasked_cols.append(unmasked_col)
                for i, ds in enumerate(configs.datasets.test_datasets):
                    for data_col in data_cols:
                        test_df, test_stackup = _load_data(configs, ds, data_col)
                        test_stackups.append(test_stackup)
                        test_dfs.append(test_df)
                        col_indices = data_col[1:-1].split(',')
                        # for example, S(1,1) has col_i = 1, col_j = 1, the unmasked_col should be [1, 1]
                        unmasked_col = [int(col_indices[0]) % port_per_layer, int(col_indices[1]) % port_per_layer]
                        test_unmasked_cols.append(unmasked_col)
                
                train_dataset = MultiLayerCombinedColumnDataset(
                    configs=configs,
                    df_list=train_dfs,
                    input_features=input_features,
                    output_col=output_col,
                    stackups_list=train_stackups,
                    unmasked_cols=train_unmasked_cols,
                    port_per_layer=port_per_layer,
                    device=device,
                )

                test_dataset = MultiLayerCombinedColumnDataset(
                    configs=configs,
                    df_list=test_dfs,
                    input_features=input_features,
                    output_col=output_col,
                    stackups_list=test_stackups,
                    unmasked_cols=test_unmasked_cols,
                    port_per_layer=port_per_layer,
                    device=device,
                )
            return train_dataset, test_dataset

    else:
        raise NotImplementedError
    
    if len(train_dfs) != 0: 
        train_df = pd.concat(train_dfs)
        if dataset_mode == 'normal' or dataset_mode == 'port':
            train_dataset = BaseDataset(
                configs=configs,
                df=train_df,
                input_features=input_features,
                output_col=output_col,
                device=device,
            )
        elif dataset_mode == 'freq':
            train_dataset = FDataset(
                configs=configs,
                df=train_df,
                input_features=input_features,
                output_col=output_col,
                device=device,
            )
        elif dataset_mode == 'final':
            train_dataset = MultiLayerDataset(
                configs=configs,
                df=train_df,
                input_features=input_features,
                output_col=output_col,
                stackups=train_stackups,
                device=device,
            )
        else:
            raise NotImplementedError
    else:
        train_dataset = None
    
    if len(test_dfs) != 0:
        test_df = pd.concat(test_dfs)
        if dataset_mode == 'normal' or dataset_mode == 'port':
            test_dataset = BaseDataset(
                configs=configs,
                df=test_df,
                input_features=input_features,
                output_col=output_col,
                device=device,
            )
        elif dataset_mode == 'freq':
            test_dataset = FDataset(
                configs=configs,
                df=test_df,
                input_features=input_features,
                output_col=output_col,
                device=device,
            )
        elif dataset_mode == 'final':
            test_dataset = MultiLayerDataset(
                configs=configs,
                df=test_df,
                input_features=input_features,
                output_col=output_col,
                stackups=test_stackups,
                device=device,
            )
        else:
            raise NotImplementedError
    else:
        test_dataset = None
        
    if configs.verbose:
        print('Loading data finished.')

    return train_dataset, test_dataset

def make_multi_ports_datasets(
    configs: Config,
    input_features: InputFeatures,
    model_name: str,
    output_cols: List[str],
    device: torch.device) -> BaseDataset:
    """
    This function creates the most basic datasets where input are (N, Cin) and output are (N, Cout = nF).
    Dataset mode contains:
        - normal: load in all data
        - freq: (not recommended) load frequency as an input feature
        - port: load in only one port

    Args:
        configs: Config
        model_name: SR(1,1) | SR(1,2)
        output_cols: SR(1,1), SR(2,2), SR(3,3), ... | SR(1,2), SR(2,3), SR(3,4), ...

    Returns:
        Dataset: torch.dataset
    """
    case = configs.case

    # dataset_mode: normal | freq | port
    dataset_mode = configs.datasets.mode
    
    # dataset mode is normal or freq
    train_dfs = []
    test_dfs = []
    indices = {}
    indices['train_idx'] = []
    indices['test_idx'] = []
    
    if dataset_mode in ['normal', 'freq']:
    
        for i, ds in enumerate(configs.datasets.train_datasets):
            train_df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}.zip'), compression='zip')
            train_df.index = train_df.index + ds
            train_dfs.append(train_df)

        for i, ds in enumerate(configs.datasets.test_datasets):
            test_df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}.zip'), compression='zip')
            test_df.index = test_df.index + ds
            test_dfs.append(test_df)

    elif dataset_mode == 'port':

        for output_col in output_cols:
            for i, ds in enumerate(configs.datasets.train_datasets):
                
                train_df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}/{ds}_{output_col[2:]}_concat.zip'), compression='zip')
                
                if output_col == model_name:
                    train_columns_names = train_df.columns

                train_df.index = train_df.index + ds + output_col[2:]
                train_dfs.append(train_df)

            for i, ds in enumerate(configs.datasets.test_datasets):

                test_df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}/{ds}_{output_col[2:]}_concat.zip'), compression='zip') 
                
                if output_col == model_name:
                    test_columns_names = test_df.columns

                test_df.index = test_df.index + ds + output_col[2:]
                test_dfs.append(test_df)

    if len(train_dfs) != 0: 

        for train_df in train_dfs:
            train_df.columns = train_columns_names

        train_df = pd.concat(train_dfs)
        if dataset_mode == 'normal' or dataset_mode == 'port':
            train_dataset = BaseDataset(
                configs=configs,
                df=train_df,
                input_features=input_features,
                output_col=model_name,
                device=device,
            )
        elif dataset_mode == 'freq':
            train_dataset = FDataset(
                configs=configs,
                df=train_df,
                input_features=input_features,
                output_col=model_name,
                device=device,
            )
        else:
            raise NotImplementedError
    else:
        train_dataset = None
    
    if len(test_dfs) != 0:

        for test_df in test_dfs:
            test_df.columns = test_columns_names

        test_df = pd.concat(test_dfs)
        if dataset_mode == 'normal' or dataset_mode == 'port':
            test_dataset = BaseDataset(
                configs=configs,
                df=test_df,
                input_features=input_features,
                output_col=model_name,
                device=device,
            )
        elif dataset_mode == 'freq':
            test_dataset = FDataset(
                configs=configs,
                df=test_df,
                input_features=input_features,
                output_col=model_name,
                device=device,
            )
        else:
            raise NotImplementedError
    else:
        test_dataset = None
        

    if configs.verbose:
        print('Loading data finished.')
    
    return train_dataset, test_dataset

def make_datasets_v1(
    configs: Config,
    input_cols,
    output_cols,
    nF,
    device,
    move_nonlinear_first: bool = False,
    nonlinear_params: List[str] = None,
) -> Tuple[CustomDataset, CustomDataset]:
    return _make_datasets_v1(
        working_dir=os.getcwd(),
        datasets=configs.datasets.name,
        input_cols=input_cols,
        output_cols=output_cols,
        nF=nF,
        config_file=os.path.join(os.getcwd(), configs.config.dir),
        verbose=configs.verbose,
        device=device,
        upsample=configs.datasets.upsample,
        upsample_ratio=configs.datasets.upsample_ratio,
        expand_dim=configs.datasets.expand_dim,
        move_nonlinear_first=move_nonlinear_first,
        nonlinear_params=nonlinear_params,
    )

def _make_datasets_v1(
    working_dir,
    datasets,
    input_cols,
    output_cols,
    nF: int,
    config_file: str,
    verbose: bool,
    device: torch.device,
    upsample: bool = False,
    upsample_ratio: int = 9,
    expand_dim: bool = False,
    move_nonlinear_first: bool = False,
    nonlinear_params: List[str] = None,
) -> Tuple[CustomDataset, CustomDataset]:
    # read data
    train_dfs = []
    test_dfs = []
    indices = {}
    indices['train_idx'] = []
    indices['test_idx'] = []

    for i, ds in enumerate(datasets):
        train_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_train_concat.zip' %(ds, ds)), compression='zip')
        train_df.index = train_df.index + '_%s' % ds
        train_dfs.append(train_df)

        test_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_test_concat.zip' %(ds, ds)), compression='zip')
        test_df.index = test_df.index + '_%s' % ds
        test_dfs.append(test_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    if verbose:
        print('Loading data finished.')
    
    train_dataset = CustomDataset(
        train_df, input_cols, output_cols, nF, None, config_file, device)
    test_dataset = CustomDataset(
        test_df, input_cols, output_cols, nF, None, config_file, device)
    
    # Move nonlinear parameters first for subnet
    if move_nonlinear_first:
        train_dataset.move_nonlinear_features_first(
            nonlinear_params=nonlinear_params,
            indices=None,
            df=train_df
        )
        test_dataset.move_nonlinear_features_first(
            nonlinear_params=nonlinear_params,
            indices=None,
            df=test_df
        )

    if upsample:
        train_dataset.upsample_input(upsample_ratio=upsample_ratio, expand_dim=expand_dim)
        test_dataset.upsample_input(upsample_ratio=upsample_ratio, expand_dim=expand_dim)
    
    return train_dataset, test_dataset

def make_physics_informed_datasets(
    configs: Config,
    input_cols,
    output_cols,
    nF,
    device,
):
    
    # read data
    train_dfs = []
    test_dfs = []
    indices = {}
    indices['train_idx'] = []
    indices['test_idx'] = []

    working_dir = os.getcwd()
    config_file = os.path.join(os.getcwd(), configs.config.dir)
    for i, ds in enumerate(configs.datasets.name):
        train_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_train_concat.zip' %(ds, ds)), compression='zip')
        train_df.index = train_df.index + '_%s' % ds
        train_dfs.append(train_df)

        test_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_test_concat.zip' %(ds, ds)), compression='zip')
        test_df.index = test_df.index + '_%s' % ds
        test_dfs.append(test_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    if configs.verbose:
        print('Loading data finished.')
    
    train_dataset = PhysicsInformedDataset(
        train_df, input_cols, output_cols, nF, None, config_file, device)
    test_dataset = PhysicsInformedDataset(
        test_df, input_cols, output_cols, nF, None, config_file, device)

    return train_dataset, test_dataset

def make_rlgc_datasets(
    configs: Config,
    input_cols,
    output_cols,
    nF,
    mode,
    device,
):
    # read data
    train_dfs = []
    test_dfs = []
    indices = {}
    indices['train_idx'] = []
    indices['test_idx'] = []

    working_dir = os.getcwd()
    config_file = os.path.join(os.getcwd(), configs.config.dir)
    for i, ds in enumerate(configs.datasets.name):
        train_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_train_rlgc_concat.zip' %(ds, ds)), compression='zip')
        train_df.index = train_df.index + '_%s' % ds
        train_dfs.append(train_df)

        test_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_test_rlgc_concat.zip' %(ds, ds)), compression='zip')
        test_df.index = test_df.index + '_%s' % ds
        test_dfs.append(test_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    if configs.verbose:
        print('Loading data finished.')
    
    train_dataset = RLGCDataset(
        train_df, input_cols, output_cols, nF, None, config_file, mode, device)
    test_dataset = RLGCDataset(
        test_df, input_cols, output_cols, nF, None, config_file, mode, device)

    return train_dataset, test_dataset
 
    
def make_model(
    configs: Config,
    in_channels: int,
    out_channels: int,
    multiport: int,
    device: torch.device,
) -> torch.nn.Module:
    if configs.model.name == "mlp":
        model = MLP(
            in_channels,
            out_channels,
            configs.model.layers
        )
    elif configs.model.name == "mlp_srsi":
        model = MLP_SRSI(
            in_channels,
            out_channels,
            configs.model.trunk_layers,
            configs.model.head_layers,
            multiport,
        )
    elif configs.model.name == "resnet":
        model = ResNet(
            in_channels,
            out_channels,
            configs.model.layers
        )
    elif configs.model.name == "resnet_srsi":
        model = ResNet_SRSI(
            in_channels,
            out_channels,
            configs.model.trunk_layers,
            configs.model.head_layers,
            multiport
        )
    elif configs.model.name == "transposed_conv_net":
        model = TransposedConvNet(
            in_features=in_channels,
            layers=configs.model.layers,
            out_features=out_channels,
            out_channels=1,
        )
    elif configs.model.name == "transposed_conv_net_srsi":
        model = TransposedConvNet(
            in_features=in_channels,
            layers=configs.model.layers,
            out_features=out_channels,
            out_channels=multiport if multiport != 0 else 2,
            multiport=(multiport != 0)
        )
    elif configs.model.name == "unet":
        model = unet(
            in_channels=in_channels,
            mid_channels=configs.model.layers,
            out_channels=out_channels
        )
    
    elif configs.model.name == "base_LSTM":
        model = BaseLSTM(
            in_channels=in_channels,
            mid_channels=configs.model.layers,
            out_channels=out_channels
        )
    else:
        raise NotImplementedError
    return model.to(device)

def make_optimizer(
    params,
    configs: Config
) -> Optimizer:
    return _make_optimizer(
        name=configs.optimizer.name,
        params=params,
        lr=configs.optimizer.lr,
        weight_decay=configs.optimizer.weight_decay,
    )

def _make_optimizer(
    name: str,
    params,
    lr,
    weight_decay,
) -> Optimizer:
    
    if name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr,
            weight_decay=weight_decay,
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr,
            weight_decay=weight_decay,
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer
        
def make_scheduler(
    optimizer: Optimizer, 
    configs: Config
) -> torch.optim.lr_scheduler:
    return _make_scheduler(
        optimizer=optimizer,
        name=configs.scheduler.name,
    )

def _make_scheduler(
    optimizer: Optimizer, 
    name: str = None,
) -> torch.optim.lr_scheduler:
    
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    else:
        raise NotImplementedError(name)

    return scheduler

def make_criterion(
    configs: Config
):
    if configs.loss.name == 'custom_mse':
        criterion = CustomMSELoss()
    elif configs.loss.name == 'mse':
        criterion = nn.MSELoss()
    elif configs.loss.name == 'L1':
        criterion = nn.L1Loss()
    elif configs.loss.name == 'Huber':
        criterion = nn.HuberLoss(delta=configs.loss.delta)
    else:
        raise NotImplementedError(configs.loss.name)
    return criterion
