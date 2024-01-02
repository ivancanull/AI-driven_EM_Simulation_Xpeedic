#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import re
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import time

from cores import *
from utils import *
from typing import List

from joblib import Parallel, delayed

__all__ = [
    "read_input_feature_xlsx",
]
# from tqdm import tqdm

def generate_port_list_multilayer(port):
    port_list = []
    for i in range(port):
        port_list.append(f'(1,{i+1})')
    return port_list

def parse_data(
        dataset_dir, 
        port,   
        nF, 
        dataset_name,
        diff):
    """
    This function is runned local where data is kept.

    Consider a dataset of the following structures:

    -- dataset
        -- subset1
            -- line1
                --
            -- line2
                --
            ...
            script.xlsx
        -- subset2
            ...

    -> save to an entire csv

    :param dataset_dir: directory where the dataset is stored
    :param port: port number of the simulated model
    :param nF: number of frequency points in each simulation
    :param dataset_name: the name of dataset 
    """
    
    snp_headers = []

    # create the df headers
    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                for n in range(nF):
                    snp_headers.append('%s(%d,%d)_%d' % (s, i+1, j+1, n))
    
    # recursively read each subset
    for subset in os.listdir(dataset_dir):
        
        if os.path.isdir(os.path.join(dataset_dir, subset)):

            keys = []
            para_df_list = []
            snp_df_list = []
            
            # define the saved data file path
            filepath = os.path.join(dataset_dir, '%s_%s.zip' % (dataset_name, subset))
            if os.path.exists(filepath):
                continue
            print('Creating %s_%s.zip' % (dataset_name, subset))
            
            # read parameters
            subset_dir = os.path.join(dataset_dir, subset)
            excel_files = [f for f in os.listdir(subset_dir) if f.endswith('.xlsx')]
            for excel_file in excel_files:
                df = pd.read_excel(os.path.join(dataset_dir, subset, excel_file), sheet_name='Mixed_N_Line_Stripline', skiprows=23)
                new_df = df.T.loc[df.T.index[1:]].copy()
                new_df.columns = df.T.loc['batch list'].to_numpy()
                new_df = new_df.set_index('save dir file name')
                new_df.index.name = None
                para_df_list.append(new_df)

            # create a parameter dataframe
            para_df = pd.concat(para_df_list)

            # read lines in the snp files
            for dir in os.listdir(subset_dir):
                if os.path.isdir(os.path.join(subset_dir, dir, 'RLGC')):
                    snp = 'TransmissionLine.s%dp' % port
                    if os.path.exists(os.path.join(subset_dir, dir, 'RLGC', snp)):
                        keys.append(dir)
                        snp_df = parse_generated_s_parameters(os.path.join(subset_dir, dir, 'RLGC', snp), port).loc[:, 1:]
                        snp_df = pd.DataFrame(np.array(snp_df).reshape((1,-1), order='F'))
                        snp_df.columns = snp_headers  
                        for p in para_df.columns:
                            if (p == 'W' or p == 'S') and isinstance(para_df.loc[dir, p], str):
                                if diff:
                                    p_list = para_df.loc[dir, p].split(',')
                                    for i, v in enumerate(p_list):
                                        snp_df.loc[:, f'{p}_{i}'] = float(v)
                                else:
                                    snp_df.loc[:, p] = float(para_df.loc[dir, p].split(',')[0])
                            else:
                                snp_df.loc[:, p] = float(para_df.loc[dir, p])

                        snp_df_list.append(snp_df)

            # concatenate the dataframe
            df = pd.concat(snp_df_list)
            df = df.set_index(keys=pd.Index(keys))

            df.to_pickle(filepath, compression='zip')

    return

def parse_data_metis(
        dataset_dir, 
        port,   
        nF, 
        dataset_name,
        diff):
    """
    This function is runned local where data is kept.

    Consider a dataset of the following structures:

    -- dataset
        -- subset1
            -- Parametrics
                -- line1
                    --
                -- line2
                    --
            ...
            script.xlsx
        -- subset2
            ...

    -> save to an entire csv

    :param dataset_dir: directory where the dataset is stored
    :param port: port number of the simulated model
    :param nF: number of frequency points in each simulation
    :param dataset_name: the name of dataset 
    """
    
    snp_headers = []

    # create the df headers
    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                for n in range(nF):
                    snp_headers.append('%s(%d,%d)_%d' % (s, i+1, j+1, n))
    
    # recursively read each subset
    for subset in os.listdir(dataset_dir):
        
        if os.path.isdir(os.path.join(dataset_dir, subset)):
            excel_file = [f for f in os.listdir(os.path.join(dataset_dir, subset)) if f.endswith('.xlsx')]
            if len(excel_file) == 0:
                raise ValueError('No excel file found in %s' % os.path.join(dataset_dir, subset))
            else:
                excel_file = excel_file[0]
            
            # read stackup xlsx file
            stackup_df = pd.read_excel(os.path.join(dataset_dir, subset, excel_file), sheet_name='Stackup')
            # select column 'Layer Name', 'Parameter Index' and discard all na value
            stackup_df = stackup_df.loc[:, ['Layer Name', 'Parameter Index']].dropna()

            # read parameter xlsx file
            para_df = read_multilayer_input_features_xlsx(os.path.join(dataset_dir, subset, excel_file))
            # process the parameter df
            # drop all columns contain na
            para_df = para_df.dropna(axis=1)
            columns_num = para_df.shape[1]
            new_columns = []
            # add index_0 to the first four columns' names
            new_columns.extend(['index_0_' + para_df.columns[i] for i in range(4)])
            # add index_1 to the next four columns' names
            new_columns.extend(['index_1_' + para_df.columns[i] for i in range(4, 8)])
            # recursively add index_i to the next seven columns' names
            for i in range((columns_num - 8) // 7):
                new_columns.extend([f'index_{i+2}_' + para_df.columns[j] for j in range(i * 7 + 8, (i + 1) * 7 + 8)])
            # rename the columns
            para_df.columns = new_columns
            # create a index of 1 to n for the parameter df
            para_df.index = np.arange(1, para_df.shape[0] + 1)

            # read lines in the snp files
            parametric_dir = os.path.join(dataset_dir, subset, 'Optimetrics', 'ParametricSetup1')
            
            keys = []
            para_df_list = []
            snp_df_list = []
        
            # define the saved data file path
            filepath = os.path.join(dataset_dir, '%s_%s.zip' % (dataset_name, subset))
            if os.path.exists(filepath):
                continue
            print('Creating %s_%s.zip' % (dataset_name, subset))
                            
            if os.path.isdir(parametric_dir):

                for dir in os.listdir(parametric_dir):
                    snp = 'TransmissionLine.s%dp' % port
                    if os.path.exists(os.path.join(parametric_dir, dir, snp)):
                        keys.append(int(dir))
                        time_start = time.time()
                        snp_df = parse_generated_s_parameters(os.path.join(parametric_dir, dir, snp), port).loc[:, 1:]
                        time_end = time.time()
                        print('Finished parsing %s in %f seconds' % (os.path.join(parametric_dir, dir, snp), time_end - time_start))
                        snp_df = pd.DataFrame(np.array(snp_df).reshape((1,-1), order='F'))
                        snp_df.columns = snp_headers  
                        snp_df_list.append(snp_df)

                        for p in para_df.columns:
                            # if p ends with '_W' or '_S':
                            if p.endswith('_W') or p.endswith('_S'):
                                if diff:
                                    p_list = para_df.loc[int(dir), p].split(',')
                                    for i, v in enumerate(p_list):
                                        snp_df.loc[:, f'{p}_{i}'] = float(v)
                                else:
                                    snp_df.loc[:, p] = float(para_df.loc[int(dir), p].split(',')[0])
                            elif p.endswith('Pattern'):
                                snp_df.loc[:, p] = para_df.loc[int(dir), p]
                            else:
                                snp_df.loc[:, p] = float(para_df.loc[int(dir), p])
                
                df = pd.concat(snp_df_list)
                df.index = np.arange(1, df.shape[0] + 1)
                df.to_pickle(filepath, compression='zip')
            else:
                raise ValueError('No parametric directory found in %s' % parametric_dir)
            
    return

def parse_data_metis_parallel(
        dataset_dir, 
        port,   
        nF, 
        dataset_name,
        diff,
        regen=False):
    """
    This function is runned local where data is kept.

    Consider a dataset of the following structures:

    -- dataset
        -- subset1
            -- Parametrics
                -- line1
                    --
                -- line2
                    --
            ...
            script.xlsx
        -- subset2
            ...

    -> save to an entire csv

    :param dataset_dir: directory where the dataset is stored
    :param port: port number of the simulated model
    :param nF: number of frequency points in each simulation
    :param dataset_name: the name of dataset 
    """
    
    num_cores = 28
    snp_headers = []

    # create the df headers
    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                for n in range(nF):
                    snp_headers.append('%s(%d,%d)_%d' % (s, i+1, j+1, n))
    
    # recursively read each subset
    for subset in os.listdir(dataset_dir):
        
        if os.path.isdir(os.path.join(dataset_dir, subset)):
            excel_file = [f for f in os.listdir(os.path.join(dataset_dir, subset)) if f.endswith('.xlsx')]
            if len(excel_file) == 0:
                raise ValueError('No excel file found in %s' % os.path.join(dataset_dir, subset))
            else:
                excel_file = excel_file[0]
            
            # read stackup xlsx file
            stackup_df = pd.read_excel(os.path.join(dataset_dir, subset, excel_file), sheet_name='Stackup')
            # select column 'Layer Name', 'Parameter Index' and discard all na value
            stackup_df = stackup_df.loc[:, ['Layer Name', 'Parameter Index']].dropna()

            # read parameter xlsx file
            para_df = read_multilayer_input_features_xlsx(os.path.join(dataset_dir, subset, excel_file))
            # process the parameter df
            # drop all columns contain na
            para_df = para_df.dropna(axis=1)
            columns_num = para_df.shape[1]
            new_columns = []
            # add index_0 to the first four columns' names
            new_columns.extend(['index_0_' + para_df.columns[i] for i in range(4)])
            # add index_1 to the next four columns' names
            new_columns.extend(['index_1_' + para_df.columns[i] for i in range(4, 8)])
            # recursively add index_i to the next seven columns' names
            for i in range((columns_num - 8) // 7):
                new_columns.extend([f'index_{i+2}_' + para_df.columns[j] for j in range(i * 7 + 8, (i + 1) * 7 + 8)])
            # rename the columns
            para_df.columns = new_columns
            # create a index of 1 to n for the parameter df
            para_df.index = np.arange(1, para_df.shape[0] + 1)

            # read lines in the snp files
            if os.path.exists(os.path.join(dataset_dir, subset, 'Optimetrics')):
                parametric_dir = os.path.join(dataset_dir, subset, 'Optimetrics', 'ParametricSetup1')
            elif os.path.exists(os.path.join(dataset_dir, subset, 'Parametrics')):
                parametric_dir = os.path.join(dataset_dir, subset, 'Parametrics', 'ParametricSetup1')
            else:
                raise ValueError('No parametric directory found in %s' % os.path.join(dataset_dir, subset))
            snp_df_list = []
        
            # define the saved data file path
            filepath = os.path.join(dataset_dir, '%s_%s.zip' % (dataset_name, subset))
            if os.path.exists(filepath):
                continue
            print('Creating %s_%s.zip' % (dataset_name, subset))

            def read_s_parameters_multilayer_parallel(
                filepath,
                dir,
                port,
            ):
                if dir.endswith('.cfg'):
                    print('This directory is configuration file')
                    return
                if os.path.exists(filepath):
                    # if all sub files are generated
                    if os.path.exists(os.path.join(dataset_dir, subset, 'Optimetrics')):
                        last_file_path = os.path.join(dataset_dir, subset, 'Optimetrics', 'ParametricSetup1', dir, f'{port}_{port}.zip')
                    elif os.path.exists(os.path.join(dataset_dir, subset, 'Parametrics')):
                        last_file_path = os.path.join(dataset_dir, subset, 'Parametrics', 'ParametricSetup1', dir, f'{port}_{port}.zip')
                    else:
                        raise ValueError('No parametric directory found in %s' % os.path.join(dataset_dir, subset))
                    
                    if os.path.exists(last_file_path) and not regen:
                        # return
                        pass
                    para_cols = []
                    time_start = time.time()
                    # snp_df = parse_generated_s_parameters(filepath, port).loc[:, 1:]
                    snp_np = parse_generated_s_parameters(filepath, port).loc[:, 1:]

                    time_end = time.time()
                
                    snp_np = np.array(snp_np).reshape((1,-1), order='F')

                    # snp_df = pd.DataFrame(np.array(snp_df).reshape((1,-1), order='F'))
                    # snp_df.columns = snp_headers
                    
                    print('Finished parsing %s in %f seconds' % (filepath, time_end - time_start))
                    paras_np = np.zeros((1, 0))
                    WS_dict = {}
                    for p in para_df.columns:
                        # if p ends with '_W' or '_S':
                        if p.endswith('_W') or p.endswith('_S'):
                            if diff:
                                raise NotImplementedError
                                p_list = para_df.loc[int(dir), p].split(',')
                                for i, v in enumerate(p_list):
                                    snp_df.loc[:, f'{p}_{i}'] = float(v)
                                    para_cols.append(f'{p}_{i}')
                            else:
                                # add a column to snp_np
                                added = np.array(float(para_df.loc[int(dir), p].split(',')[0])).reshape(1,1)
                                WS_dict[p] = added
                                paras_np = np.append(paras_np, added, axis=1)

                                # snp_df.loc[:, p] = float(para_df.loc[int(dir), p].split(',')[0])
                                para_cols.append(p)
                        elif p.endswith('Pattern'):
                            added = [[0]]
                            paras_np = np.append(paras_np, added, axis=1)
                            
                            # snp_df.loc[:, p] = para_df.loc[int(dir), p]
                            para_cols.append(p)
                        else:
                            added = np.array(float(para_df.loc[int(dir), p])).reshape(1,1)
                            paras_np = np.append(paras_np, added, axis=1)
                            # snp_df[p] = float(para_df.loc[int(dir), p])
                            para_cols.append(p)
                    # print('Finished reading parameters dataframe')
                    # store snp_dfs seperately                    

                    for i in range(port):
                        for j in range(port):
                            
                            if os.path.exists(os.path.join(dataset_dir, subset, 'Optimetrics')):
                                saved_file_path = os.path.join(dataset_dir, subset, 'Optimetrics', 'ParametricSetup1', dir, f'{i+1}_{j+1}.zip')
                            elif os.path.exists(os.path.join(dataset_dir, subset, 'Parametrics')):
                                saved_file_path = os.path.join(dataset_dir, subset, 'Parametrics', 'ParametricSetup1', dir, f'{i+1}_{j+1}.zip')
                            else:
                                raise ValueError('No parametric directory found in %s' % os.path.join(dataset_dir, subset))
                            if not os.path.exists(saved_file_path) or regen:
                                
                                # para_cols.extend(snp_cols)
                                # selected_snp_df = snp_df.loc[:, para_cols]

                                # parse parallel cols coords distances
                                # get layer and column index
                                layer_i = get_layer(i, port_each_layer=3, layer_n=3)
                                layer_j = get_layer(j, port_each_layer=3, layer_n=3)
                                n_i = get_column_index(i % (port // 2), port_each_layer=3, layer_n=3)
                                n_j = get_column_index(j % (port // 2), port_each_layer=3, layer_n=3)

                                
                                coord_delta = generate_location(n1=n_i, 
                                                                n2=n_j, 
                                                                total_n=9,
                                                                W1=WS_dict[f'index_{layer_i+2}_W'],
                                                                W2=WS_dict[f'index_{layer_j+2}_W'],
                                                                S1=WS_dict[f'index_{layer_i+2}_S'],
                                                                S2=WS_dict[f'index_{layer_j+2}_S'])
                                
                                # too far maybe
                                # if abs(coord_delta) > 100:
                                #     continue

                                coord_delta_y_np = np.array(layer_j - layer_i).reshape(1,1)
                                coord_delta_x_np = coord_delta.reshape(1,1)

                                output_np = np.append(paras_np, coord_delta_x_np, axis=1)
                                output_np = np.append(output_np, coord_delta_y_np, axis=1)

                                snp_cols = [f'SR({i+1},{j+1})_{k}' for k in range(nF)]
                                snp_cols.extend([f'SI({i+1},{j+1})_{k}' for k in range(nF)])
                                start = i * port * nF * 2 + j * nF * 2
                                end = i * port * nF * 2 + j * nF * 2 + nF * 2
                                output_np = np.append(output_np, snp_np[:, start:end], axis=1)

                                written_cols = para_cols + ['coord_delta_x', 'coord_delta_y'] + snp_cols 
                                pd.DataFrame(output_np, columns=written_cols).to_pickle(saved_file_path, compression='zip')
                                
                    return
                else:
                    raise ValueError('No parametric directory %s found in %s' % (filepath, parametric_dir))

            if os.path.isdir(parametric_dir):

                with Parallel(n_jobs=num_cores) as parallel:
                    parallel(delayed(read_s_parameters_multilayer_parallel)(
                        os.path.join(parametric_dir, dir, 'TransmissionLine.s%dp' % port),
                        dir,
                        port,
                    ) for dir in os.listdir(parametric_dir))
                
                # for dir in os.listdir(parametric_dir):
                #     read_s_parameters_multilayer_parallel(
                #         os.path.join(parametric_dir, dir, 'TransmissionLine.s%dp' % port),
                #         dir,
                #         port,
                #     )

                
            else:
                raise ValueError('No parametric directory found in %s' % parametric_dir)
            
    return

def parse_data_metis_parallel_2(
        dataset_dir, 
        port,   
        nF, 
        dataset_name,
        diff,
        regen=False):
    """
    This function is runned local where data is kept.

    Consider a dataset of the following structures:

    -- dataset
        -- subset1
            -- Parametrics
                -- line1
                    --
                -- line2
                    --
            ...
            script.xlsx
        -- subset2
            ...

    -> save to an entire csv

    :param dataset_dir: directory where the dataset is stored
    :param port: port number of the simulated model
    :param nF: number of frequency points in each simulation
    :param dataset_name: the name of dataset 
    """
    
    num_cores = 28
    snp_headers = []

    # create the df headers
    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                for n in range(nF):
                    snp_headers.append('%s(%d,%d)_%d' % (s, i+1, j+1, n))
    
    # recursively read each subset
    for subset in os.listdir(dataset_dir):
        
        if os.path.isdir(os.path.join(dataset_dir, subset)):
            
           
            # read lines in the snp files
            if os.path.exists(os.path.join(dataset_dir, subset, 'Optimetrics')):
                parametric_dir = os.path.join(dataset_dir, subset, 'Optimetrics', 'ParametricSetup1')
            elif os.path.exists(os.path.join(dataset_dir, subset, 'Parametrics')):
                parametric_dir = os.path.join(dataset_dir, subset, 'Parametrics', 'ParametricSetup1')
            else:
                raise ValueError('No parametric directory found in %s' % os.path.join(dataset_dir, subset))
        
            # define the saved data file path
            filepath = os.path.join(dataset_dir, '%s_%s.zip' % (dataset_name, subset))
            if os.path.exists(filepath):
                pass
            print('Creating %s_%s.zip' % (dataset_name, subset))

            def read_s_parameters_multilayer_parallel(
                filepath,
                dir,
                port,
            ):
                if dir.endswith('.cfg'):
                    print('This directory is configuration file')
                    return
                if os.path.exists(filepath):
                    # if all sub files are generated
                    if os.path.exists(os.path.join(dataset_dir, subset, 'Optimetrics')):
                        op = 'Optimetrics'
                    elif os.path.exists(os.path.join(dataset_dir, subset, 'Parametrics')):
                        op = 'Parametrics'
                    else:
                        raise ValueError('No parametric directory found in %s' % os.path.join(dataset_dir, subset))

                    exists = True
                    for i in range(port):
                        for j in range(i, port):
                            file_path = os.path.join(dataset_dir, subset, op, 'ParametricSetup1', dir, f'{i+1}_{j+1}.zip')
                            exists = exists & os.path.exists(file_path)
                    
                    if exists and not regen:
                        return

                    time_start = time.time()
                    snp_df = parse_generated_s_parameters(filepath, port)
                    snp_df = snp_df.iloc[:, 1:]
                    time_end = time.time()
                    print(f'Finished parsing {filepath} in {time_end - time_start} seconds')
                    time_start = time.time()

                    for i in range(port):
                        for j in range(i, port):
                            if os.path.exists(os.path.join(dataset_dir, subset, 'Optimetrics')):
                                saved_file_path = os.path.join(dataset_dir, subset, 'Optimetrics', 'ParametricSetup1', dir, f'{i+1}_{j+1}.zip')
                            elif os.path.exists(os.path.join(dataset_dir, subset, 'Parametrics')):
                                saved_file_path = os.path.join(dataset_dir, subset, 'Parametrics', 'ParametricSetup1', dir, f'{i+1}_{j+1}.zip')
                            else:
                                raise ValueError('No parametric directory found in %s' % os.path.join(dataset_dir, subset))
                            if not os.path.exists(saved_file_path) or regen:
                                # get layer and column index
                                snp_cols = [f'SR({i+1},{j+1})_{k}' for k in range(nF)]
                                snp_cols.extend([f'SI({i+1},{j+1})_{k}' for k in range(nF)])
                                start = i * port * 2 + j * 2
                                end = i * port * 2 + j * 2 + 2
                                df = pd.DataFrame(np.array(snp_df.iloc[:, start:end]).T.reshape([1,-1]), columns=snp_cols)
                                df.to_pickle(saved_file_path, compression='zip')
                    time_end = time.time()
                    print(f'Finished writing {filepath} in {time_end - time_start} seconds')
                    return
                else:
                    raise ValueError('No parametric directory %s found in %s' % (filepath, parametric_dir))

            if os.path.isdir(parametric_dir):
                with Parallel(n_jobs=num_cores) as parallel:
                    parallel(delayed(read_s_parameters_multilayer_parallel)(
                        os.path.join(parametric_dir, dir, 'TransmissionLine.s%dp' % port),
                        dir,
                        port,
                    ) for dir in os.listdir(parametric_dir))
                
            else:
                raise ValueError('No parametric directory found in %s' % parametric_dir)
            
    return

def get_layer(
    port: int,
    port_each_layer: int=10,
    layer_n: int=5):
    # port: 0, 1, 2, ..., total_n - 1
    # layer: 0, 1, 2, 3, 4
    return port % (port_each_layer * layer_n) // port_each_layer

def get_column_index(
    port: int,
    port_each_layer: int=10,
    layer_n: int=5):
    # port: 0, 1, 2, ..., total_n - 1
    # column: 0, 1, 2, ..., column_n - 1
    return (port % (port_each_layer * layer_n) % port_each_layer) * 5 + 2

def calculate_coord(
    n,
    total_n,
    w,
    s):
    # n: 0, 1, 2, ..., total_n - 1
    total_length = total_n * w + (total_n - 1) * s
    coord_n = n * w + n * s + w / 2
    center_coord_n = coord_n - total_length / 2
    return center_coord_n

def generate_location(
    n1: int,
    n2: int,
    total_n: int,
    W1: float,
    W2: float,
    S1: float,
    S2: float):
    # n: 0, 1, 2, ..., total_n - 1
    center_coord_n1 = calculate_coord(n1, total_n, W1, S1)
    center_coord_n2 = calculate_coord(n2, total_n, W2, S2)
    return center_coord_n2 - center_coord_n1
    
def parse_data_inductor(
        dataset_dir,
        filepath
    ):
    # dataset_dir = '../Data/Dataset/Inductor_Results/S-Parameters_Inductor'
    # filepath = '../Data/Dataset/Inductor_Results/Inductor.zip'

    keys = []
    snp_headers = []
    snp_df_list = []
    val_list = ['w', 's', 'n', 'r']
    port = 2
    nF = 201

    # create the df headers
    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                for n in range(nF):
                    snp_headers.append('%s(%d,%d)_%d' % (s, i+1, j+1, n))

    key_i = 0
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.s2p'):
            
            val = {}

            for i, idx in enumerate(val_list):
                
                # val_name = file_name.split('_')[i].split('.')[0]                
                # val[idx] = float(val_name.replace(idx, '').replace('d', '.'))
                val_name = file_name.split('_')[i+4].split('.')[0]  
                val[idx] = float(val_name.replace('d', '.'))

            snp_df = pd.read_csv(os.path.join(dataset_dir, file_name), skiprows=8, delim_whitespace=True, header=None).loc[:, 1:]
            snp_df = pd.DataFrame(np.array(snp_df).reshape((1,-1), order='F'))
            if snp_df.shape[1] != 1608:
                continue
            keys.append(f'Inductor_{key_i}')
            key_i += 1
            snp_df.columns = snp_headers

            for idx in val_list:
                snp_df.loc[:, idx] = val[idx]

            snp_df_list.append(snp_df)
    
    # concatenate the dataframe
    df = pd.concat(snp_df_list)
    df = df.set_index(keys=pd.Index(keys))

    df.to_pickle(filepath, compression='zip')

def read_s_parameter(
        filepath,
    ):
    df = pd.read_pickle(filepath, compression='zip')
    print(df)

    val_list = ['w', 's', 'n', 'r']
    for val in val_list:
        print(f'{val} min: {df[val].min()} max: {df[val].max()}')
    
    return

def read_input_feature_xlsx(
    case, 
    read_template,
    template_file):

    # get columns
    if read_template:
        para_df = pd.read_excel(template_file, sheet_name='Mixed_N_Line_Stripline', skiprows=23)
        new_para_df = para_df.T.loc[para_df.T.index[1:]].copy()
        return para_df.T.loc['batch list'].to_numpy()
    
    else:
        current_dir = os.getcwd()
        home_dir = os.path.abspath(os.path.join(current_dir, '..'))
        data_dir = os.path.join(home_dir, 'Data')
        excel_dir = os.path.join(data_dir, '%s.xlsx' % case)
        
        para_df = pd.read_excel(excel_dir, sheet_name='Mixed_N_Line_Stripline', skiprows=23)
    
        # Set new df
        new_para_df = para_df.T.loc[para_df.T.index[1:]].copy()
        new_para_df.columns = para_df.T.loc['batch list'].to_numpy()
        new_para_df = new_para_df.set_index('save dir file name')
        new_para_df.index.name = None
        
        def split_func(x):
            return float(x.split(',')[0])
        new_para_df['W'] = new_para_df['W'].apply(split_func)
        
        return new_para_df

def concat_subset(dataset_dir, dataset_name):
    """Concatenate subsets together
    :param dataset_dir: dataset dir that contains the data
    :param dataset_name: dataset name
    """
    df_list = []
    for subset in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, subset)):
            filepath = os.path.join(dataset_dir, '%s_%s.zip' % (dataset_name, subset))
            print(f'reading file path {filepath}')
            print(pd.read_pickle(filepath).columns)
            df_list.append(pd.read_pickle(filepath))
    df = pd.concat(df_list)
    concat_filepath = os.path.join(dataset_dir, '%s_concat.zip' % dataset_name)
    df.to_pickle(concat_filepath, compression='zip')
    return 

def concat_subset_port_parallel(
    dataset_dir,
    dataset_name,
    ports: List[str],
    num_cores: int = 8,):
    with Parallel(n_jobs=num_cores) as parallel:
        parallel(delayed(concat_subset_port)(
            dataset_dir,
            dataset_name,
            port,
        ) for port in ports)

def concat_subset_port(
    dataset_dir,
    dataset_name,
    port: str,):
    df_list = []
    concat_filepath = os.path.join(dataset_dir, f'{dataset_name}_{port}_concat.zip')
    if os.path.exists(concat_filepath):
        return

    for subset in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, subset)):
            filepath = os.path.join(dataset_dir, '%s_%s.zip' % (dataset_name, subset))
            print(f'reading file path {filepath}')
            df = pd.read_pickle(filepath)
            df_columns = df.columns
            selected_columns = []
            for col in df.columns:
                if not (('SR' in col or 'SI' in col) and (port not in col)):
                    selected_columns.append(col)
            selected_df = df[selected_columns]
            df_list.append(selected_df)

    df = pd.concat(df_list)
    df.to_pickle(concat_filepath, compression='zip')
    return 

def concat_subset_port_parallel_multilayer(
    dataset_dir,
    dataset_name,
    ports: List[str],
    num_cores: int = 1,
    batch_num: int = 100):
    def concat_subset_port_multilayer(
        dataset_dir,
        dataset_name,
        port: str,
    ):
        df_list = []
        index_list = []
        concat_filepath = os.path.join(dataset_dir, f'{dataset_name}_{port}_concat.zip')
        if os.path.exists(concat_filepath):
            pass
        
        for subset in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, subset)):
                if os.path.exists(os.path.join(dataset_dir, subset, 'Optimetrics')):
                    parametric_dir = os.path.join(dataset_dir, subset, 'Optimetrics', 'ParametricSetup1')
                elif os.path.exists(os.path.join(dataset_dir, subset, 'Parametrics')):
                    parametric_dir = os.path.join(dataset_dir, subset, 'Parametrics', 'ParametricSetup1')
                
                for dir in os.listdir(parametric_dir):
                    if os.path.isdir(os.path.join(parametric_dir, dir)):
                        # split port (m,n) to two int numbers
                        m = int(port.split(',')[0].replace('(', ''))
                        n = int(port.split(',')[1].replace(')', ''))

                        snp_port_path = os.path.join(parametric_dir, dir, f'{m}_{n}.zip')
                        if os.path.exists(snp_port_path):
                            df = pd.read_pickle(snp_port_path)
                            df_list.append(df)
                            # write subset index 1, 2, 3, ...
                            index = int(subset.split('_')[-1]) * batch_num + int(dir)
                            index_list.append(f'{index}')
                        else:
                            print (f'No file {snp_port_path} found')
                            # raise ValueError(f'No file {snp_port_path} found')
        if len(df_list) != 0:
            df = pd.concat(df_list)
            df.index = index_list
            df.to_pickle(concat_filepath, compression='zip')    
        return

    with Parallel(n_jobs=num_cores) as parallel:
        parallel(delayed(concat_subset_port_multilayer)(
            dataset_dir,
            dataset_name,
            port,
        ) for port in ports)
            
def parse_inductor(
    args
):
    ds = args.dataset
    dataset_dir = os.path.join(args.dir, ds)
    filepath = os.path.join(dataset_dir, f'{ds}_concat.zip')
    parse_data_inductor(dataset_dir, filepath)
    read_s_parameter(filepath)
    return

def parse_stripline(
    args,
):
    ds = args.dataset
    dataset_dir = os.path.join(args.dir, ds)
    parse_data(
        dataset_dir, 
        args.port,
        args.nF,
        ds,
        args.diff)
    check_data(dataset_dir, ds)
    concat_subset(dataset_dir, ds)
    return

def parse_stripline_select(
    args,
):
    ds = args.dataset
    dataset_dir = os.path.join(args.dir, ds)
    parse_data(
        dataset_dir, 
        args.port,
        args.nF,
        ds,
        args.diff)
    check_data(dataset_dir, ds)
    port_list = generate_port_list(args.port)
    concat_subset_port_parallel(dataset_dir, ds, port_list)
    return

def parse_multilayer(
    args,
):
    ds = args.dataset
    dataset_dir = os.path.join(args.dir, ds)
    parse_data_metis_parallel_2(
        dataset_dir, 
        args.port,
        args.nF,
        ds,
        args.diff,
        args.regen)

    port_list = generate_port_list_multilayer(args.port)
    concat_subset_port_parallel_multilayer(dataset_dir, ds, port_list, args.batch_num)
    return

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='initial_sample', help='dataset name')
    parser.add_argument('-r', '--dir', type=str, default='/media/cadlab/Dailow2/Tml_data', help='dataset parent directory')
    parser.add_argument('-m', '--mode', type=str, default='Stripline', help='dataset parse mode, select from Stripline, Inductor')
    parser.add_argument('-p', '--port', type=int, default=4, help='port number')
    parser.add_argument('-n', '--nF', type=int, default=501, help='frequency point number')
    parser.add_argument('-b', '--batch_num', type=int, default=100, help='batch number of the dataset')
    parser.add_argument('--diff', action='store_true', default=False, help='different W and S')
    parser.add_argument('--regen', action='store_true', default=False, help='regenerate the data')
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()
    
    if args.mode == 'Stripline':
        parse_stripline(args)
    elif args.mode == 'Inductor':
        parse_inductor(args)
    elif args.mode == 'Stripline_Select':
        parse_stripline_select(args)
    elif args.mode == 'MultiLayer':
        parse_multilayer(args)
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main()