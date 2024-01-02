import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from cores import *

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='configuration file')
    parser.add_argument('-m', '--mode', type=str, default='rlgc', help='write certain type of files')
    # parser.add_argument('--rlgc', type=str, default=None, help='dataset name')
    return parser.parse_args()

def parse_column(
    extracted_df,
    index,
    column,
    nF
):
    columns_to_extract = []
    for i in range(nF):
        columns_to_extract.append(f'{column}_{i}')
    
    return extracted_df.loc[index, columns_to_extract].to_numpy().reshape(nF, 1)


def write_rlgc(
    configs,
    rlgc_file):

    rlgc_out_dir = os.path.join('../Data/Out', configs.case)
    if not os.path.exists(rlgc_out_dir):
        os.makedirs(rlgc_out_dir) 

    # Parse input features config
    input_features = InputFeatures(os.path.join(os.getcwd(), configs.config.dir))

    input_cols, variables_range, decimals = input_features.get_variable_range()
    f = parse_frequency(input_features.frequency)
    nF = len(f)

    #############################
    #### Write all rlgc data ####
    #############################
    '''
    port_list = [[1,1], [1,2], [2,1], [2,2]]
    df_cols = []
    for k in 'R', 'L', 'C', 'G':
        for i,j in port_list:
            df_cols.append(f'{k}({i},{j})')
    
    df = pd.read_pickle(rlgc_file)
    for index, row in df.iterrows():
        output_np = f.reshape(nF, 1)
        for col in df_cols:
            col_np = parse_column(df, index, col, nF)
            output_np = np.concatenate([output_np, col_np], axis=1)
        output_df = pd.DataFrame(output_np, columns=['F'] + df_cols)
        output_df_file = os.path.join(rlgc_out_dir, f'TransmissionLine_{index}_Inference.rlgc')
        output_df.to_csv(output_df_file, sep=' ', index=False)
        # Read the existing content of the file
        with open(output_df_file, 'r') as file:
            existing_content = file.read()

        # Open the file in write mode
        with open(output_df_file, 'w') as file:
            # Write the new content (two rows) at the beginning of the file
            new_content = '#Hz R R(Ohm/m) L(H/m) C(F/m) G(S/m)\n' + '#N=2\n'
            file.write(new_content)
            
            # Write the original content back to the file
            file.write(existing_content)
    
    '''
    
    ##################################
    #### Select the closeset data ####
    ##################################

    # load test dataset of truth values
    working_dir = os.getcwd()
    test_dfs = []
    train_dfs = []
    config_file = os.path.join(os.getcwd(), configs.config.dir)
    for i, ds in enumerate(configs.datasets.name):
        test_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_test_concat.zip' %(ds, ds)), compression='zip')
        test_df.index = test_df.index + '_%s' % ds
        test_dfs.append(test_df)
        train_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_train_concat.zip' %(ds, ds)), compression='zip')
        train_df.index = train_df.index + '_%s' % ds
        train_dfs.append(train_df)

    test_df = pd.concat(test_dfs)
    train_df = pd.concat(train_dfs)

    test_parameters = test_df.loc[:, input_cols]
    train_parameters = train_df.loc[:, input_cols]
    config_tensor_min = []
    config_tensor_max = []
    
    with open(config_file) as file:
        config = json.load(file)

    for input_col in input_cols:
        test_parameters.loc[:, input_col] = (test_parameters[input_col] - config[input_col]['min']) / (config[input_col]['max'] - config[input_col]['min'])
        train_parameters.loc[:, input_col] = (train_parameters[input_col] - config[input_col]['min']) / (config[input_col]['max'] - config[input_col]['min'])
        
        config_tensor_min.append(config[input_col]['min'])
        config_tensor_max.append(config[input_col]['max'])

    train_parameters_np = train_parameters.to_numpy()

    import shutil

    selected_out_dir = os.path.join('../Data/Out', configs.case+'_selected')
    if not os.path.exists(selected_out_dir):
        os.makedirs(selected_out_dir) 

    for index, row in test_df.iterrows():
        # calculate the distance
        delta = test_parameters.loc[index, input_cols].to_numpy().reshape(1,-1) - train_parameters_np
        distance = np.sqrt(np.sum(delta ** 2, axis = -1))
        if (min_distance := np.min(distance)) < 0.3:
            output_df_file = os.path.join(rlgc_out_dir, f'TransmissionLine_{index}_Inference.rlgc')
            selected_df_file = os.path.join(selected_out_dir, f'TransmissionLine_{index}_Inference.rlgc')
            shutil.copy(output_df_file, selected_df_file)

def main():
    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)

    if args.mode == 'rlgc':
        file_name = f'../Data/Out/{configs.case}_2_Inference_RLGC.zip'
        write_rlgc(configs, file_name)

if __name__ == "__main__":
    main()
