import os
import random
import pickle
import numpy as np
import pandas as pd

__all__ = [
    "read_input_feature_xlsx",
    "get_dataset_indices",
    "parse_frequency"
]

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
        
        return 

def get_dataset_indices(
    working_dir,
    df, 
    dataset_name, 
    k_fold,
    train_test_split_ratio,
    read_only,
    regen):
    
    # Randomly shuffle the indices
    index_file = os.path.join(working_dir, '../Data/Indices/index_%s.pkl' % dataset_name)
    
    # Read only for test
    if read_only and not os.path.exists(index_file):
        raise ValueError('Indices of %s not exist!' % dataset_name)
    
    # Indices file exists, read the file
    if os.path.exists(index_file) and not regen:
        with open(index_file, 'rb') as f:
            indices = pickle.load(f)

        return indices

    else:
        index_list = list(dict.fromkeys(df.index.get_level_values(0)))

        # Generate indices
        random.seed(42)
        np.random.shuffle(index_list)
            
        # Split the indices into 80% training set, 10% testing set and 10% validation set
        indices = {}

        if k_fold:
            if len(train_test_split_ratio) != 2:
                raise ValueError('Train test ratio must have 2 elements for k_fold mode')
            else:
                indices['train_idx'] = index_list[:int(len(index_list) * train_test_split_ratio[0])]
                indices['test_idx'] = index_list[int(len(index_list) * train_test_split_ratio[0]):]
        
        else:
            if len(train_test_split_ratio) != 3:
                raise ValueError('Train test ratio must have 2 elements for k_fold mode')
            else:
                indices['train_idx'] = index_list[:int(len(index_list) * train_test_split_ratio[0])]
                indices['val_idx'] = index_list[int(len(index_list) * train_test_split_ratio[0]):int(len(index_list) * (train_test_split_ratio[0] + train_test_split_ratio[1]))]
                indices['test_idx'] = index_list[int(len(index_list) * (train_test_split_ratio[0] + train_test_split_ratio[1])):]

        with open(index_file, 'wb') as f:
            pickle.dump(indices, f)

        return indices

def support_cols(
    model_name: str,
    port_num: int,
):
    return
    

def convert_Hz_to_num(hz: str):
    if hz == "Hz":
        return 1
    elif hz == "KHz":
        return 1e3
    elif hz == "MHz":
        return 1e6
    elif hz == "GHz":
        return 1e9
    else:
        raise ValueError('hz unit not correct')

def parse_frequency(f):
    f_start = f[0] * convert_Hz_to_num(f[1])
    f_end = f[2] * convert_Hz_to_num(f[3])
    f_step = f[4] * convert_Hz_to_num(f[5])
    f = np.arange(f_start, f_end + f_step, f_step)
    return f
