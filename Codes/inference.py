from math import e
import os
import argparse
import matplotlib.pyplot as plt
import torch
import json 

from cores import *
from utils import *
from preprocess_config import *
import pandas as pd

WRITE_MAX_MIN = False

class MultiLayerInferenceInputFeatures(InputFeatures):
    def __init__(
        self,
        config_dir
    ):
        with open(config_dir, encoding="utf-8") as f:
            config = json.load(f)

        self.frequency = config["Frequency"]
        self.layer_num = config["Layer"] 
        self.frequency_np, self.nF = self.get_frequency()
        for key, value in config["Parameters"].items():
            setattr(self, key, value)
        self.sampled_variables = config["Variables"]
        self.pattern = config["Pattern"]
        # support mode: Dot, Pair, Mesh
        self.differentlayer = config["DifferentLayer"]
        # self.combination = config["Combination"]
        # if self.combination not in ['Dot', 'Pair', 'Mesh', 'BBDesign']:
        #     raise ValueError('The combination mode is not supported.')
        
        self.variable_num = len(self.sampled_variables)
        self.sample_num = self.get_sample_num()
        
    def get_frequency(self):
        """
        Convert the frequency range specified in the input features to a numpy array of frequencies and the number of frequencies.

        Returns:
        frequency_np (numpy.ndarray): Array of frequencies.
        nF (int): Number of frequencies.
        """
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
        f_start = self.frequency[0] * convert_Hz_to_num(self.frequency[1])
        f_end = self.frequency[2] * convert_Hz_to_num(self.frequency[3])
        f_step = self.frequency[4] * convert_Hz_to_num(self.frequency[5])
        frequency_np = np.arange(f_start, f_end + f_step, f_step)
        nF = len(frequency_np)
        return frequency_np, nF
    
    def get_sample_num(self):
        """
        Get the number of samples.

        Returns:
        sample_num (int): Number of samples.
        """
        sample_num = len(getattr(self, self.sampled_variables[0]))
        for v in self.sampled_variables:
            if sample_num != len(getattr(self, v)):
                raise ValueError('The number of samples is not consistent.')
        return sample_num

    def create_input_tensor(self, 
                            device, 
                            train_input_features, 
                            port_per_layer: int = 10,
                            col_indices: List[int] = None,
                            multiport: bool = False,):
        if not multiport:
            self.parameters = torch.zeros(self.sample_num, self.variable_num)
            for i, v in enumerate(self.sampled_variables):
                self.parameters[:, i] = torch.tensor(getattr(self, v))
        else:
            self.parameters = torch.zeros(self.sample_num, self.variable_num - port_per_layer + 2)
            self.parameters[:, 0] = torch.tensor(getattr(self, f'W_{col_indices[0]+1}'))
            self.parameters[:, 1] = torch.tensor(getattr(self, f'W_{col_indices[1]+1}'))
            v_i = 2
            for v in self.sampled_variables:
                if not (v != 'W' and v.split('_')[0] != 'W'):
                    self.parameters[:, v_i] = torch.tensor(getattr(self, v))
                    v_i += 1
        self.parameters = self.parameters.to(torch.float32).to(device)

        # load train input features
        # define min and max for normalization
        if self.variable_num != len(train_input_features.sampled_variables):
            raise ValueError('The number of sampled variables is not consistent.')

        if not multiport:
            config_tensor_min = torch.zeros([1, self.variable_num])
            config_tensor_max = torch.zeros([1, self.variable_num])
            # normalize method: Min-Max
            for i, v in enumerate(train_input_features.sampled_variables):
                if getattr(train_input_features, v).is_list():
                    config_tensor_min[0, i] = min(getattr(train_input_features, v).min)
                    config_tensor_max[0, i] = max(getattr(train_input_features, v).max)
                else:
                    config_tensor_min[0, i] = getattr(train_input_features, v).min
                    config_tensor_max[0, i] = getattr(train_input_features, v).max

            self.config_tensor_min = torch.Tensor(config_tensor_min).to(device)
            self.config_tensor_max = torch.Tensor(config_tensor_max).to(device)
        else:
            config_tensor_min = torch.zeros([1, self.variable_num - port_per_layer + 2])
            config_tensor_max = torch.zeros([1, self.variable_num - port_per_layer + 2])
            v_i = 2
            min_W = 10000
            max_W = 0
            W_list = [f"W_{i+1}" for i in range(port_per_layer)]
            for W in W_list:
                if getattr(train_input_features, W).is_list():
                    min_W = min(min_W, min(getattr(train_input_features, W).min))
                    max_W = max(max_W, max(getattr(train_input_features, W).max))
                else:
                    min_W = min(min_W, getattr(train_input_features, W).min)
                    max_W = max(max_W, getattr(train_input_features, W).max)
            config_tensor_min[0, 0] = min_W
            config_tensor_max[0, 0] = max_W
            config_tensor_min[0, 1] = min_W
            config_tensor_max[0, 1] = max_W
            for i, v in enumerate(train_input_features.sampled_variables):
                if not (v.split('_')[0] == "W" and v != "W"):
                    if getattr(train_input_features, v).is_list():
                        config_tensor_min[0, v_i] = min(getattr(train_input_features, v).min)
                        config_tensor_max[0, v_i] = max(getattr(train_input_features, v).max)
                    else:
                        config_tensor_min[0, v_i] = getattr(train_input_features, v).min
                        config_tensor_max[0, v_i] = getattr(train_input_features, v).max
                    v_i += 1
            
            self.config_tensor_min = torch.Tensor(config_tensor_min).to(device)
            self.config_tensor_max = torch.Tensor(config_tensor_max).to(device)

        self.parameters = (self.parameters - self.config_tensor_min) / (self.config_tensor_max - self.config_tensor_min)
        self.device = device
        return self.parameters

    

def inference(
    configs: Config,
    train_input_features,
    inference_input_features,
    output_col: str,
    # max_min_dict: dict,
    device,
    load_model_col: str = None,
    multiport: bool = False,
    port_num: int = 1,
):
    """
    Inference for a single output column
    """
    if load_model_col is None:
        load_model_col = output_col
    if hasattr(configs.model, 'read'):
        load_model = configs.model.read
    else:
        load_model = configs.trial

    if not multiport:
        best_path = os.path.join(configs.model_dir, f'{load_model}_{load_model_col}_best.pt')
    else:
        best_path = os.path.join(configs.model_dir, f'{load_model}_{load_model_col}_multiport_best.pt')


    output_log_name = f'{configs.trial}_{output_col}'
    port_per_layer=inference_input_features.pattern.count('S')

    if not multiport:
        in_channels = inference_input_features.variable_num
    else:
        in_channels = inference_input_features.variable_num - port_per_layer + 2
    
    # load model
    net =  make_model(
        configs=configs,
        #in_channels=len(input_features.sampled_variables) if configs.multilayers else input_features.variable_num,
        in_channels=in_channels,
        out_channels=inference_input_features.nF,
        multiport=port_num if multiport else 0,
        device=device
    )
    if not os.path.exists(best_path):
        raise ValueError(f"Model not trained {best_path}!")
    
    best = torch.load(best_path)
    net.load_state_dict(best['model_state_dict'])
    net.to(device)

    max_tensor, min_tensor = best['max_min']
    # calculate which row
    row = 0

    # inference
    net.eval()
    output_list = []
    for i in range(port_per_layer):

        # load data
        X_inf = inference_input_features.create_input_tensor(device, 
                                                            train_input_features,
                                                            port_per_layer=port_per_layer,
                                                            col_indices=[i,i],
                                                            multiport=True
                                                            )
        with torch.no_grad():
            Y_inf = net(X_inf)
            Y_inf = Y_inf * (max_tensor - min_tensor) + min_tensor
            Y_inf = Y_inf.cpu().numpy()
            output_list.append(Y_inf)
    output_np = np.concatenate(output_list, axis=1)
    output_col = []
    for i in range(port_per_layer):
        for j in range(inference_input_features.layer_num * 2):
            for ri in ['SR', 'SI']:
                output_col.append(f'{ri}({i + 1},{j * port_per_layer + i + 1})')
    for i in range(output_np.shape[0]):
        output_df = pd.DataFrame(np.reshape(output_np[i], (len(output_col), -1)).T, columns=output_col)
    print(output_df)
    # write inference results
    #  np.savetxt(os.path.join(configs.results_dir, f'{output_log_name}.txt'), Y_inf, delimiter=',')
    return

def load_train_dataset_max_and_min(
    configs: Config,
    output_col: str,
):
    
    dataset_mode = configs.datasets.mode
    if dataset_mode == 'final':   
        if 'SR' in output_col or 'SI' in output_col:
            load_file_col = output_col[2:]
        else:
            load_file_col = output_col 
        df_list = []
        if hasattr(configs.datasets, 'datasets'):
            for ds in configs.datasets.datasets:
                df = pd.read_pickle(os.path.join(f'../Data/Dataset/{configs.case}/{ds}/{ds}_{load_file_col}_concat.zip'), compression='zip')
                df_list.append(df)
        else:
            for ds in configs.datasets.train_datasets: 
                df = pd.read_pickle(os.path.join(f'../Data/Dataset/{configs.case}/{ds}/{ds}_{load_file_col}_concat.zip'), compression='zip')
                df_list.append(df)
    else:
        raise NotImplementedError
    df_array = pd.concat(df_list).to_numpy()
    print(np.max(df_array), np.min(df_array))
    return np.max(df_array), np.min(df_array)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    return parser.parse_args()

def read_max_min(configs: Config):
    with open(os.path.join(configs.model_dir, f'{configs.trial}_max_min.txt'), 'r') as f:
        data = f.read() 
    json_data = json.loads(data)
    return json_data

def write_max_min(
    configs: Config,
    output_cols: List[str],
):
    max_min_dict = {}
    for output_col in output_cols:
        # Define max and min value for output normalization
        max_min_dict[output_col] = load_train_dataset_max_and_min(configs, output_col)
    with open(os.path.join(configs.model_dir, f'{configs.trial}_max_min.txt'), 'w') as f:
        json.dump(max_min_dict, f)

    # reading the data from the file 
    with open(os.path.join(configs.model_dir, f'{configs.trial}_max_min.txt'), 'r') as f:
        data = f.read() 
    print(data)
    return

def main():
    
    # parse arguments
    args = parse_args()
    configs, train_input_features = preprocess_config(args, write_results=True)
    inference_input_features = MultiLayerInferenceInputFeatures(configs.inference_config.dir)

    # port_list = []
    # layer_num = train_input_features.layer_num
    # port_num = 20
    # port_per_layer = (port_num // 2) // layer_num 
    # if port_num % layer_num != 0:
    #     raise ValueError("port number must be integer number of layer number")
    # port_list = []
    # for i in range(layer_num):
    #     for j in range(2):
    #         for m in range(2):
    #             for k in range(i, layer_num):
    #                 port_list.append(f'({i * port_per_layer + 1},{k * port_per_layer + 1 + j + m * port_num // 2})')
    
    port_list = ['(1,1)']

    # if WRITE_MAX_MIN:
    #     write_max_min(
    #         configs=configs,
    #         output_cols=port_list,
    #     )

    # max_min_dict = read_max_min(configs)
    
    output_cols = []
    for port in port_list:
        # define output cols and dataloaders
        if configs.model.name.split('_')[-1] == 'srsi': 
            output_cols.append(port)
        else:
            output_cols +=  [f'SR{port}', f'SI{port}']

    for output_col in output_cols:

        inference(
            configs=configs,
            train_input_features=train_input_features,
            inference_input_features=inference_input_features,
            output_col=output_col,
            # max_min_dict=max_min_dict,
            device=torch.device("cuda:0"),
            load_model_col='(1,1)',
            multiport=True,
            port_num=10
            )

    return

if __name__ == "__main__":
    main()
    