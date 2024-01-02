from cores import *
from utils import *

import numpy as np
import pandas as dp

from scipy.stats import skewnorm
from scipy.optimize import fsolve
from scipy.stats import truncnorm
from sklearn.linear_model import Ridge

import os
import argparse
import torch
import random
import matplotlib.pyplot as plt

TEST_RIDGE = False
TEST_COLUMN = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    return parser.parse_args()

def test_ridge(
    configs: Config,
    input_features: InputFeatures,
    output_col,
    device
):
    batch_size = configs.batch_size
    train_dataset, test_dataset = make_datasets(
        configs=configs,
        input_features=input_features,
        output_col=output_col,
        device=torch.device(device),
        multiport=False,
    )

    # Define max and min value for output normalization
    max_tensor, min_tensor = train_dataset.get_max_min()
    tensor_range = (max_tensor - min_tensor).item()
    train_dataset.set_max_min(max_tensor, min_tensor)
    train_dataset.normalize_output()

    # sample training dataset

    """
    Split train dataset a portion as validation
    """
    if hasattr(configs.datasets, 'sampling_method'):
        sample_num = int(configs.datasets.sampling_ratio * len(train_dataset))
        if configs.datasets.sampling_method == 'guided':
            train_ids, val_ids = train_dataset.sample_split(sample_num)
        elif configs.datasets.sampling_method == 'random':
            train_ids = np.random.choice(train_dataset.__len__(), sample_num, replace=False)
            val_ids = np.setdiff1d(np.arange(train_dataset.__len__()), train_ids)
        else:
            raise NotImplementedError
    else:
        raise ValueError("Please specify sampling method in configs.")
    print(f'The size of training dataset: {len(train_ids)}')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        sampler=train_subsampler)
    val_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=len(val_ids),
                    sampler=val_subsampler)
    
    X_train = train_dataset.parameters
    y_train = train_dataset.output
    
    clf = Ridge()
    clf.fit(X_train.cpu(), y_train.cpu())

    X_val, y_val = next(iter(val_dataloader))
    y_val_pred = clf.predict(X_val.cpu())


    X_example_train, y_example_train = next(iter(train_dataloader))
    X_example_val, y_example_val = next(iter(val_dataloader))

    # change X_example_train to numpy array
    X_example_train = X_example_train
    y_example_train = y_example_train
    X_example_val = X_example_val
    y_example_val = y_example_val

    X_example = torch.concatenate([X_example_train, X_example_val])
    output = clf.predict(X_example.cpu())
    y_example = torch.concatenate([y_example_train, y_example_val])
    fig = plot_X_y(input_features.frequency_np, 
                    y_example, 
                    output, 
                    0,
                    'ridge', 
                    configs)

def test_column(
    input_features,
    dfs,
):
    
    f = input_features.frequency_np
    # to see if each lines are the same
    if hasattr(configs.datasets, 'datasets'):
        for ds in configs.datasets.datasets:
            stackup_writer = StackupWriter.load_pickle(configs, ds)
    else:
        for i, ds in enumerate(configs.datasets.train_datasets): 
            stackup_writer = StackupWriter.load_pickle(configs, ds)
    stackups = stackup_writer.stackups 
    stackup_idx_list = []
    for i in range(dfs[0].shape[0]):
        df_idx = dfs[0].index[i]
        stack_idx = int(df_idx.split('_')[-1]) - 1
        stackup_idx_list.append(stack_idx)

    # calculate max, min
    max_v = np.max(dfs[0].to_numpy())
    min_v = np.min(dfs[0].to_numpy())
    print(max_v, min_v)
    for df_idx, stackup_idx in enumerate(stackup_idx_list):
        stackup = stackups[stackup_idx]
        for i in range(9 - 1): # each layer has 10 different W
            for j in range(i+1, 10 - 1):
                if stackup.layers[3].w_list[i * 5 + 2] == stackup.layers[3].w_list[j * 5 + 2]:
                    
                    # calculate the difference
                    diff = (dfs[i].iloc[df_idx, :].to_numpy() - dfs[j].iloc[df_idx, :].to_numpy())
                    mean_relative_diff = np.mean(diff / (max_v - min_v))
                    max_relative_diff = np.max(np.abs(diff / (max_v - min_v)))
                    print(f'stackup {stackup_idx} - column {i} and column {j} has the same W: {stackup.layers[3].w_list[i * 5 + 2]}, mean diff: {mean_relative_diff * 100:.4f}%, max diff: {max_relative_diff * 100:.4f}%')
                    if max_relative_diff <= 0.01 and df_idx >= 10:
                        continue
                    # plot the figure
                    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
                    ax.plot(f, dfs[i].iloc[df_idx, : f.shape[0]], label=f'column {i} sr')
                    ax.plot(f, dfs[j].iloc[df_idx, : f.shape[0]], label=f'column {j} sr')
                    ax.plot(f, dfs[i].iloc[df_idx, f.shape[0]:], label=f'column {i} si')
                    ax.plot(f, dfs[j].iloc[df_idx, f.shape[0]:], label=f'column {j} si')
                    ax.annotate(f'W = {stackup.layers[3].w_list[i * 5 + 2]}, Max Error = {max_relative_diff * 100:.4f}%', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                                horizontalalignment='left', verticalalignment='top')
                    ax.legend(loc='upper right')

                    pdf_dir, png_dir = configs.fig_dir
                    plt.savefig(f'{pdf_dir}/{stackup_idx}_column_{i}_{j}.pdf')
                    plt.savefig(f'{png_dir}/{stackup_idx}_column_{i}_{j}.png')


    return
    for stackup_idx, stackup in enumerate(stackups):
        # to see the W are same or not
        print(stackup.layers[3].w_list)
        for i in range(9): # each layer has 10 different W
            for j in range(i+1, 10):
                if stackup.layers[3].w_list[i * 5 + 2] == stackup.layers[3].w_list[j * 5 + 2]:
                    print(f'stackup {stackup_idx} - layer {i} and layer {j} has the same W: {stackup.layers[3].w_list[i * 5 + 2]}')
        # load S parameter

def func_mode_alpha(alpha, target_mode):
    sigma = alpha / np.sqrt(1 + alpha ** 2)
    skewness = (4 - np.pi) / 2 * (sigma * np.sqrt(2 / np.pi)) ** 3 / (1 - 2 * sigma ** 2 / np.pi) ** (3 / 2)
    mu_z = np.sqrt(2 / np.pi) * sigma
    eta_z = np.sqrt(1 - mu_z ** 2)
    mode = mu_z - skewness * eta_z / 2 - np.sign(alpha) / 2 * np.exp(-2 * np.pi * np.abs(alpha))
    return mode - target_mode


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def test_skew():

    fig, ax = plt.subplots(1, 1)
    
    start = 3
    nominal = 4
    end = 10

    mode = ((nominal - start) * 2 / (end - start) - 1) 

    alpha = fsolve(func_mode_alpha, np.sqrt(2 / np.pi), args=(mode), xtol=1e-06, maxfev=500)

    alpha = alpha[0]
    print(func_mode_alpha(alpha, mode))
    rv = skewnorm(alpha, scale=1)

    x = np.linspace(-1, 1, 100)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    
    r = skewnorm.rvs(alpha, size=1000)
    print(np.count_nonzero(np.abs(r) <= 1))
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.set_xlim([-1, 1])
    ax.legend(loc='best', frameon=False)

    plt.savefig('./test_func.png')

    return

def test_trunc():
    start = 3
    nominal = 4
    end = 10

    sd = max((end-nominal) / 3, (nominal-start) / 3)
    rv = get_truncated_normal(mean=nominal, sd=sd, low=start, upp=end)

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(start, end, 100)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    
    r = rv.rvs(size=1000)
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    print(np.count_nonzero((r >= start) & (r <= end)))
    ax.set_xlim([start, end])
    ax.legend(loc='best', frameon=False)

    plt.savefig('./test_func.png')

    return

def test_multiline():
    multiline_input_features = MultiLineInputFeatures('../Configs/Parameters/Stripline_Diff-Pair_10-parameter_0915.json')
    print(multiline_input_features.variable_num)
    print(multiline_input_features.sampled_variables)

    configs = Config()
    configs.load('../Configs/Stripline_Diff-Pair_10-parameter_0915.yml', recursive=True)
    sampler = UniformSampler(multiline_input_features)
    samples_np = sampler.sample(sample_num=200)
    print(samples_np)

    parameters_df = generate_parameters_combinations(
        multiline_input_features,
        configs,
        samples_np,
        False
    )
    print(parameters_df['W'])

    return

def plot_snp(
    snp_file: str,
    port: int
):
    snp_df = parse_generated_s_parameters(snp_file, port)

    freq_np = snp_df.loc[1:, 0].to_numpy()
    s_np = snp_df.loc[1:, 1:].to_numpy()

    fig, ax = plt.subplots(port, 3,  figsize=(24, 5 * port), constrained_layout=True)
    sri = ['SR', 'SI']
    for i in range(port):
        for j in range(2):
            ax[i, j].plot(freq_np, s_np[:, i * 2 + j])
        dB = np.log10(s_np[:, i * 2] ** 2 + s_np[:, i * 2 + 1] ** 2) * 10
        if np.any(dB < -100):
            ax[i, 2].axhline(y=-100, color='r', linestyle='--')
        if np.any(dB < -170):
            ax[i, 2].axhline(y=-170, color='r', linestyle='--')

        ax[i, 2].plot(freq_np, dB)

    plt.savefig(f'./snp_{port}.png')

def test_plot_snp():
    snp_file = '../TransmissionLine.s20p'
    port = 20
    plot_snp(snp_file, port)

def evaluate_ws(
    configs: Config):

    port = configs.node
    case = configs.case
    ds = configs.dataset_generation.name
    input_features = MultiLayerInputFeatures(configs.config.dir)
    df = pd.read_pickle(f'../Data/Dataset/{case}/{ds}_concat.zip', compression='zip')
    interest_port = (port // 2 + 1) // 2

    pdf_dir = os.path.join(configs.fig.dir, 'pdf', configs.trial)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join(configs.fig.dir, 'png', configs.trial)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir) 

    # plot sr, si
    # fixed wITRSITITiTI
    fig, ax = plt.subplots(11, 2, figsize=(24, 33), constrained_layout=True)
    sri = ['SR', 'SI']

    
    for k in range(2):
        cols = []
        for c in range(input_features.nF):
            cols.append(f'{sri[k]}({interest_port},{interest_port})_{c}')
        benchmark = df.loc['Stripline61', cols].to_numpy()
        for i in range(11):
            for j in range(11):
                line = f'Stripline{i * 11 + j + 1}'
                s_np = df.loc[line, cols].to_numpy()
                w_neigh, s_neigh = df.loc[line, 'W_0'], df.loc[line, 'S_0']
                ax[i, k].plot(input_features.frequency_np, s_np - benchmark, label=f'W_neigh = {w_neigh}, S_neigh = {s_neigh}', linestyle=':', linewidth=0.5)
                ax[i, k].set_ylim(-0.00025, 0.00025)
            ax[i, k].legend(loc='upper right')

    plt.savefig(f'{pdf_dir}/ws_3.pdf')
    plt.savefig(f'{png_dir}/ws_3.png')

    plt.close()
    fig, ax = plt.subplots(11, 2, figsize=(24, 33), constrained_layout=True)

    for k in range(2):
        cols = []
        for c in range(input_features.nF):
            cols.append(f'{sri[k]}({interest_port},{interest_port})_{c}')
        benchmark = df.loc['Stripline61', cols].to_numpy()
        for i in range(11):
            for j in range(11):
                line = f'Stripline{j * 11 + i + 1}'
                s_np = df.loc[line, cols].to_numpy()
                w_neigh, s_neigh = df.loc[line, 'W_0'], df.loc[line, 'S_0']
                ax[i, k].plot(input_features.frequency_np, s_np - benchmark, label=f'W_neigh = {w_neigh}, S_neigh = {s_neigh}', linestyle=':', linewidth=0.5)
                ax[i, k].set_ylim(-0.00025, 0.00025)
            ax[i, k].legend(loc='upper right')

    plt.savefig(f'{pdf_dir}/ws_4.pdf')
    plt.savefig(f'{png_dir}/ws_4.png')

    plt.close()


def test_distributed():

    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    rank = 0
    case = configs.case
    configs.trial = get_filename(args.setting)[0]
    
    # parse arguments
    distributed = True
    if distributed:
        # distributed, get SLURM variables
        rank = int(os.environ['SLURM_PROCID'])
        gpu = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        device = f"cuda:{gpu}"
        print(f"rank:{rank} gpu:{gpu} total tasks: {world_size} cuda device count:{torch.cuda.device_count()}")

    else:
        if torch.cuda.is_available():
            device = torch.device("cude:0")
        else:
            device = torch.device("cpu")


    # define port list for this process
    proc_port_list = []
    total_port_list = []

    for i in range(4):
        total_port_list += [f'SR(1,{i+1})', f'SI(1,{i+1})']

    for node in total_port_list:
        if (total_port_list.index(node) % world_size) == rank:
            proc_port_list.append(node)
    
    print(f"rank:{rank} gpu:{gpu} nodes:{proc_port_list}")

def test_port():
    
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    
    snp_file = '../Data/Dataset/Final_Cases/Final_Cases_1_1019_train/Final_Cases_1_1019_train_(1,1)_concat.zip'
    test_df = pd.read_pickle(snp_file, compression='zip')
    
    input_features = MultiLayerInputFeatures('../Configs/Parameters/Final_Cases_1_1019.json')

    output_col = 'SR(1,11)'

    train_dataset, test_dataset = make_datasets(
        configs=configs,
        input_features=input_features,
        output_col=output_col,
        device=torch.device("cuda:0")
    )
    print(train_dataset.parameters)
    print(train_dataset.output)

def test_multilayer():
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    
    
    def build_stackup(configs,
                      w,s):
        
        pattern = 'SSSSSSSSSS'

        Air1 = AirLayer(nominal=configs)
        Metal1 = GroundLayer(nominal=configs)
        Die1 = DieLayer(nominal=configs)
        Metal2 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die2 = DieLayer(nominal=configs)
        Metal3 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die3 = DieLayer(nominal=configs)
        Metal4 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die4 = DieLayer(nominal=configs)
        Metal5 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die5 = DieLayer(nominal=configs)
        Metal6 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die6 = DieLayer(nominal=configs)
        Metal7 = GroundLayer(nominal=configs)
        Air2 = AirLayer(nominal=configs)

        stackup = Stackup([Air1, Metal1, Die1, Metal2, Die2, Metal3, Die3, Metal4, Die4, Metal5, Die5, Metal6, Die6, Metal7, Air2])
        return stackup
    
    def build_stackup_2(configs,
                        w,s):
        
        pattern = 'SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS'

        Air1 = AirLayer(nominal=configs)
        Metal1 = GroundLayer(nominal=configs)
        Die1 = DieLayer(nominal=configs)
        Metal2 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die2 = DieLayer(nominal=configs)
        Metal3 = GroundLayer(nominal=configs)
        Air2 = AirLayer(nominal=configs)

        stackup = Stackup([Air1, Metal1, Die1, Metal2, Die2, Metal3, Air2])
        return stackup
    
    def build_stackup_3(configs,
                      w,s):
        
        pattern = 'GGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGG'

        Air1 = AirLayer(nominal=configs)
        Metal1 = GroundLayer(nominal=configs)
        Die1 = DieLayer(nominal=configs)
        Metal2 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die2 = DieLayer(nominal=configs)
        Metal3 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die3 = DieLayer(nominal=configs)
        Metal4 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die4 = DieLayer(nominal=configs)
        Metal5 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die5 = DieLayer(nominal=configs)
        Metal6 = SignalLayer(nominal=configs, pattern=pattern, w=w, s=s)
        Die6 = DieLayer(nominal=configs)
        Metal7 = GroundLayer(nominal=configs)
        Air2 = AirLayer(nominal=configs)

        stackup = Stackup([Air1, Metal1, Die1, Metal2, Die2, Metal3, Die3, Metal4, Die4, Metal5, Die5, Metal6, Die6, Metal7, Air2])
        return stackup
    
    layer_configs = MultiLayerInputFeatures(configs.config.dir)


    # sampler = UniformSampler(layer_configs)
    # samples_np = sampler.sample(sample_num=configs.dataset_generation.train.sampling_num)

    # print(samples_np)
    stackups = []

    for w in layer_configs.W.values:
        for s in layer_configs.S.values:
            stackups.append(build_stackup_3(layer_configs, w, s))

    stackup_writer = StackupWriter(stackups)
    stackup_writer.write_xlsx(configs, saved_name=f'{configs.dataset_generation.name}')
    stackup_writer.to_pickle(configs, saved_name=f'{configs.dataset_generation.name}')
    saved_name=f'{configs.dataset_generation.name}'
    stackups_pickle = os.path.join(os.getcwd(), configs.dataset_generation.dir, saved_name, f'{saved_name}.pkl')
    stackups = pickle.load(open(stackups_pickle, 'rb'))
    for stackup in stackups:
        for layer in stackup.layers:
            print(layer)

def test_read_dataframe():
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    configs.trial = get_filename(args.setting)[0]
  
    input_features = MultiLayerInputFeatures(configs.config.dir)
    dataset = make_datasets(configs=configs, input_features=input_features, output_col='SR(1,1)', device=torch.device('cuda'))
    
def test_sample_optimizer():

    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    case = configs.case
    
    # Parse input features config
    if configs.multilayers:
        input_features = MultiLayerInputFeatures(configs.config.dir)
    else:
        input_features = InputFeatures(configs.config.dir)

    # port_list = ['(1,1)', '(1,21)', '(1,51)', '(2,2)', '(2,12)']
    port_list = ['(1,1)']

    train_dataset, test_dataset = make_datasets(
                configs=configs,
                input_features=input_features,
                output_col='SR(1,1)',
                device=torch.device("cuda:0"),
    )
    sampler = GuidedSampler(input_features)
    complete_samples_np = sampler.complete_sample()
    sample_optimizer = SampleOptimizer(input_features)
    
    sample_range = complete_samples_np.shape[0]  
    sample_num = 100
    for i in range(100):
        indices = random.sample(range(sample_range), sample_num)
        samples_np = complete_samples_np[indices]
        u = sample_optimizer.range_uniformity(samples_np)
        print(u)


if __name__ == "__main__":
    

    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    rank = 0
    case = configs.case
    configs.trial = get_filename(args.setting)[0]
    # model path
    current_dir = os.getcwd()
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))
    model_dir = os.path.join(home_dir, 'Models', case)
    if not os.path.exists(model_dir) and (not configs.distributed or rank == 0):
        os.makedirs(model_dir) 
    configs.model_dir = model_dir

    # define fig dir as (pdf_dir, png_dir)
    pdf_dir = os.path.join(configs.fig.dir, 'pdf', configs.trial)
    if not os.path.exists(pdf_dir) and (not configs.distributed or rank == 0):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join(configs.fig.dir, 'png', configs.trial)
    if not os.path.exists(png_dir) and (not configs.distributed or rank == 0):
        os.makedirs(png_dir) 
    configs.fig_dir = (pdf_dir, png_dir)

    # parse input features config
    if configs.multilayers:
        input_features = MultiLayerInputFeatures(configs.config.dir)
    else:
        input_features = InputFeatures(configs.config.dir)

    port_list = ['(1,1)', '(2,2)', '(3,3)', '(4,4)', '(5,5)', '(6,6)', '(7,7)', '(8,8)', '(9,9)',]
    if TEST_COLUMN:
        dfs = []
        for output_col in port_list:
           ds = configs.datasets.datasets[0]
           df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{case}/{ds}/{ds}_{output_col}_concat.zip'), compression='zip')
           dfs.append(df)
        test_column(input_features, dfs)

    # port_list = ['(1,1)', '(1,21)', '(1,51)', '(2,2)', '(2,12)']
    

    output_cols = []
    for port in port_list:
        # define output cols and dataloaders
        if configs.model.name.split('_')[-1] == 'srsi': 
            output_cols.append(port)
        else:
            output_cols +=  [f'SR{port}', f'SI{port}']

    for output_col in output_cols:
        
        if TEST_RIDGE:
            test_ridge(
                configs=configs,
                input_features=input_features,
                output_col=output_col,
                device=torch.device("cuda:0")
                )

    """
    plot neighbour impact
    """
    # config_list = ['../Configs/Stripline_Diff-Pair_14-parameter_0925_DoE.yml',
    #                '../Configs/Stripline_Diff-Pair_18-parameter_0925_DoE.yml',
    #                '../Configs/Stripline_Diff-Pair_22-parameter_0925_DoE.yml']
    
    # for cfg in config_list:
    #     configs = Config()
    #     configs.load(cfg, recursive=True)
    #     # define fig dir as (pdf_dir, png_dir)
    #     configs.trial = get_filename(cfg)[0]
    #     evaluate_ws(configs)

