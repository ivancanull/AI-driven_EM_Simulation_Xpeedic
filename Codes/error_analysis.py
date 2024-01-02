import os
import argparse

import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from scipy import integrate
from utils import *
from cores import *
def smooth_error(
    X,
    error,
    variables,
    variables_range
):
    """
    Smooth the function between X and error.
    :param X: the input features
    :param error: the error metric
    :return PDF_model: the probability distribution 
        function model of the error 
        corresponding to the input 
        feartures, of shape(feature_num)
    """
    
    space_num = 20
    
    X = X.detach().numpy()
    error = error.detach().numpy()
    
    PDF_model = []
    in_channels = len(variables)
    
    error_mean = np.zeros([in_channels, space_num])
    X_mean = np.zeros([in_channels, space_num])

    ranges_swap = np.swapaxes(variables_range, 0, 1)
    X = X * (ranges_swap[1, :] - ranges_swap[0, :]) + ranges_swap[0, :]
    print(X)


    for idx, feature in enumerate(variables):    
        X_range = np.linspace(variables_range[idx, 0], variables_range[idx, 1], num=space_num+1, endpoint=True)
        for i in range(space_num):
            X_mean[idx, i] = (X_range[i] + X_range[i+1]) / 2
            idx_split = (X_range[i] < X[..., idx]) & (X[..., idx] < X_range[i+1])
            if idx_split.any():
                error_mean[idx, i] = np.mean(error[idx_split])

        # build a smooth predictor for mse error distribution
        degree = 3
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
        model.fit(X_mean[idx].reshape(-1, 1), error_mean[idx].reshape(-1, 1))
        print(error_mean[idx])
        result, _ = integrate.quad(
            lambda x: float(model.predict(np.array(x).reshape(-1, 1))[0,0]), 
            variables_range[idx, 0], 
            variables_range[idx, 1])
        
        PDF_model.append((model, 1.0 / result))
        
    return PDF_model

def generate_samples(
    X,
    input_features,
    mse_error_mean,
    sample_path,
    configs
):
    """
    Generate sample ranges using scaled LHS
    :param X: the X ranges
    :param mse_error_mean: the average mse error of SR and SI
    :param post_fix: '(i,j)' as a string
    :param args: arguments passed from test.py
    """
    variables, variables_range, decimals = input_features.get_variable_range()

    PDF_model = smooth_error(X.cpu(), mse_error_mean.cpu(), variables, variables_range)
    
    in_channels = len(variables)

    # define new sample number
    sample_num = configs.dataset_generation.train.sampling_num
    sample_range = np.zeros((len(PDF_model), sample_num + 1))
    for idx, (pdf, scale) in enumerate(PDF_model):
        
        # define the start and stop of the sample range
        sample_range[idx, 0] = variables_range[idx, 0]
        sample_range[idx, sample_num] = variables_range[idx, 1]
        
        # define resolution to search for the resampled area
        resolution = 5000
        X = np.linspace(variables_range[idx, 0], variables_range[idx, 1], num=resolution+1, endpoint=True)
        stop = 1
        for i in range(sample_num - 1):
            # determine the location of the resampled area
            # find the small area where integration is 1 / sample_num
            while integrate.quad(
                lambda x: float(pdf.predict(np.array(x).reshape(-1, 1))[0,0]), 
                X[0], 
                X[stop])[0] * scale < 1.0 / sample_num * (i + 1):
                stop += 1
            sample_range[idx, i + 1] = X[stop - 1]
    
    torch.save({
        'sample_num': sample_num,
        'sample_range': sample_range,
    }, f"{sample_path}/sample.pt")
    return
    
def analyze_error(
    f,
    input_features,
    X,
    y,
    pred,
    output_cols,
    sample_path,
    configs
):
    """
    Analyze the error
    :param f: frequency points
    :param X: input features shape of (N, inChannel)
    :param y: truth output features shape of (N, nF, outChannel)
    :param pred: predicted output features shape of (N, nF, outChannel)
    :param output_cols: output columns name
    :param args: arguments passed from test.py
    """

    # print (f"X shape: {X.shape}, y shape: {y.shape}")
    absolute_error = pred - y
    mse_error = torch.mean(absolute_error ** 2, dim=-1, keepdim=True)

    # Worst cases
    sr_error, sr_idx = torch.sort(mse_error[:, 0, :], dim=0, descending=True)
    si_error, si_idx = torch.sort(mse_error[:, 1, :], dim=0, descending=True)
    sr_idx = torch.squeeze(sr_idx)
    si_idx = torch.squeeze(si_idx)
    fig = plot_X_y_test(f, y[sr_idx[0:5]], pred[sr_idx[0:5]], output_cols, configs)
        
    # Best cases
    sr_error, sr_idx = torch.sort(mse_error[:, 0, :], dim=0, descending=False)
    si_error, si_idx = torch.sort(mse_error[:, 1, :], dim=0, descending=False)
    sr_idx = torch.squeeze(sr_idx)
    si_idx = torch.squeeze(si_idx)
    # fig = plot_X_y_test(f, y[sr_idx[0:5]], pred[sr_idx[0:5]], output_cols, configs)

    # Calculate error and generate new samples
    mse_error_mean = torch.mean(mse_error, dim=-2, keepdim=True).cpu()
    generate_samples(X, input_features, mse_error_mean, sample_path, configs)
    
    return

def _analysis_epoch(
    f,
    input_features,
    net: torch.nn.Module,
    criterion,
    writer: SummaryWriter,
    output_cols_name: str,
    train_dataset: CustomDataset,
    train_dataloader: torch.utils.data.DataLoader,
    train_losslogger: ValLossLogger,
    sample_dir,
    configs,
):

    train_losslogger.clean()
    net.eval()

    truth = []
    preds = []
    features = []
    for i, (X_train, y_train) in enumerate(train_dataloader, 0):

        output = net(X_train)

        # Set the loss functions
        train_loss = criterion(output, y_train)

        y_train = train_dataset.denormalize_output(y_train)
        output = train_dataset.denormalize_output(output)

        X_train = train_dataset.downsample_input(variable_num=input_features.get_variable_num(), X=X_train)
        features.append(torch.clone(X_train.detach()))
        truth.append(torch.clone(y_train.detach()))
        preds.append(torch.clone(output.detach()))
        
        # Calculate the loss function
        train_losslogger.update(train_loss.item(), X_train.shape[0])
        train_losslogger.update_error(y_train, output, X_train.shape[0])
        
    print('Test Loss: %6f'
        % (train_losslogger.loss)
    )
    # Epoch done, write loss
    if writer is not None:
        sr_mean_error, si_mean_error, sr_max_error, si_max_error = train_losslogger.error
        writer.add_scalar(f'{output_cols_name}_Loss/analysis', train_losslogger.loss)
        writer.add_scalar(f'{output_cols_name}_Error/sr_mean', si_mean_error)
        writer.add_scalar(f'{output_cols_name}_Error/si_mean', sr_mean_error)
        writer.add_scalar(f'{output_cols_name}_Error/sr_max', sr_max_error)
        writer.add_scalar(f'{output_cols_name}_Error/si_max', si_max_error)

        # Use Mean(SR,SI) mean error as metric
    
    features = torch.cat(features)
    truth = torch.cat(truth)
    preds = torch.cat(preds)

    features = train_dataset.denormalize_features(features.cpu())

    # Define and sort relative error
    absolute_error = preds - truth
    range = torch.max(truth, dim=(-1), keepdim=True)[0] - torch.min(truth, dim=(-1), keepdim=True)[0]
    relative_error = torch.mean(torch.abs(absolute_error) / range, dim=(1, 2),)
    mae_error, mae_idx = torch.sort(relative_error, dim=0, descending=True)
    
    # Plot mae histogram
    fig, ax = plt.subplots()
    ax.hist(mae_error.cpu().numpy().reshape(-1), bins=50)
    ax.set_xlabel('Mean Absolute Error')
    ax.set_ylabel('Number')
    ax.set_title('Error distribution of training dataset')
    pdf_dir, png_dir = configs.fig_dir
    plt.savefig(f'{pdf_dir}/error_histogram_{output_cols_name}.pdf')
    plt.savefig(f'{png_dir}/error_histogram_{output_cols_name}.png')

    print(mae_error.cpu().numpy().reshape(-1)[0:10])

    print(features[mae_idx[0:100]].numpy().reshape(100, -1))
    print(truth[mae_idx[0:10]].shape)
    fig = plot_X_y_test(f, truth[mae_idx[0:10]], preds[mae_idx[0:10]], output_cols_name, configs, prefix='error_analysis')



    return
    mse_error = torch.mean(absolute_error, dim=(1, 2), keepdim=True)

    # Worst cases
    sr_error, sr_idx = torch.sort(mse_error[:, 0, :], dim=0, descending=True)
    si_error, si_idx = torch.sort(mse_error[:, 1, :], dim=0, descending=True)
    sr_idx = torch.squeeze(sr_idx)
    si_idx = torch.squeeze(si_idx)
    fig = plot_X_y_test(f, y[sr_idx[0:5]], preds[sr_idx[0:5]], output_cols_name, configs)
        
    # Best cases
    sr_error, sr_idx = torch.sort(mse_error[:, 0, :], dim=0, descending=False)
    si_error, si_idx = torch.sort(mse_error[:, 1, :], dim=0, descending=False)
    sr_idx = torch.squeeze(sr_idx)
    si_idx = torch.squeeze(si_idx)
    # fig = plot_X_y_test(f, y[sr_idx[0:5]], pred[sr_idx[0:5]], output_cols, configs)

    # Calculate error and generate new samples
    mse_error_mean = torch.mean(mse_error, dim=-2, keepdim=True).cpu()



    analyze_error(
        f=f,
        input_features=input_features,
        X=features,
        y=truth,
        pred=preds,
        output_cols=output_cols_name,
        sample_path=sample_dir,
        configs=configs
    )

    if writer is not None:
        writer.flush()


def error_analysis(
    configs: Config,
    input_features: InputFeatures,
    output_cols,
    writer: SummaryWriter,
    device
):
    # Get parameters
    batch_size = configs.batch_size
    output_cols_name = output_cols[0][-5:]
    # ckpt_path = os.path.join(configs.model_dir, f'{output_cols_name}_ckpt.pt')
    best_path = os.path.join(configs.model_dir, f'{output_cols_name}_best.pt')

    output_log_name = f'{configs.case}_{output_cols_name}_error_analysis'

    variables, variables_range, decimals = input_features.get_variable_range()
    f = parse_frequency(input_features.frequency)
    nF = len(f)

    in_channels = len(variables)
    k_fold = configs.datasets.k_fold        
    
    if configs.datasets.split_mode == 0:
        # Split the train and test dataset
        if k_fold:
            train_dataset, _, test_dataset = make_datasets_v0(
                configs=configs,
                input_cols=variables,
                output_cols=output_cols,
                nF=nF,
                read_only=False,
                device=torch.device(device)
            )
            k_fold_num = configs.datasets.k_num
            kfold = KFold(n_splits=k_fold_num, shuffle=True)
        else:
            train_dataset, val_dataset, test_dataset = make_datasets_v0(
                configs=configs,
                input_cols=variables,
                output_cols=output_cols,
                nF=nF,
                read_only=False,
                device=torch.device(device)
            )

    elif configs.datasets.split_mode == 1: 
        # The train and test dataset are already splited in dataset generation
        train_dataset, test_dataset = make_datasets_v1(
            configs=configs,
            input_cols=variables,
            output_cols=output_cols,
            nF=nF,
            device=torch.device(device)
        )
    
    else: 
        raise NotImplementedError

    print(f"Parameters shape: {train_dataset.parameters.shape}")

    in_channels = train_dataset.parameters.shape[-1]

    # Define max and min value for output normalization
    max_tensor, min_tensor = train_dataset.get_max_min()
    train_dataset.normalize_output()

    net = make_model(
        configs=configs,
        in_channels=in_channels,
        out_channels=nF,
        device=torch.device(device)
    )
    optimizer = make_optimizer(
        net.parameters(),
        configs=configs,
    )
    scheduler = make_scheduler(
        optimizer=optimizer,
        configs=configs,
    )

    criterion = make_criterion(
        configs=configs,
    )

    # Load best model        
    if not os.path.exists(best_path):
        raise ValueError(f"Model not trained {best_path}!")
    else:
        best = torch.load(best_path)
        # split = best['split']
        # fold_continue = best['fold']
        # sub_epoch_continue = best['sub_epoch']
        net.load_state_dict(best['model_state_dict'])
        optimizer.load_state_dict(best['optimizer_state_dict'])
        scheduler.load_state_dict(best['scheduler_state_dict'])

        train_dataloader = torch.utils.data.DataLoader(
                            train_dataset,)
            
        train_losslogger = ValLossLogger()
        _analysis_epoch(f=f, input_features=input_features, net=net, criterion=criterion,
                    writer=writer, output_cols_name=output_log_name, train_dataset=train_dataset,
                    train_dataloader=train_dataloader, train_losslogger=train_losslogger,
                    sample_dir=configs.out_dir, configs=configs)
        return
        # Get train examples
        example_num = configs.fig.example_num
        train_ids = np.arange(len(train_dataset))
        example_train_ids = np.random.permutation(train_ids)[0:example_num]
        X_example, y_example = train_dataset[example_test_ids]
        output = net(X_example)

        # Denormalize
        y_example = test_dataset.denormalize_output(y_example)
        output = test_dataset.denormalize_output(output)
    
        # Plot and save figure
        fig = plot_X_y_test(f, y_example, output, output_cols_name, configs)
        if writer is not None:
            writer.add_figure(f'{output_log_name}_Pred', fig)

        # Training is finished.
        return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    return parser.parse_args()


def parse_generated_s_parameters(
    s
):
        
    # Initialize an empty list to store processed rows
    processed_rows = []

    # Read the CSV file line by line
    # Replace 'your_compact_data.csv' with your actual file name
    with open(s, 'r') as file:
        rows = file.readlines()

    # Process each row and compact rows
    compact_row = []
    for row in rows:
        # Remove leading and trailing whitespace
        row = row.strip('\n').strip()
        
        # Skip empty rows
        if not row or row.startswith(('!', '#')) or row == '\t':
            continue
        
        
        # Check if the row begins with a tab
        if row.startswith('\t'):
            # Split by tab and remove the empty first item
            items = row.split('\t')[1:]
        else:
            # Split by tab
            items = row.split('\t')
        
        # Append items to compact_row
        compact_row.append(items)
        
        # If compact_row has 4 items, add to processed_rows and reset compact_row
        if len(compact_row) == 4:
            processed_rows.append([float(number) for sublist in compact_row for number in sublist])
            compact_row = []

    # Concatenate processed rows into a DataFrame
    final_data = np.array(processed_rows)
    return final_data

def analyze_s_parameters_error():
    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    
    case = configs.case
    port = 4

    png_dir = os.path.join('../Data/Out', configs.case+'_png')
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    pdf_dir = os.path.join('../Data/Out', configs.case+'_pdf')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir) 
    
    # Parse input features config
    input_features = InputFeatures(os.path.join(os.getcwd(), configs.config.dir))

    variables, variables_range, decimals = input_features.get_variable_range()
    f = parse_frequency(input_features.frequency)
    nF = len(f)

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

    # load inferenced dataset
    inferenced_data_dir = os.path.join('../Data/Out', case)
    snp_headers = []
    # create the df headers
    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                for n in range(nF):
                    snp_headers.append('%s(%d,%d)_%d' % (s, i+1, j+1, n))
    snp_np_list = []
    indices = []
    # read lines in the snp files
    for file in os.listdir(inferenced_data_dir):
        if file.endswith(f'.s{port}p'):
            # Find the indices of the first and last hyphens
            first_hyphen_index = file.index('_')
            last_hyphen_index = file.rindex('_')

            # Extract the content between the first and last hyphens
            index = file[first_hyphen_index + 1:last_hyphen_index]
            indices.append(index)
            snp_np = np.reshape(parse_generated_s_parameters(os.path.join(inferenced_data_dir, file))[:, 1:], (1, -1), order='F')
            snp_np_list.append(snp_np)
    
    snp_np_total = np.concatenate(snp_np_list, axis=0)
    snp_df = pd.DataFrame(snp_np_total, index=indices, columns=snp_headers)
    selected_test_df = test_df.loc[snp_df.index]

    # compare the two datasets
    # individual error rate
    # mean error rate
    SRI = ['SR', 'SI']

    test_max = np.zeros((port, port, 2))
    test_min = np.zeros((port, port, 2))

    # define whether to ignore F = 0Hz
    ignore_f0 = True

    error_analysis_np = np.zeros([len(indices), 4*4*2*2 + 4])
    error_analysis_header = []

    cols = []
    for i in range(port):
        for j in range(port):
            for s in range(2):
                index = f'{SRI[s]}({i+1},{j+1})'
                if ignore_f0:
                    for n in range(4, nF):
                        cols.append(f'{index}_{n}')
                else:
                    for n in range(nF):
                        cols.append(f'{index}_{n}')
    selected_test_np = selected_test_df.loc[:, cols].to_numpy()
    selected_inference_np = snp_df.loc[:, cols].to_numpy()
    max = np.max(selected_test_np)
    min = np.min(selected_test_np)
    total_range = max - min
    total_mean = np.mean(np.abs(selected_test_np - selected_inference_np) / total_range, axis=-1)
    total_max = np.max(np.abs(selected_test_np - selected_inference_np) / total_range, axis=-1)


    for i in range(port):
        for j in range(port):
            for s in range(2):
                index = f'{SRI[s]}({i+1},{j+1})'
                cols = []
                if ignore_f0:
                    for n in range(1, nF):
                        cols.append(f'{index}_{n}')
                else:
                    for n in range(nF):
                        cols.append(f'{index}_{n}')
                selected_test_np = selected_test_df.loc[:, cols].to_numpy()
                selected_inference_np = snp_df.loc[:, cols].to_numpy()
                
                test_max[i, j, s] = np.max(selected_test_np)
                test_min[i, j, s] = np.min(selected_test_np)
                test_range = test_max[i, j, s] - test_min[i, j, s]            

                error = selected_test_np - selected_inference_np
                abs_error = np.abs(error)                
                # mean error
                error_analysis_np[:, i * 16 + j * 4 + s * 2] = np.mean(abs_error / test_range, axis=-1)
                error_analysis_header.append(f'{index}_mean')
                # max error
                error_analysis_np[:, i * 16 + j * 4 + s * 2 + 1] = np.max(abs_error / test_range, axis=-1)
                error_analysis_header.append(f'{index}_max')

    # Calcluate distance between test and train

    input_features = InputFeatures(os.path.join(os.getcwd(), configs.config.dir))
    input_cols, variables_range, decimals = input_features.get_variable_range()

    test_parameters = selected_test_df.loc[:, input_cols]
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
    
    closest_neighbour = []
    min_distances = []
    for index, row in selected_test_df.iterrows():
        # calculate the distance
        delta = test_parameters.loc[index, input_cols].to_numpy().reshape(1,-1) - train_parameters_np
        distance = np.sqrt(np.sum(delta ** 2, axis = -1))
        closest_neighbour.append(train_df.index[np.argmin(distance)])
        min_distances.append(np.min(distance)) 

    error_analysis_np[:, -4] = total_mean
    error_analysis_np[:, -3] = total_max
    error_analysis_np[:, -2] = np.array(min_distances)

    error_analysis_header.append(f'total_mean')
    error_analysis_header.append(f'total_max')
    error_analysis_header.append(f'min_distance')
    error_analysis_header.append(f'closest_neighbour')
    

    error_analysis_df = pd.DataFrame(error_analysis_np, index=indices, columns=error_analysis_header)
    error_analysis_df['closest_neighbour'] = closest_neighbour

    csv_path = os.path.join('../Data/Out', case + '_2.csv')
    error_analysis_df.to_csv(csv_path)

    rlgc_out_dir = os.path.join('../Data/Out', configs.case+'_2')
    if not os.path.exists(rlgc_out_dir):
        os.makedirs(rlgc_out_dir) 
    
    # Sort the DataFrame by the 'Age' column in ascending order
    sorted_df = error_analysis_df.sort_values(by='total_mean', ascending=True)
    
    if ignore_f0:
        nF -= 1
        f = f[1:]

    
    count = 0
    for index, row in sorted_df.iterrows():

        count += 1
        if count >= 20:
            continue
        fig, ax = plt.subplots(port, 2, figsize=(16, 20), constrained_layout=True)
        for j in range(port):
            for s in range(2):
                index_in = f'{SRI[s]}({1},{j+1})'
                cols = []
                for n in range(nF):
                    cols.append(f'{index_in}_{n}')
                selected_test_np = selected_test_df.loc[index, cols].to_numpy().reshape(-1)
                selected_inference_np = snp_df.loc[index, cols].to_numpy().reshape(-1)
                
                ax[j, s].plot(f, selected_test_np, label=f'Truth of {index_in}')
                ax[j, s].plot(f, selected_inference_np, label=f'Prediction of {index_in}')

                ax[j, s].legend(loc='upper right')
                ax[j, s].set_xlabel('Frequency (Hz)')
                ax[j, s].set_ylabel('S-parameter (dB)')
                bottom = test_min[0, j, s]
                top = test_max[0, j, s]
                
                if port == 3:
                    ax[j, s].set_ylim(bottom - 0.1 * (top - bottom), max + 0.1 * (top - bottom))
                

        plt.savefig(f'{pdf_dir}/{index}.pdf')
        plt.savefig(f'{png_dir}/{index}.png')





def main():
    
    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)

    verbose = configs.verbose
    trial = configs.trial
    if configs.distributed:
        # distributed, get SLURM variables
        rank = int(os.environ['SLURM_PROCID'])
        gpu = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        device = f"cuda:{gpu}"
        print(f"rank:{rank} gpu:{gpu} cuda device count:{torch.cuda.device_count()}")
    else:
        # CUDA or CPU
        if torch.cuda.is_available():
            dev = "cuda:0"
            print("CUDA avaiable")
        else:  
            dev = "cpu"
            print("CUDA not avaiable")
        device = dev
        rank = 0
        world_size = 1
    
    # define port list for this process
    port_list = []
    locate_id = 0
    for i in range(configs.node):
        if configs.passivity:
            j_range = range(i, configs.node)
        else:
            j_range = range(configs.node)
        for j in j_range:
            if (locate_id == rank):
                port_list.append((i+1,j+1)) 
            if locate_id == (world_size - 1):
                locate_id = 0
            else:
                locate_id += 1

    if trial:
        port_list = [(1,1)]
    if verbose:
        print(f"Port list: {port_list}")

    case = configs.case
    # model path
    current_dir = os.getcwd()
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))
    model_dir = os.path.join(home_dir, 'Models', case)
    if not os.path.exists(model_dir) and (not configs.distributed or rank == 0):
        os.makedirs(model_dir) 
    configs.model_dir = model_dir
    
    # define fig dir as (pdf_dir, png_dir)
    pdf_dir = os.path.join(configs.fig.dir, 'pdf', case)
    if not os.path.exists(pdf_dir) and (not configs.distributed or rank == 0):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join(configs.fig.dir, 'png', case)
    if not os.path.exists(png_dir) and (not configs.distributed or rank == 0):
        os.makedirs(png_dir) 
    configs.fig_dir = (pdf_dir, png_dir)

    out_dir = os.path.join(home_dir, 'Data/Out/Rescaled', case)
    if not os.path.exists(out_dir) and (not configs.distributed or rank == 0):
        os.makedirs(out_dir)
    configs.out_dir = out_dir

    # Parse input features config
    input_features = InputFeatures(os.path.join(os.getcwd(), configs.config.dir))

    # Define writer
    writer = SummaryWriter(log_dir='../Log', filename_suffix=case)


    for i,j in port_list:
        # define output cols and dataloaders
        output_cols = ['SR(%d,%d)' % (i, j), 'SI(%d,%d)' % (i, j)]
        error_analysis(configs, input_features, output_cols, writer, device)

    if writer is not None:
        writer.close()
    return

if __name__ == "__main__":
    # main()
    analyze_s_parameters_error()