import os
import argparse

import torch

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

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

    range = torch.max(y, dim=(-1), keepdim=True)[0] - torch.min(y, dim=(-1), keepdim=True)[0]
    relative_error = torch.mean(torch.abs(absolute_error) / range, dim=(1, 2),)

    # Worst cases
    mae_error, mae_idx = torch.sort(relative_error, dim=0, descending=True)
    fig = plot_X_y_test(f, y[mae_idx[0:10]], pred[mae_idx[0:10]], output_cols, configs, prefix='test_worst')
    
    # Best cases
    mae_error, mae_idx = torch.sort(relative_error, dim=0, descending=False)
    fig = plot_X_y_test(f, y[mae_idx[0:10]], pred[mae_idx[0:10]], output_cols, configs, prefix='test_best')

    # Calculate error and generate new samples
    mse_error_mean = torch.mean(mse_error, dim=-2, keepdim=True).cpu()
    generate_samples(X, input_features, mse_error_mean, sample_path, configs)
    
    return

def _test_epoch(
    configs: Config,
    net: torch.nn.Module,
    input_features: InputFeatures,
    criterion,
    output_col: str,
    test_dataloader: torch.utils.data.DataLoader,
    test_losslogger: LossLogger,
    device: torch.device,
):
    test_losslogger.clean()
    net.eval()

    # get the top five worst cases of the test set
    worst_cases = [torch.Tensor().to(device) for i in range(4)]
    worst_cases_idx = torch.Tensor().to(device) 
    best_cases = [torch.Tensor().to(device) for i in range(4)]
    best_cases_idx = torch.Tensor().to(device) 

    for i, (X_test, y_test) in enumerate(test_dataloader, 0):

        output = net(X_test)
        # Set the loss functions
        test_loss = criterion(output, y_test)
        # Calculate the loss function
        test_losslogger.update_error(test_loss, y_test, output, X_test.shape[0])
        
        # find the maximum relative loss
        max_error, _ = torch.max(torch.abs(y_test - output), dim=-1)

        # find the top five worst cases and best cases
        descending = torch.topk(max_error, 5)
        ascending = torch.topk(max_error, 5, largest=False)

        worst_cases[0] = torch.cat((worst_cases[0], X_test[descending.indices]), dim=0)
        worst_cases[1] = torch.cat((worst_cases[1], y_test[descending.indices]), dim=0)
        worst_cases[2] = torch.cat((worst_cases[2], output[descending.indices]), dim=0)
        worst_cases[3] = torch.cat((worst_cases[3], max_error[descending.indices].reshape(-1, 1)), dim=0)

        best_cases[0] = torch.cat((best_cases[0], X_test[ascending.indices]), dim=0)
        best_cases[1] = torch.cat((best_cases[1], y_test[ascending.indices]), dim=0)
        best_cases[2] = torch.cat((best_cases[2], output[ascending.indices]), dim=0)
        best_cases[3] = torch.cat((best_cases[3], max_error[ascending.indices].reshape(-1, 1)), dim=0)

    worst_cases_total = torch.topk(worst_cases[3], 5, dim=0)
    best_cases_total = torch.topk(best_cases[3], 5, dim=0, largest=False)
    worst_cases_total_idx = worst_cases_total.indices
    best_cases_total_idx = best_cases_total.indices

    # plot max error
    plot_X_y_test(
        input_features.frequency_np,
        worst_cases[1][worst_cases_total_idx].squeeze(1),
        worst_cases[2][worst_cases_total_idx].squeeze(1),
        output_col,
        configs,
        prefix='test_worst'
    )

    plot_X_y_test(
        input_features.frequency_np,
        best_cases[1][best_cases_total_idx].squeeze(1),
        best_cases[2][best_cases_total_idx].squeeze(1),
        output_col,
        configs,
        prefix='test_best'
    )

    print(f'{output_col} Test Loss: {test_losslogger.loss:.2e}, Test Mean ER: {test_losslogger.mean_error * 100:.4f} %, Test Max ER: {test_losslogger.max_error * 100:.4f} %')

    return

def plot_3d_contour(
    configs: Config,
    input_features,
    output_col: str,
    net,
    device,
    test_dataset = None
):

    if not isinstance(input_features, MultiLayerInputFeatures):
        raise ValueError("input_features must be MultiLayerInputFeatures!")

    
    
    if input_features.sampled_variables != ["W", "S"]:
        return
    elif test_dataset is None:
        # sample
        samples_np, X, Y, wn, sn = make_samples(input_features=input_features,
                                method='test_sweep',
                                step_ratio=0.1,)
        stackups = create_stackups_from_samples(input_features,
                                                samples_np,
                                                layer_num=input_features.layer_num)

        parameters_tensor = torch.zeros([len(stackups), len(input_features.sampled_variables)])
        for i in range(parameters_tensor.shape[0]):
            for v_i, v in enumerate(input_features.sampled_variables):
                if v == "W":
                    parameters_tensor[i, v_i] = stackups[i].layers[3].w_list[0]
                elif v == "S":
                    parameters_tensor[i, v_i] = stackups[i].layers[3].spaces[0]
                else:
                    raise ValueError("Error")
                
        parameters_tensor = parameters_tensor.to(device)

        # define min and max for normalization
        config_tensor_min = torch.zeros([1, len(input_features.sampled_variables)])
        config_tensor_max = torch.zeros([1, len(input_features.sampled_variables)])
        # Normalize method: Min-Max
        for i, v in enumerate(input_features.sampled_variables):
            config_tensor_min[0, i] = getattr(input_features, v).min
            config_tensor_max[0, i] = getattr(input_features, v).max

        config_tensor_min = torch.Tensor(config_tensor_min).to(device)
        config_tensor_max = torch.Tensor(config_tensor_max).to(device)

        parameters_tensor = (parameters_tensor - config_tensor_min) / (config_tensor_max - config_tensor_min)
        net.eval()
        output = net(parameters_tensor)
        
        fig = plt.figure(figsize=(20, 5), constrained_layout=True)

        for i in range(1):
            for j in range(5):
                Z = np.zeros([wn, sn])
                for wi in range(wn):
                    for si in range(sn):
                        w = stackups[wi * sn].layers[3].w_list[0]
                        s = stackups[wi * sn + si].layers[3].spaces[0]
                        index = f'{configs.dataset_generation.name}_0_{wi * 10 + si + 1}'
                        y = output[wi * sn + si, input_features.nF // 5 * j].cpu().detach().numpy()
                        Z[wi, si] = y

                # set up a figure twice as wide as it is tall
                ax = fig.add_subplot(1, 5, j + 1, projection='3d')

                ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='royalblue')
                # ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
                ax.set_title(f'{output_col} at frequency {j / 5 * 100} GHz')
                ax.set(xlabel='W', ylabel='S', zlabel='S-parameter')

        plt.savefig(f'{configs.fig_dir[0]}/test_evaluate_ws_{output_col}.pdf')
        plt.savefig(f'{configs.fig_dir[1]}/test_evaluate_ws_{output_col}.png')


    else:
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        output = net(test_dataset.parameters).cpu().detach().numpy()
        X = test_dataset.parameters[:, 0].cpu().detach().numpy()
        Y = test_dataset.parameters[:, 1].cpu().detach().numpy()
        for i in range(1):
            for j in range(5):
                 # set up a figure twice as wide as it is tall
                ax = fig.add_subplot(2, 5, i * 5 + j + 1, projection='3d')

                ax.plot_surface(X, Y, output[:, (i * 5 + j) * input_features.nF // 10], cmap=cm.coolwarm, edgecolor='royalblue')
                # ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
                ax.set_title(f'{output_col} at frequency {j / 5 * 100} GHz')
                ax.set(xlabel='W', ylabel='S', zlabel='S-parameter')
        plt.savefig(f'{configs.fig_dir[0]}/test_evaluate_ws_{output_col}.pdf')
        plt.savefig(f'{configs.fig_dir[1]}/test_evaluate_ws_{output_col}.png')

def test(
    configs: Config,
    input_features,
    output_col: str,
    device,
    load_model_col: str = None,
):
    # Get parameters
    batch_size = configs.batch_size
    if load_model_col is None:
        load_model_col = output_col
    if hasattr(configs.model, 'read'):
        load_model = configs.model.read
    else:
        load_model = configs.trial

    best_path = os.path.join(configs.model_dir, f'{load_model}_{load_model_col}_best.pt')
    output_log_name = f'{configs.trial}_{output_col}'

    train_dataset, test_dataset = make_datasets(
        configs=configs,
        input_features=input_features,
        output_col=output_col,
        device=torch.device(device),
    )

    # Define max and min value for output normalization
    max_tensor, min_tensor = train_dataset.get_max_min()
    tensor_range = (max_tensor - min_tensor).item()
    
    if test_dataset is None:
        train_dataset.set_max_min(max_tensor, min_tensor)
        train_dataset.normalize_output()
        test_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size)
    else:
        train_dataset.set_max_min(max_tensor, min_tensor)
        train_dataset.normalize_output()
        test_dataset.set_max_min(max_tensor, min_tensor)
        test_dataset.normalize_output()
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size)

    net = make_model(
        configs=configs,
        in_channels=len(input_features.sampled_variables) if configs.multilayers else input_features.variable_num,
        out_channels=input_features.nF,
        device=device
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

    
        
    # Run the training loop for defined number of epochs
    test_losslogger = LossLogger()
    # Load best model
        
    if not os.path.exists(best_path):
        raise ValueError(f"Model not trained {best_path}!")
    
    best = torch.load(best_path)
    # split = best['split']
    # fold_continue = best['fold']
    # sub_epoch_continue = best['sub_epoch']
    net.load_state_dict(best['model_state_dict'])
    optimizer.load_state_dict(best['optimizer_state_dict'])
    scheduler.load_state_dict(best['scheduler_state_dict'])

    _test_epoch(
        configs=configs,
        net=net, 
        input_features=input_features,
        criterion=criterion,
        output_col=output_log_name, 
        test_dataloader=test_dataloader, 
        test_losslogger=test_losslogger,
        device=device
    )
    
    # draw 3d couter plot
    plot_3d_contour(
        configs=configs,
        net=net, 
        input_features=input_features,
        output_col=output_col, 
        device=device,
        test_dataset=test_dataset,
    )
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    return parser.parse_args()

def main():
    
    # parse arguments
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

    port_list = ['(1,1)',]
    
    output_cols = []
    for port in port_list:
        # define output cols and dataloaders
        if configs.model.name.split('_')[-1] == 'srsi': 
            output_cols.append(port)
        else:
            output_cols +=  [f'SR{port}', f'SI{port}']

    for output_col in output_cols:

        test(
            configs=configs,
            input_features=input_features,
            output_col=output_col,
            device=torch.device("cuda:0")
            )

    return

if __name__ == "__main__":
    main()
    