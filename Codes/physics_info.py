import os
import argparse

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


from utils import *
from cores import *

def isepoch(
    epoch,
    per,
):
    return epoch % per == (per - 1)

class FCN(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)

class FCN_5(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)

def train(
    configs,
    train_dataset,
    mode: str,
    output_cols: str,
    variables,
    device,
    nF,
    criterion = nn.MSELoss,
    pulse_duration = 0,
):
    
    # set value and mode
    train_dataset.mode = mode
    
    # Predict DC value
    ckpt_path = os.path.join(configs.model_dir, f'{output_cols}_{train_dataset.mode}_ckpt.pt')
    best_path = os.path.join(configs.model_dir, f'{output_cols}_{train_dataset.mode}_best.pt')

    # predict_dc
    if mode == "dc":
        net = FCN(len(variables), 512, 2).to(torch.device(device))
    elif mode == "fft_0":
        net = FCN(len(variables), 512, 1).to(torch.device(device))
    elif mode == "fft_rest":
        net = unet_fourier(len(variables), configs.model.channels, 2*(nF-1)-1, configs.model.kernels).to(torch.device(device))
    elif mode == "fft":
        net = unet_fourier(len(variables), configs.model.channels, 2*(nF-1), configs.model.kernels).to(torch.device(device))
    elif mode == "pulse_location":
        net = FCN(len(variables), 256, 1).to(torch.device(device))
    elif mode == "pulse_peak":
        net = FCN(len(variables), 1024, 15).to(torch.device(device))
    elif mode == "pulse":
        net = unet_pulse(len(variables), [128, 64, 64, 128, 256, 512], pulse_duration).to(torch.device(device))
    elif mode == "pulse_peak_max" or mode == "pulse_peak_a":
        net = FCN(len(variables), 256, 1).to(torch.device(device))
    else:
        raise NotImplementedError


    optimizer = make_optimizer(
        net.parameters(),
        configs=configs,
    )
    initialize_weights(net)

    # Define loss and dataset
    train_losslogger = LossLogger()
    val_losslogger = LossLogger()
    batch_size = configs.batch_size
    train_ids, val_ids = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size, sampler=train_subsampler)
    val_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size, sampler=val_subsampler)
    best_loss = 100.0
    epoch_continue = 0
    # Continue training
    if configs.ckpt.read:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            epoch_continue = ckpt['epoch_continue']
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
    for epoch in range(epoch_continue, configs.num_epochs):
        
        train_losslogger.clean()
        net.train()

        for i, (X_train, y_train) in enumerate(train_dataloader):

            output = net(X_train)

            # Set the loss functions
            train_loss = criterion(output, y_train)
            
            # Clear the gradients
            optimizer.zero_grad()
            
            # Backward propagation
            train_loss.backward()
            
            # Update parameters
            optimizer.step()

            # Calculate the loss function
            train_losslogger.update(train_loss.item(), X_train.shape[0])
            
        # Validation
        if isepoch(epoch, configs.validation.valid_per_epoch):

            torch.save({
                'epoch_continue': epoch+1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, ckpt_path)

            val_losslogger.clean()
            net.eval()

            for i, (X_val, y_val) in enumerate(val_dataloader, 0):

                output = net(X_val)

                # Set the loss functions
                val_loss = criterion(output, y_val)
                
                # Calculate the loss function
                val_losslogger.update(val_loss.item(), X_val.shape[0])
                # val_losslogger.update_error(y_val, output, X_val.shape[0])

            if val_losslogger.loss < best_loss:
                best_loss = val_losslogger.loss
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, best_path)

            print('Epoch: %10d, Train Loss: %6f, Val Loss: %6f'
                    % (epoch+1, train_losslogger.loss, val_losslogger.loss))
    
    # test
    return net

def test(
    configs,
    test_dataset,
    net: nn.Module,
    mode: str,
    output_cols: str,
):
    criterion = nn.MSELoss()

    batch_size = configs.batch_size
    # set value and mode
    test_dataset.mode = mode
    
    # continue training
    best_path = os.path.join(configs.model_dir, f'{output_cols}_{test_dataset.mode}_best.pt')

    # Define data loaders for training and testing data in this fold
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size)
    
    if not os.path.exists(best_path):
        raise ValueError(f'Best model {best_path} not exists!')    
    best = torch.load(best_path)
    net.load_state_dict(best['model_state_dict'])

    test_losslogger = PhysicsLossLogger(mode)


    for i, (X_test, y_test) in enumerate(test_dataloader):

        output = net(X_test)

        # Set the loss functions
        test_loss = criterion(output, y_test)
        
        # Calculate the loss function
        test_losslogger.update(test_loss.item(), X_test.shape[0])
        test_losslogger.update_error(y_test, output, X_test.shape[0])
    
    if mode == 'dc':
        srmean, simean, srmax, simax = test_losslogger.error
        print(f'SR mean absolute error: {srmean}, SI mean absolute error: {simean}')
        print(f'SR max absolute error: {srmax}, SI max absolute error: {simax}')
    elif mode == 'fft_0':
        mean, max = test_losslogger.error
        print(f'Mean absolute error: {mean}, Max absolute error: {max}')
    elif mode == 'fft_rest':
        mean, max = test_losslogger.error
        print(f'Mean absolute error: {mean}, Max absolute error: {max}')
    elif mode == 'fft':
        mean, max = test_losslogger.error
        print(f'Mean absolute error: {mean}, Max absolute error: {max}')
    elif mode == 'pulse_location':
        mean, max = test_losslogger.error
        print(f'Mean absolute error: {int(mean * 1000)}, Max absolute error: {int(max * 1000)}')
    elif 'pulse_peak' in mode:
        mean, max = test_losslogger.error
        print(f'Mean absolute error: {mean}, Max absolute error: {max}')
    else:
        raise NotImplementedError
    return net

def analyze(
    configs: Config,
    input_features: InputFeatures,
    output_cols,
    writer: SummaryWriter,
    device
):
    
    variables, variables_range, decimals = input_features.get_variable_range()
    f = parse_frequency(input_features.frequency)
    nF = len(f)

    train_dataset, test_dataset = make_physics_informed_datasets(
        configs=configs,
        input_cols=variables,
        output_cols=output_cols,
        nF=nF,
        device=torch.device(device),
    )

    train_dataset.get_DC_value()
    train_dataset.get_ifft()
    test_dc = test_dataset.get_DC_value()


    fft_0_net = train(configs, train_dataset, "fft_0", output_cols, variables, device, nF, nn.HuberLoss(delta=0.1))
    fft_0_net.eval()
    fft_0_net = test(configs, train_dataset, fft_0_net, "fft_0", output_cols)

    fft_rest_net = train(configs, train_dataset, "fft_rest", output_cols, variables, device, nF, nn.MSELoss())
    fft_rest_net.eval()
    fft_rest_net = test(configs, train_dataset, fft_rest_net, "fft_rest", output_cols)

    num = 5
    example_parameters = test_dataset.parameters[0:num]
    example_signal = test_dataset.output[0:num]

    predict_dc = None
    signal_ifft = get_ifft(example_signal)
    predict_ifft = signal_ifft.copy()
    envelop = get_envelop(example_signal)

    predict_ifft[:, 0:1] = fft_0_net(example_parameters).detach().cpu().numpy()
    predict_ifft[:, 1:] = fft_rest_net(example_parameters).detach().cpu().numpy() / 15
    predict_signal = get_fft(predict_ifft)

    plot_X_y_physics(
        f, y=test_dataset.output[0:num], pred=predict_signal, configs=configs, output_cols_name=output_cols, 
        dc=test_dc[0:num], predict_dc=predict_dc, envelop=envelop, ifft=[signal_ifft, predict_ifft], prefix='physics')

def analyze_fft(
    configs: Config,
    input_features: InputFeatures,
    output_cols,
    writer: SummaryWriter,
    device
):
    """
    Analyze the ifft 
    """
    variables, variables_range, decimals = input_features.get_variable_range()
    f = parse_frequency(input_features.frequency)
    nF = len(f)

    train_dataset, test_dataset = make_physics_informed_datasets(
        configs=configs,
        input_cols=variables,
        output_cols=output_cols,
        nF=nF,
        device=torch.device(device),
    )

    train_dataset.get_DC_value()
    train_dataset.get_ifft()
    train_dataset.get_ifft_pulse_approaximation()
    train_dataset.get_ifft_pulse_curve_fitting()
    # pulse_start, pulse_end, train_ifft_pulse = train_dataset.get_ifft_pulse()
    # pulse_duration = pulse_end - pulse_start
    
    test_dc = test_dataset.get_DC_value()
    test_dataset.get_ifft()
    test_dataset.get_ifft_pulse_approaximation()
    train_dataset.get_ifft_pulse_curve_fitting()

    # test_ifft_pulse = test_dataset.get_ifft_pulse_location()
    # test_dataset.get_ifft_pulse(pulse_start, pulse_end)
    ifft_length = (nF-1)*2

    fft_0_net = train(configs, train_dataset, "fft_0", output_cols, variables, device, nF, nn.HuberLoss(delta=0.1))
    fft_0_net.eval()
    fft_0_net = test(configs, train_dataset, fft_0_net, "fft_0", output_cols)

    # train location
    pulse_location_net = train(configs, train_dataset, "pulse_location", output_cols, variables, device, nF, nn.MSELoss())
    pulse_location_net.eval()
    pulse_location_net = test(configs, train_dataset, pulse_location_net, "pulse_location", output_cols)

    # train peak
    # pulse_peak_net = train(configs, train_dataset, "pulse_peak", output_cols, variables, device, nF, nn.MSELoss())
    # pulse_peak_net.eval()
    # pulse_peak_net = test(configs, train_dataset, pulse_peak_net, "pulse_peak", output_cols)

    # train peak max
    pulse_peak_max_net = train(configs, train_dataset, "pulse_peak_max", output_cols, variables, device, nF, nn.MSELoss())
    pulse_peak_max_net.eval()
    pulse_peak_max_net = test(configs, train_dataset, pulse_peak_max_net, "pulse_peak_max", output_cols)

    # train peak a
    pulse_peak_a_net = train(configs, train_dataset, "pulse_peak_a", output_cols, variables, device, nF, nn.MSELoss())
    pulse_peak_a_net.eval()
    pulse_peak_a_net = test(configs, train_dataset, pulse_peak_a_net, "pulse_peak_a", output_cols)

    # # train pulse
    # pulse_net = train(configs, train_dataset, "pulse", output_cols, variables, device, nF, nn.MSELoss(), pulse_duration)
    # pulse_net.eval()
    # pulse_net = test(configs, train_dataset, pulse_peak_net, "pulse", output_cols)
    

    num = 15

    example_parameters = test_dataset.parameters
    example_signal = test_dataset.output

    predict_fft_0 = fft_0_net(example_parameters).detach().cpu().numpy()
    predict_pulse_location_center = pulse_location_net(example_parameters).detach().cpu().numpy() * ifft_length
    predict_pulse_peak_max = pulse_peak_max_net(example_parameters).detach().cpu().numpy()
    predict_pulse_peak_a = pulse_peak_a_net(example_parameters).detach().cpu().numpy()
    
    x = np.arange(-7, 8)
    predict_pulse = []
    def normal_distribution_func(x, a):
        y = []
        for i in range(a.shape[0]):
            y.append(np.exp(-x ** 2 / max(float(a[i]), 0.001)))    
        return np.vstack(y)
    
    for i in range(predict_pulse_peak_max.shape[0]):
        y = normal_distribution_func(x, predict_pulse_peak_a[i]) * predict_pulse_peak_max[i]
        predict_pulse.append(y)
    predict_pulse = np.vstack(predict_pulse)
    
    predict_dc = None
    signal_ifft = get_ifft(example_signal)
    predict_ifft = np.zeros_like(signal_ifft)
    envelop = get_envelop(example_signal)
    predict_pulse_location = np.arange(-7, 8, dtype=int) + predict_pulse_location_center.astype(int)
    
    predict_ifft[:, 0:1] = predict_fft_0
    for i in range(predict_pulse_location.shape[0]):
        predict_ifft[i, predict_pulse_location[i]] = predict_pulse[i]
    
    # print the example ifft
    threadshold = 3e-4
    predict_ifft[np.abs(predict_ifft) < threadshold] = 0

    predict_signal = get_fft(predict_ifft)

    # calculate loss
    
    # real value
    true_signal = example_signal.detach().cpu().numpy()
    absolute_error = np.abs(np.real(predict_signal) - true_signal[:, 0, :])
    true_range = np.max(true_signal[:, 0, :], axis=1, keepdims=True) - np.min(true_signal[:, 0, :], axis=1, keepdims=True)
    relative_error = absolute_error / true_range
    print(f"Real RAE: {np.mean(relative_error)}, MRE: {np.max(relative_error)}")
    
    # imag value
    absolute_error = np.abs(np.imag(predict_signal) - true_signal[:, 1, :])
    true_range = np.max(true_signal[:, 1, :], axis=1, keepdims=True) - np.min(true_signal[:, 1, :], axis=1, keepdims=True)
    relative_error = absolute_error / true_range
    print(f"Imag RAE: {np.mean(relative_error)}, MRE: {np.max(relative_error)}")
    
    plot_X_y_physics(
        f, y=test_dataset.output[0:num], pred=predict_signal[0:num], configs=configs, output_cols_name=output_cols, 
        dc=test_dc[0:num], predict_dc=predict_dc, envelop=envelop[0:num], ifft=[signal_ifft[0:num], predict_ifft[0:num]], predict_pulse_location=predict_pulse_location_center[0:num],
        predict_pulse_peak=None, prefix='physics_test')

def analyze_hilbert(
    configs: Config,
    input_features: InputFeatures,
    output_cols,
    writer: SummaryWriter,
    device
):
    variables, variables_range, decimals = input_features.get_variable_range()
    f = parse_frequency(input_features.frequency)
    nF = len(f)

    train_dataset, test_dataset = make_physics_informed_datasets(
        configs=configs,
        input_cols=variables,
        output_cols=output_cols,
        nF=nF,
        device=torch.device(device),
    )

    train_dataset.get_DC_value()
    train_dataset.get_ifft()
    train_dataset.get_ifft_pulse_approaximation()
    train_dataset.get_ifft_pulse_curve_fitting()
    # pulse_start, pulse_end, train_ifft_pulse = train_dataset.get_ifft_pulse()
    # pulse_duration = pulse_end - pulse_start
    
    num = 15

    example_parameters = test_dataset.parameters
    example_signal = test_dataset.output
    
    predict_dc = None
    signal_ifft = get_ifft(example_signal)
    predict_ifft = np.zeros_like(signal_ifft)
    envelop = get_envelop(example_signal)
    
    hilbert = get_hilbert(example_signal)

    print(hilbert.shape)

    plot_X_y_hilbert(
        f, y=test_dataset.output[0:num], pred=None, hilbert=hilbert[0:num], configs=configs, output_cols_name=output_cols, 
        prefix='physics_hilbert')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    return parser.parse_args()

def main():
    
    # parse arguments
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)

    verbose = configs.verbose
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
    
    port_list = [[1,1], [1,2], [1,3], [1,4]]
    port_list = [[1,1]]
    
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

    # Parse input features config
    input_features = InputFeatures(os.path.join(os.getcwd(), configs.config.dir))

    # Define writer
    writer = SummaryWriter(log_dir='../Log', filename_suffix=case)


    for i,j in port_list:
        # define output cols and dataloaders
        output_cols = ['SR(%d,%d)' % (i, j), 'SI(%d,%d)' % (i, j)]
        analyze_fft(configs, input_features, output_cols, writer, device)

    if writer is not None:
        writer.close()
    return

if __name__ == "__main__":
    main()
    