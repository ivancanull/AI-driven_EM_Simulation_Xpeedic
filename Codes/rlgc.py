import os
import argparse

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import *
from cores import *

class LSTM_proj(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.nF = out_channels
        self.hidden_size = mid_channels
        self.proj_size = 1
        self.num_layers = 3
        self.lstm = nn.LSTM(input_size=in_channels+1, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True,
                            proj_size=self.proj_size)

    
    def forward(self, x):
        #
        batch_num = x.shape[0]
        x = x.unsqueeze(1).repeat(1, self.nF, 1)
        f = torch.linspace(0, 1, self.nF).reshape(1, self.nF, 1).repeat(batch_num, 1, 1).to(x.device)
        x = torch.cat([f, x], dim=-1)
        h0 = torch.zeros(self.num_layers, batch_num, self.proj_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_num, self.hidden_size).to(x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return torch.unsqueeze(torch.squeeze(output, dim=-1), dim=1)

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
        return torch.unsqueeze(self.layers(x), 1)

class FCN_deeper(nn.Module):
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
        return torch.unsqueeze(self.layers(x), 1)

class LSTM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.nF = out_channels
        self.hidden_size = mid_channels
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, 1)

    
    def forward(self, x: Tensor) -> Tensor:
        #
        batch_num = x.shape[0]
        x = x.unsqueeze(1).repeat(1, self.nF, 1)
        h0 = torch.zeros(self.num_layers, batch_num, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_num, self.hidden_size).to(x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.linear(output)
        return torch.unsqueeze(torch.squeeze(output, dim=-1), dim=1)
    
class SmoothMSELoss(nn.Module):
    def __init__(self, alpha = 0.1):
        super(SmoothMSELoss, self).__init__()
        self.alpha = alpha
    def forward(self, inputs, targets):
        loss_mse = nn.MSELoss()
        delta = inputs[:, :, 0:-1] - inputs[:, :, 1:]
        derivative = delta[:, :, 0:-1] - delta[:, :, 1:]
        loss_1 = loss_mse(inputs, targets)
        loss_2 = self.alpha * torch.sqrt(torch.mean(torch.abs(derivative) ** 2))
        loss_3 = self.alpha * torch.sqrt(torch.mean(torch.abs(delta) ** 2))
        loss = loss_1 + loss_2 + loss_3
        return loss

def isepoch(
    epoch,
    per,
):
    return epoch % per == (per - 1)

def build_net(output_col, variables, f, device):
    # predict_dc
    if output_col == "R(1,1)":
        net = LSTM(
            len(variables),
            64,
            len(f),
        ).to(torch.device(device))
    elif output_col == "R(2,1)":
        net = LSTM_proj(
            len(variables),
            64,
            len(f),
        ).to(torch.device(device))
    elif "G" in output_col:
        net = FCN(
            len(variables),
            32,
            1,
        ).to(torch.device(device))
    elif output_col == "L(1,1)":
        net = LSTM(
            len(variables),
            64,
            len(f) - 1,
        ).to(torch.device(device))
    elif output_col == "L(2,1)":
        net = LSTM_proj(
            len(variables),
            64,
            len(f) - 1,
        ).to(torch.device(device))
    elif 'C' in output_col:
        net = FCN(
            len(variables),
            32,
            1,
        ).to(torch.device(device))
    else:
        raise NotImplementedError
    return net

def build_criterion(mode):
    # predict_dc
    if mode == "R" or mode == "G" or mode == 'C':
        criterion = nn.MSELoss()
    elif mode == "L":
        criterion = SmoothMSELoss(1)
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError
    return criterion

def train(
    configs,
    train_dataset,
    mode: str,
    net,
    output_cols: str,
    variables,
    device,
    f,
    writer: SummaryWriter,
):
    
    # set value and mode
    train_dataset.mode = mode
    
    # Predict DC value
    ckpt_path = os.path.join(configs.model_dir, f'RLGC_{output_cols}_ckpt.pt')
    best_path = os.path.join(configs.model_dir, f'RLGC_{output_cols}_best.pt')

    

    # define loss function
    criterion = build_criterion(mode)

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
        
        # Plot the figure
        if configs.fig.plot_during_train and isepoch(epoch, configs.fig.plot_per_epoch):
            # Get train examples
            example_num = configs.fig.example_num
            example_train_ids = np.concatenate(
                (np.random.permutation(train_ids)[0:example_num], np.random.permutation(val_ids)[0:example_num]))
            X_example, y_example = train_dataset[example_train_ids]
            output = net(X_example)

            # Denormalize
            y_example = train_dataset.denormalize_output(y_example)
            output = train_dataset.denormalize_output(output)
            
            # Plot and save figure
            fig = plot_RLGC(f, y_example, output, epoch+1, output_cols, configs)
            if writer is not None:
                writer.add_figure(f'{output_cols}_Pred', fig)

    torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, best_path)
    
    # test
    return net

def test(
    configs,
    test_dataset,
    mode: str,
    net,
    output_cols: str,
    variables,
    device,
    f,
) -> torch.Tensor:
    
    # set value and mode
    test_dataset.mode = mode
    
    # Predict DC value
    best_path = os.path.join(configs.model_dir, f'RLGC_{output_cols}_best.pt')

    X_train = test_dataset.parameters

    # Load best model
    if os.path.exists(best_path):
        ckpt = torch.load(best_path)
        net.load_state_dict(ckpt['model_state_dict'])
    
    net.eval()

    output = net(X_train)
    output = torch.squeeze(test_dataset.denormalize_output(output), dim=1)
    
    if mode == "R":
        pass
    elif mode == "L":
        output = torch.cat([output[:, 1:2], output], dim=1)
    elif mode == "C":
        output = output.repeat(1, len(f))
    elif mode == "G":
        # Create the linespace
        linespace = torch.linspace(0, 1, len(f)).view(1, -1).to(output.device)  # Creating a linespace from 0 to 1 with C points
        # Stack the original tensor along a new axis to create the linespace
        output = torch.squeeze(output.unsqueeze(2) * linespace, -2)
    else:
        raise NotImplementedError

    # test
    return output

def analyze_rlgc(
    configs: Config,
    input_features: InputFeatures,
    output_cols,
    writer: SummaryWriter,
    device,
    df,
    inference_df
):
    """
    Analyze the rlgc 
    """
        
    for output_col in output_cols:
        mode = output_col[0:1]

        variables, variables_range, decimals = input_features.get_variable_range()
        f = parse_frequency(input_features.frequency)
        nF = len(f)

        train_dataset, test_dataset = make_rlgc_datasets(
            configs=configs,
            input_cols=variables,
            output_cols=[output_col],
            nF=nF,
            mode=mode,
            device=torch.device(device),
        )

        # Analyze RLGC Causal RLGC(f) Models for Transmission Lines From Measured S-Parameters
        # example_y = test_dataset.output[0: 10]
        
        # fig, ax = plt.subplots(constrained_layout=True)
        # if mode == 'R':
        #     example_y = example_y - example_y[:, :, 0:1]
        #     f_torch = torch.Tensor(f).to(example_y.device)
        #     Rs = example_y / torch.sqrt(f_torch)
            
        #     for i in range(Rs.shape[0]):
        #         ax.plot(f, Rs[i].cpu().numpy().reshape(-1))
        # elif mode == 'L':
        #     example_y = example_y - example_y[:, :, -2:-1]
        #     f_torch = torch.Tensor(f).to(example_y.device)[1:]
        #     Rs = example_y * 2 * 3.14 * torch.sqrt(f_torch)
        #     for i in range(Rs.shape[0]):
        #         ax.plot(f[1:], Rs[i].cpu().numpy().reshape(-1))
        # else:
        #     pass
        # plt.savefig(f'{output_col}_model.png')

        # Define max and min value for output normalization
        max_tensor, min_tensor = train_dataset.get_max_min()
        train_dataset.normalize_output()        
        test_dataset.set_max_min(max_tensor, min_tensor)
        test_dataset.normalize_output()


        # build network
        net = build_net(output_col, variables, f, device)

        # net = train(configs, train_dataset, mode, net, output_col, variables, device, f, writer)
        output = test(configs, test_dataset, mode, net, output_col, variables, device, f)
        cols = []
        # expand output cols
        for i in range(nF):
            cols.append(output_col+'_%d'%i)

        
        inference_df.loc[:, cols] = output.cpu().detach().numpy()

        cols = []
        if output_col[-5:] == '(1,1)':
            for i in range(nF):
                cols.append(output_col[0]+'(2,2)'+'_%d'%i)
            inference_df.loc[:, cols] = output.cpu().detach().numpy()
        elif output_col[-5:] == '(2,1)':
            for i in range(nF):
                cols.append(output_col[0]+'(1,2)'+'_%d'%i)
            inference_df.loc[:, cols] = output.cpu().detach().numpy()
        else:
            raise ValueError
    return inference_df
    
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
    
    port_list = [[1,1], [1,2], [2,1], [2,2]]

    # get columns
    df_cols = ['F']
    for k in 'R', 'L', 'C', 'G':
        for i,j in port_list:
            df_cols.append(f'{k}({i},{j})')
    df = pd.DataFrame(columns=df_cols, index=range(1, 502))

    port_list = [[1,1], [2,1]]
    
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

    # read data
    test_dfs = []
    indices = {}
    indices['test_idx'] = []

    working_dir = os.getcwd()
    config_file = os.path.join(os.getcwd(), configs.config.dir)
    for i, ds in enumerate(configs.datasets.name):

        test_df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/%s/%s_test_rlgc_concat.zip' %(ds, ds)), compression='zip')
        test_df.index = test_df.index + '_%s' % ds
        test_dfs.append(test_df)

    test_df = pd.concat(test_dfs)
    inference_df = test_df.copy()

    for i,j in port_list:
        # define output cols and dataloaders
        output_cols = ['R(%d,%d)' % (i, j), 'L(%d,%d)' % (i, j), 'C(%d,%d)' % (i, j), 'G(%d,%d)' % (i, j)]
        # output_cols = ['R(%d,%d)' % (i, j)]
        # output_cols = ['G(%d,%d)' % (i, j)]
        inference_df = analyze_rlgc(configs, input_features, output_cols, writer, device, df, inference_df)
    
    inference_df.to_pickle(f'../Data/Out/{configs.case}_2_Inference_RLGC.zip', compression='zip')

    # if writer is not None:
    #     writer.close()
    return

if __name__ == "__main__":
    main()
