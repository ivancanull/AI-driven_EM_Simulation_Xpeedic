import os
import argparse

import torch
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer

from sklearn.model_selection import train_test_split

from utils import *
from cores import *

def _train_epoch(
    epoch: int,
    net: torch.nn.Module,
    optimizer: Optimizer,
    criterion,
    writer: SummaryWriter,
    output_col: str,
    train_dataloader: torch.utils.data.DataLoader,
):
    net.train()

    total_loss = 0.0
    total_size = 0
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
        total_loss += train_loss
        total_size += X_train.shape[0]

    # Epoch done, write loss
    if writer is not None:
        writer.add_scalar(f'{output_col}_Loss/train', total_loss / total_size, epoch)

    return

def _val_epoch(
    epoch: int,
    net: torch.nn.Module,
    criterion,
    writer: SummaryWriter,
    output_col: str,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    train_losslogger: LossLogger,
    val_losslogger: LossLogger,
    output_log_file_name: str,
    fold: int = None,
):
    train_losslogger.clean()
    val_losslogger.clean()
    net.eval()

    for i, (X_train, y_train) in enumerate(train_dataloader):

        output = net(X_train)
        # Set the loss functions
        train_loss = criterion(output, y_train)
        # Calculate the loss function
        train_losslogger.update_error(train_loss, y_train, output, X_train.shape[0])

    for i, (X_val, y_val) in enumerate(val_dataloader, 0):

        output = net(X_val)
        # Set the loss functions
        val_loss = criterion(output, y_val)
        # Calculate the loss function
        val_losslogger.update_error(val_loss, y_val, output, X_val.shape[0])
    
    # Epoch done, write loss
    if writer is not None:
        writer.add_scalar(f'{output_col}_Loss/loss', val_losslogger.loss, epoch)
    
    if fold is None:
        log_content = f'Epoch: {epoch+1}, Train Loss: {train_losslogger.loss:.2e}, Val Loss: {val_losslogger.loss:.2e}, Train Mean ER: {train_losslogger.mean_error * 100:.4f} %, Train Max ER: {train_losslogger.max_error * 100:.4f} %,  Val Mean ER: {val_losslogger.mean_error * 100:.4f} %, Val Max ER: {val_losslogger.max_error * 100:.4f} %'
        print(log_content)
        with open(output_log_file_name, 'a') as f:
            f.write(log_content)
            f.write('\n')
    else:
        log_content = f'Fold: {fold}, Epoch: {epoch+1}, Train Loss: {train_losslogger.loss:.2e}, Val Loss: {val_losslogger.loss:.2e}, Train Mean ER: {train_losslogger.mean_error * 100:.4f} %, Train Max ER: {train_losslogger.max_error * 100:.4f} %,  Val Mean ER: {val_losslogger.mean_error * 100:.4f} %, Val Max ER: {val_losslogger.max_error * 100:.4f} %'
        print(log_content)
        with open(output_log_file_name, 'a') as f:
            f.write(log_content)
            f.write('\n')
    return

def train(
    configs: Config,
    input_features: InputFeatures,
    output_col,
    writer: SummaryWriter,
    device,
    data_cols: List[str] = None
):

    # Get parameters
    num_epochs = configs.num_epochs
    batch_size = configs.batch_size
    if isinstance(output_col, str):
        multiport = False
    elif isinstance(output_col, list):
        multiport = True
        port_num = len(output_col)
    else:
        raise ValueError("Please specify output_col as str or list.")
    
    if not multiport:
        ckpt_path = os.path.join(configs.model_dir, f'{configs.trial}_{output_col}_ckpt.pt')
        best_path = os.path.join(configs.model_dir, f'{configs.trial}_{output_col}_best.pt')
        output_log_name = f'{configs.trial}_{output_col}'
        output_log_file_name = f'{output_log_name}.txt'
        output_log_file = open(output_log_file_name, 'w')
        output_log_file.close()
    else:
        ckpt_path = os.path.join(configs.model_dir, f'{configs.trial}_{output_col[0]}_multiport_ckpt.pt')
        best_path = os.path.join(configs.model_dir, f'{configs.trial}_{output_col[0]}_multiport_best.pt')
        output_log_name = f'{configs.trial}_{output_col[0]}_multiport'
        output_log_file_name = f'{output_log_name}.txt'
        output_log_file = open(output_log_file_name, 'w')
        output_log_file.close()
    
    train_dataset, test_dataset = make_datasets(
        configs=configs,
        input_features=input_features,
        output_col=output_col,
        device=torch.device(device),
        multiport=multiport,
        data_cols=data_cols
    )
    
    # Define max and min value for output normalization
    max_tensor, min_tensor = train_dataset.get_max_min()
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

    # Define data loaders for training and testing data in this fold
    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=train_subsampler)
    val_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=val_subsampler)

    in_channels = train_dataset.parameters.shape[1]
    net = make_model(
        configs=configs,
        #in_channels=len(input_features.sampled_variables) if configs.multilayers else input_features.variable_num,
        in_channels=in_channels,
        out_channels=input_features.nF,
        multiport=port_num if multiport else 0,
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
    
    count_parameters(net)
    initialize_weights(net)
    maximum_loss = 100.0
    epoch_continue = 0
    
    # Continue training
    if configs.ckpt.read:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            epoch_continue = ckpt['epoch']
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                
    # run the training loop for defined number of epochs
    train_losslogger = LossLogger()
    val_losslogger = LossLogger()

    for epoch in range(configs.num_epochs):
        # Define real epoch 

        if epoch < epoch_continue:
            continue
        
        # Train an epoch
        _train_epoch(
            epoch=epoch, 
            net=net, 
            optimizer=optimizer, 
            criterion=criterion,
            writer=writer, 
            output_col=output_log_name, 
            train_dataloader=train_dataloader, 
        )

        # Validation
        if isepoch(epoch, configs.validation.valid_per_epoch):
            _val_epoch(
                epoch=epoch, 
                net=net, 
                criterion=criterion,
                writer=writer, 
                output_col=output_log_name, 
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader, 
                train_losslogger=train_losslogger,
                val_losslogger=val_losslogger,
                output_log_file_name=output_log_file_name,
            )
            if val_losslogger.loss < maximum_loss:
                maximum_loss = val_losslogger.loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'max_min': (max_tensor, min_tensor)
                }, best_path)
        
        # Plot the figure
        if configs.fig.plot_during_train and isepoch(epoch, configs.fig.plot_per_epoch):
            # Get train examples

            example_num = configs.fig.example_num
            
            if configs.datasets.mode == "normal" or configs.datasets.mode == "port":
                # Define data loaders for training and testing data in this fold
                train_example_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                                    batch_size=example_num)
                val_example_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                                    batch_size=example_num, 
                                                                    sampler=val_subsampler)
            elif configs.datasets.mode == "freq":
                # Define data loaders for training and testing data in this fold
                train_example_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                                    batch_size=example_num * input_features.nF)
                val_example_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                                    batch_size=example_num * input_features.nF)
            elif configs.datasets.mode == "final" or configs.datasets.mode == "final_multi_column":
                train_example_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                                    batch_size=example_num)
                val_example_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                                    batch_size=example_num, 
                                                                    sampler=val_subsampler)
            
            else:
                raise NotImplementedError
            X_example_train, y_example_train = next(iter(train_example_dataloader))
            X_example_val, y_example_val = next(iter(val_example_dataloader))
            X_example, y_example = torch.cat((X_example_train, X_example_val)), torch.cat((y_example_train, y_example_val))
            output = net(X_example)
            # Denormalize
            y_example = train_dataset.denormalize_output(y_example)
            output = train_dataset.denormalize_output(output)
            # Change shape
            if configs.datasets.mode == "freq":
                y_example = torch.reshape(y_example, shape=[example_num, -1])
                output = torch.reshape(output, shape=[example_num, -1])
            # Plot and save figure
            fig = plot_X_y(
                input_features.frequency_np, 
                y_example, 
                output, 
                epoch + 1,
                output_col, 
                configs,
                multiport)
            if writer is not None:
                writer.add_figure(f'{output_log_name}_Pred', fig)

        # Save checkpoints
        if configs.ckpt.save and isepoch(epoch, configs.ckpt.save_per_epoch):
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'max_min': (max_tensor, min_tensor)
            }, ckpt_path)    
        
        # early stopping 
        if hasattr(configs.ckpt, 'early_stop_thres') and isepoch(epoch, configs.validation.valid_per_epoch):
            early_stop_thres = configs.ckpt.early_stop_thres
            if val_losslogger.loss < early_stop_thres:
                break

    torch.cuda.empty_cache()
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
    log_dir = os.path.join(home_dir, 'Logs', case)
    if not os.path.exists(model_dir) and (not configs.distributed or rank == 0):
        os.makedirs(model_dir) 
    configs.model_dir = model_dir
    if not os.path.exists(log_dir) and (not configs.distributed or rank == 0):
        os.makedirs(log_dir) 
    configs.log_dir = log_dir
    
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

    # port_list = ['(1,1)', '(1,11)']
    # final_multi_column mode
    layer_num = input_features.layer_num
    port_num = configs.node
    port_per_layer = (port_num // 2) // layer_num 
    if port_num % layer_num != 0:
        raise ValueError("port number must be integer number of layer number")
    port_list = []
    data_cols = []
    trained = ['(1,1)','(1,2)']
    for i in range(layer_num):
        for j in range(2):
            if f'({i+1},{j+1})' in trained:
                continue
            port_list_i = []
            data_cols_i = []
            for m in range(2):
                for k in range(i, layer_num):
                    port_list_i.append(f'({i * port_per_layer + 1},{k * port_per_layer + 1 + j + m * port_num // 2})')
            for m in range(len(port_list_i)):
                pi, pj = int(port_list_i[m][1:-1].split(',')[0]), int(port_list_i[m][1:-1].split(',')[1])
                data_col_ii = []
                for k in range(port_per_layer - pj % port_per_layer + 1):
                    data_col_ii.append(f'({pi+k},{pj+k})')
                data_cols_i.append(data_col_ii)
            port_list.append(port_list_i)
            data_cols.append(data_cols_i)

    writer = None

    # check the port list
    # for i in range(len(port_list)):
    #     print(port_list[i])
    #     print(data_cols[i])

    for i in range(len(port_list)):
        train(
            configs=configs,
            input_features=input_features,
            output_col=port_list[i],
            writer=writer,
            device=torch.device("cuda:0"),
            data_cols=data_cols[i]
            )
    return

if __name__ == "__main__":
    main()