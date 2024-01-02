import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from typing import List

__all__ = [
    "plot_X_y",
    "plot_X_y_test",
    "plot_X_y_physics",
    "plot_X_y_hilbert",
    "plot_RLGC_example",
    "plot_RLGC"
]

def plot_X_y(
    f: np.ndarray, 
    y: torch.Tensor, 
    pred: torch.Tensor, 
    epoch,
    output_cols_name,
    configs
):
    """
    Plot the curves of sr, si, amp, cos, sin for both truth and prediction
    :param X: the x label Frequency 
    :param y: the truth y results, tensor of shape (N, nf, 5)
    :param pred: the predict y results, tensor of shape (N, nf, 5)
    """
    
    line = y.shape[0] 
    if line % 2 != 0:
        raise ValueError("Example num must be even number!")
    
    if isinstance(y, Tensor):
        y = y.cpu().detach().numpy()
    if isinstance(pred, Tensor):
        pred = pred.cpu().detach().numpy()

    fig, ax = plt.subplots(line // 2, 2, figsize=(16, 5 * line // 2), constrained_layout=True)
    srsi = (f.shape[0] == (y.shape[1] // 2))
    for ii, labels in enumerate(['train', 'val']):
        for i in range(line // 2):

            if srsi:
                ax[i, ii].plot(f, y[i + ii * line // 2, : f.shape[0]], label=f'truth of sr {labels}_data')
                ax[i, ii].plot(f, y[i + ii * line // 2, f.shape[0] :], label=f'truth of si {labels}_data')
            else:
                ax[i, ii].plot(f, y[i + ii * line // 2, :], label=f'truth of {labels}_data')
            if pred is not None:

                if srsi:
                    ax[i, ii].plot(f, pred[i + ii * line // 2, : f.shape[0]], label=f'pred of sr {labels}_data')
                    ax[i, ii].plot(f, pred[i + ii * line // 2, f.shape[0]: ], label=f'pred of si {labels}_data')
                else:
                    ax[i, ii].plot(f, pred[i + ii * line // 2, :], label=f'pred of {labels}_data')

            
    for i in range(line // 2):
        for j in range(2):
            ax[i, j].legend(loc='upper right')
            ax[i, j].set_xlabel('Frequency (Hz)')
            ax[i, j].set_ylabel('S-parameter (dB)')

    # Save figures
    pdf_dir, png_dir = configs.fig_dir
    fig.suptitle(f'Training and Validation Results of Port{output_cols_name} in Epoch {epoch}')

    plt.savefig(f'{pdf_dir}/train_{output_cols_name}_epoch_{epoch}.pdf')
    plt.savefig(f'{png_dir}/train_{output_cols_name}_epoch_{epoch}.png')
    return fig

def plot_X_y_test(
    f: torch.Tensor, 
    y: torch.Tensor, 
    pred: torch.Tensor, 
    output_cols_name,
    configs,
    prefix: str = 'test'
):
    """
    Plot the curves of sr, si, amp, cos, sin for both truth and prediction
    :param X: the x label Frequency 
    :param y: the truth y results, tensor of shape (N, nf, 5)
    :param pred: the predict y results, tensor of shape (N, nf, 5)
    """
    
    line = y.shape[0] 
    fig, ax = plt.subplots(line, figsize=(16, 5 * line), constrained_layout=True)
    srsi = (f.shape[0] == (y.shape[1] // 2))
    for i in range(line):
        
        if srsi:
            
            ax[i].plot(f, y[i, : f.shape[0]].cpu().detach().numpy(), label=f'truth of sr test_data')
            ax[i].plot(f, y[i, f.shape[0] :].cpu().detach().numpy(), label=f'truth of si test_data')
        else:
            ax[i].plot(f, y[i, :].cpu().detach().numpy(), label=f'truth of test_data')

        if pred is not None:
            if srsi:
                ax[i].plot(f, pred[i, : f.shape[0]].cpu().detach().numpy(), label=f'pred of sr test_data')
                ax[i].plot(f, pred[i, f.shape[0] :].cpu().detach().numpy(), label=f'pred of si test_data')
            else:
                ax[i].plot(f, pred[i, :].cpu().detach().numpy(), label=f'pred of test_data')
        
    for i in range(line):
        ax[i].legend(loc='upper right')
        ax[i].set_xlabel('Frequency (Hz)')
        ax[i].set_ylabel('S-parameter (dB)')

    # Save figures
    pdf_dir, png_dir = configs.fig_dir
    fig.suptitle(f'Test Results of Port{output_cols_name}')

    plt.savefig(f'{pdf_dir}/{prefix}_{output_cols_name}.pdf')
    plt.savefig(f'{png_dir}/{prefix}_{output_cols_name}.png')
    return fig

def plot_X_y_physics(
    f: torch.Tensor, 
    y: torch.Tensor, 
    pred: torch.Tensor, 
    configs,
    output_cols_name,
    dc: torch.Tensor = None,
    predict_dc: torch.Tensor = None,
    envelop: List[np.ndarray] = None,
    ifft: List[np.ndarray] = None,
    predict_pulse_location: np.ndarray = None,
    predict_pulse_peak: np.ndarray = None,
    prefix: str = 'physics'
):
    
    """
    Plot the curves of sr, si, DC and envelop
    :param X: the x label Frequency 
    :param y: the truth y results, tensor of shape (N, 2, nf)
    """
    
    line = y.shape[0] 
    ifft_len = ifft[0].shape[1]
    if ifft is None:
        col = 2
    else:
        col = 3

    fig, ax = plt.subplots(line, col, figsize=(16 * col, 5 * line), constrained_layout=True)

    for i in range(line):
        
        ax[i, 0].plot(f, y[i, 0, :].cpu().detach().numpy(), label=f'sr truth')
        ax[i, 1].plot(f, y[i, 1, :].cpu().detach().numpy(), label=f'si truth')

        if dc is not None:
            ax[i, 0].plot([f[0].item(), f[-1].item()], [dc[i, 0].cpu().detach().item(), dc[i, 0].cpu().detach().item()], label=f'sr dc')
            ax[i, 1].plot([f[0].item(), f[-1].item()], [dc[i, 1].cpu().detach().item(), dc[i, 1].cpu().detach().item()], label=f'si dc')
        if predict_dc is not None:
            ax[i, 0].plot([f[0].item(), f[-1].item()], [predict_dc[i, 0].cpu().detach().item(), predict_dc[i, 0].cpu().detach().item()], label=f'sr predict dc')
            ax[i, 1].plot([f[0].item(), f[-1].item()], [predict_dc[i, 1].cpu().detach().item(), predict_dc[i, 1].cpu().detach().item()], label=f'si predict dc')

        if envelop is not None:
            ax[i, 0].plot(f, envelop[0][i, 0, :], label=f'sr envelop')
            ax[i, 1].plot(f, envelop[0][i, 1, :], label=f'si envelop')
            ax[i, 0].plot(f, envelop[1][i, 0, :], label=f'sr envelop')
            ax[i, 1].plot(f, envelop[1][i, 1, :], label=f'si envelop')

        if pred is not None:
            ax[i, 0].plot(f, np.real(pred[i]), label=f'sr prediction')
            ax[i, 1].plot(f, np.imag(pred[i]), label=f'si prediction')
        
        if ifft is not None:
            ax[i, 2].plot(np.arange(ifft_len), np.real(ifft[0][i]), label=f'ifft')
            ax[i, 2].plot(np.arange(ifft_len), np.real(ifft[1][i]), label=f'ifft predict')

        if predict_pulse_location is not None and predict_pulse_peak is None:
            ax[i, 2].axvline(x=int(predict_pulse_location[i]))
        elif predict_pulse_peak is not None:
            ax[i, 2].stem([int(predict_pulse_location[i])], [predict_pulse_peak[i]])
        


    for i in range(line):
        for j in range(col):
            ax[i, j].legend(loc='upper right')
            if j < 2:
                ax[i, j].set_xlabel('Frequency (Hz)')
            ax[i, j].set_ylabel('S-parameter (dB)')

    # Save figures
    pdf_dir, png_dir = configs.fig_dir
    fig.suptitle(f'Test Results of Port{output_cols_name}')

    plt.savefig(f'{pdf_dir}/{prefix}_{output_cols_name}.pdf')
    plt.savefig(f'{png_dir}/{prefix}_{output_cols_name}.png')
    return fig

def plot_X_y_hilbert(
    f: torch.Tensor, 
    y: torch.Tensor, 
    pred: torch.Tensor, 
    configs,
    output_cols_name,
    hilbert = None,
    dc: torch.Tensor = None,
    predict_dc: torch.Tensor = None,
    envelop: List[np.ndarray] = None,
    ifft: List[np.ndarray] = None,
    predict_pulse_location: np.ndarray = None,
    predict_pulse_peak: np.ndarray = None,
    prefix: str = 'hilbert'
):
    
    """
    Plot the curves of sr, si, DC and envelop
    :param X: the x label Frequency 
    :param y: the truth y results, tensor of shape (N, 2, nf)
    """
    
    line = y.shape[0] 
    if ifft is not None:
        ifft_len = ifft[0].shape[1]
    if hilbert is None:
        col = 2
    else:
        col = 4

    fig, ax = plt.subplots(line, col, figsize=(16 * col, 5 * line), constrained_layout=True)

    for i in range(line):
        
        ax[i, 0].plot(f, y[i, 0, :].cpu().detach().numpy(), label=f'sr truth')
        ax[i, 1].plot(f, y[i, 1, :].cpu().detach().numpy(), label=f'si truth')

        if hilbert is not None:
            ax[i, 2].plot(f, np.real(hilbert[i]), label=f'hilbert real')
            ax[i, 3].plot(f, np.imag(hilbert[i]), label=f'hilbert real')

        # if dc is not None:
        #     ax[i, 0].plot([f[0].item(), f[-1].item()], [dc[i, 0].cpu().detach().item(), dc[i, 0].cpu().detach().item()], label=f'sr dc')
        #     ax[i, 1].plot([f[0].item(), f[-1].item()], [dc[i, 1].cpu().detach().item(), dc[i, 1].cpu().detach().item()], label=f'si dc')
        # if predict_dc is not None:
        #     ax[i, 0].plot([f[0].item(), f[-1].item()], [predict_dc[i, 0].cpu().detach().item(), predict_dc[i, 0].cpu().detach().item()], label=f'sr predict dc')
        #     ax[i, 1].plot([f[0].item(), f[-1].item()], [predict_dc[i, 1].cpu().detach().item(), predict_dc[i, 1].cpu().detach().item()], label=f'si predict dc')

        # if envelop is not None:
        #     ax[i, 0].plot(f, envelop[0][i, 0, :], label=f'sr envelop')
        #     ax[i, 1].plot(f, envelop[0][i, 1, :], label=f'si envelop')
        #     ax[i, 0].plot(f, envelop[1][i, 0, :], label=f'sr envelop')
        #     ax[i, 1].plot(f, envelop[1][i, 1, :], label=f'si envelop')

        # if pred is not None:
        #     ax[i, 0].plot(f, np.real(pred[i]), label=f'sr prediction')
        #     ax[i, 1].plot(f, np.imag(pred[i]), label=f'si prediction')
        
        # if ifft is not None:
        #     ax[i, 2].plot(np.arange(ifft_len), np.real(ifft[0][i]), label=f'ifft')
        #     ax[i, 2].plot(np.arange(ifft_len), np.real(ifft[1][i]), label=f'ifft predict')

        # if predict_pulse_location is not None and predict_pulse_peak is None:
        #     ax[i, 2].axvline(x=int(predict_pulse_location[i]))
        # elif predict_pulse_peak is not None:
        #     ax[i, 2].stem([int(predict_pulse_location[i])], [predict_pulse_peak[i]])
        


    for i in range(line):
        for j in range(col):
            ax[i, j].legend(loc='upper right')
            if j < 2:
                ax[i, j].set_xlabel('Frequency (Hz)')
            ax[i, j].set_ylabel('S-parameter (dB)')

    # Save figures
    pdf_dir, png_dir = configs.fig_dir
    fig.suptitle(f'Test Results of Port{output_cols_name}')

    plt.savefig(f'{pdf_dir}/{prefix}_{output_cols_name}.pdf')
    plt.savefig(f'{png_dir}/{prefix}_{output_cols_name}.png')
    return fig

def plot_RLGC(
    f: torch.Tensor, 
    y: torch.Tensor, 
    pred: torch.Tensor, 
    epoch,
    output_cols_name,
    configs
):
    line = y.shape[0]

    mode = output_cols_name[0] 

    if line % 2 != 0:
        raise ValueError("Example num must be even number!")
    
    fig, ax = plt.subplots(line // 2, 2, figsize=(16, 5 * line // 2), constrained_layout=True)
    
    for ii, labels in enumerate(['train', 'val']):
        for i in range(line // 2):

            if mode == 'L':
                ax[i, ii].plot(f[1:], y[i + ii * line // 2, 0, :].cpu().detach().numpy(), label=f'{mode} truth of {labels}_data')
            elif mode == 'C':
                ax[i, ii].plot(f, np.repeat(y[i + ii * line // 2, 0, :].cpu().detach().numpy(), len(f), axis=-1), label=f'{mode} truth of {labels}_data')
            elif mode == 'G':
                ax[i, ii].plot(f, np.linspace(0, y[i + ii * line // 2, 0, :].cpu().detach().numpy(), len(f)), label=f'{mode} truth of {labels}_data')
            else:
                ax[i, ii].plot(f, y[i + ii * line // 2, 0, :].cpu().detach().numpy(), label=f'{mode} truth of {labels}_data')

            if pred is not None:
                if mode == 'L':
                    ax[i, ii].plot(f[1:], pred[i + ii * line // 2, 0, :].cpu().detach().numpy(), label=f'{mode} pred of {labels}_data')
                elif mode == 'C':
                    ax[i, ii].plot(f, np.repeat(pred[i + ii * line // 2, 0, :].cpu().detach().numpy(), len(f), axis=-1), label=f'{mode} truth of {labels}_data')
                elif mode == 'G':
                    ax[i, ii].plot(f, np.linspace(0, pred[i + ii * line // 2, 0, :].cpu().detach().numpy(), len(f)), label=f'{mode} truth of {labels}_data')                
                else:
                    ax[i, ii].plot(f, pred[i + ii * line // 2, 0, :].cpu().detach().numpy(), label=f'{mode} pred of {labels}_data')
            
    for i in range(line // 2):
        for j in range(2):
            ax[i, j].legend(loc='upper right')
            ax[i, j].set_xlabel('Frequency (Hz)')
            ax[i, j].set_ylabel(mode)

    # Save figures
    pdf_dir, png_dir = configs.fig_dir
    fig.suptitle(f'Training and Validation Results of {output_cols_name} in Epoch {epoch}')

    plt.savefig(f'{pdf_dir}/train_{output_cols_name}_epoch_{epoch}.pdf')
    plt.savefig(f'{png_dir}/train_{output_cols_name}_epoch_{epoch}.png')
    return fig


def plot_RLGC_example(
    f: torch.Tensor, 
    y: torch.Tensor, 
    # pred: torch.Tensor, 
    configs,
    output_cols_name,
    # hilbert = None,
    # dc: torch.Tensor = None,
    # predict_dc: torch.Tensor = None,
    # envelop: List[np.ndarray] = None,
    # ifft: List[np.ndarray] = None,
    # predict_pulse_location: np.ndarray = None,
    # predict_pulse_peak: np.ndarray = None,
    prefix: str = 'hilbert'
):
    line = y.shape[0] 
    title_list = [['R', 'L'], ['G', 'C']]
    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    for i in range(line):
        
        # Plot R
        ax[0, 0].plot(f, y[i, 0, :].cpu().detach().numpy())
        
        # Plot L
        ax[0, 1].plot(f, y[i, 1, :].cpu().detach().numpy())

        # Plot G
        ax[1, 0].plot(f, y[i, 3, :].cpu().detach().numpy())

        # Plot C
        ax[1, 1].plot(f, y[i, 2, :].cpu().detach().numpy())

    for i in range(2):
        for j in range(2):
            # ax[i, j].legend(loc='upper right')
            ax[i, j].set_xlabel('Frequency (Hz)')
            ax[i, j].set_title(f'{title_list[i][j]}')
            
    # Save figures
    pdf_dir, png_dir = configs.fig_dir
    fig.suptitle(f'RLGC of {output_cols_name}')

    plt.savefig(f'{pdf_dir}/{prefix}_{output_cols_name}.pdf')
    plt.savefig(f'{png_dir}/{prefix}_{output_cols_name}.png')
    return fig
