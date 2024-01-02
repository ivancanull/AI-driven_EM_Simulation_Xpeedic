import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import argparse

from torch import threshold

from cores import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    return parser.parse_args()

def evaluate_plot(
    variables,
    f: np.ndarray, 
    X: np.ndarray,
    y: np.ndarray,  
    output_cols_name,
):
    """
    Plot the curves of sr, si, amp, cos, sin for both truth and prediction
    :param X: the x label Frequency 
    :param y: the truth y results, tensor of shape (N, nf, 5)
    :param pred: the predict y results, tensor of shape (N, nf, 5)
    """
    
    line = len(variables)
    sample_num = X.shape[1]
    fig, ax = plt.subplots(line, 2, figsize=(32, 5 * line), constrained_layout=True)

    for i, variable in enumerate(variables):
        for j in range(sample_num):
            # SR
            ax[i, 0].plot(f, y[i, j, 0, :], label=f'SR{output_cols_name}, {variable}={X[i, j]}')
            ax[i, 1].plot(f, y[i, j, 1, :], label=f'SI{output_cols_name}, {variable}={X[i, j]}')
            
    for i in range(line):
        for j in range(2):
            ax[i, j].legend(loc='upper right')
            ax[i, j].set_xlabel('Frequency (Hz)')
            ax[i, j].set_ylabel('S-parameter (dB)')

    # Save figures
    pdf_dir, png_dir = '../Figures/pdf/evaluate_p9', '../Figures/png/evaluate_p9' 
    fig.suptitle(f'Evaluation Results of Port{output_cols_name}')

    plt.savefig(f'{pdf_dir}/Evaluate_{output_cols_name}.pdf')
    plt.savefig(f'{png_dir}/Evaluate_{output_cols_name}.png')
    return fig

def evaluate(
    cols,
    variable: str,
    df: pd.DataFrame,

    num: int,
):
    df.sort_values(variable, axis=0, inplace=True)
    parameter = np.array(df.loc[:, variable])
    output = np.array(df.loc[: , cols])
    # Get indices
    indices = np.linspace(0, parameter.shape[0]-1, num, dtype=int)
    
    return parameter[indices], output[indices].reshape(num, 2, -1)

def main():
    # Read data
    p9_dir = '../Data/Dataset/p_9'
    working_dir = '/public/home/zhanghf/AI_EM/AI-driven_EM_Simulation/Codes'
    config_dir = '../Data/Config/Stripline_Diff-Pair_10-parameter.json'

    # Parse input features config
    input_features = InputFeatures(config_dir)
    variables, ranges, decimals = input_features.get_variable_range()

    # Parse freq
    f = parse_frequency(input_features.frequency)
    nF = len(f)
    variable_num = len(variables)
    output_cols = ['SR(1,1)', 'SI(1,1)']
    cols = []

    # Expand output cols
    for i in range(2):
        for j in range(nF):
            cols.append(output_cols[i]+'_%d'%j)
    sample_num = 20
    X_np = np.zeros((variable_num, sample_num))
    y_np = np.zeros((variable_num, sample_num, 2, nF))

    # Sweep the parameters list
    for idx, variable in enumerate(variables):
        df = pd.read_pickle(os.path.join(working_dir, '../Data/Dataset/p_9/p_9_Stripline_Diff-Pair_10-parameter_iter_1_%s.zip' % variable), compression='zip')
        X, y = evaluate(cols, variable, df, sample_num)
        X_np[idx, ...] = X
        y_np[idx, ...] = y

    # Plot X and y
    evaluate_plot(variables, f, X_np, y_np, '(1,1)')
    return

def evaluate_multilayer():
    samples_dir = 'D:\\Samples\\Final_Cases\\Final_Cases_ws_sweep_1012\\Final_Cases_ws_sweep_1012_0\\Optimetrics\\ParametricSetup1\\'

    # define fig dir as (pdf_dir, png_dir)
    pdf_dir = os.path.join('..\\Figures', 'pdf', 'evaluate_multilayer')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join('..\\Figures', 'png', 'evaluate_multilayer')
    if not os.path.exists(png_dir):
        os.makedirs(png_dir) 

    def evaluate_multilayer_plot(
        samples_dir,
        port_list,
        title,
        name
    ):
        line = len(port_list)
        nF = 500
        f = np.arange(nF)
        oberserve_number = 5
        fig, ax = plt.subplots(oberserve_number, 2, figsize=(16 * 2, 5 * oberserve_number), constrained_layout=True)
        for i in range(oberserve_number):
            for j, port in enumerate(port_list):
                df = pd.read_pickle(os.path.join(samples_dir, f'{i+1}', f'{port[0]}_{port[1]}.zip'))
                sr_cols = [f'SR({port[0]},{port[1]})_{k}' for k in range(nF)]
                si_cols = [f'SI({port[0]},{port[1]})_{k}' for k in range(nF)]
            
                ax[i, 0].plot(f, df[sr_cols].to_numpy().reshape(-1), label=f'SR({port[0],port[1]}) of case {i+1}', alpha=0.5)
                ax[i, 1].plot(f, df[si_cols].to_numpy().reshape(-1), label=f'SI({port[0],port[1]}) of case {i+1}', alpha=0.5)

        for i in range(oberserve_number):
            for j in range(2):
                ax[i, j].legend(loc='upper right')
                ax[i, j].set_xlabel('Frequency (Hz)')
                ax[i, j].set_ylabel('S-parameter')
        # Save figures
        fig.suptitle(f'Evaluation Results of {title} S-parameters')

        plt.savefig(f'{pdf_dir}/Evaluate_{name}.pdf')
        plt.savefig(f'{png_dir}/Evaluate_{name}.png')

    def evaluate_up_and_down(
        samples_dir,
    ):
        port_list = [(1,1),
                     (1,11),
                     (1,21),
                     (1,31),
                     (1,41)]
        title = 'Up and Down'
        name = 'up_and_down'
        evaluate_multilayer_plot(samples_dir, port_list, title, name)

    def evaluate_up_and_down_back(
        samples_dir,
    ):
        port_list = [(1,51),
                     (1,61),
                     (1,71),
                     (1,81),
                     (1,91)]
        title = 'Up and Down Back'
        name = 'up_and_down_back'
        evaluate_multilayer_plot(samples_dir, port_list, title, name)

    def evaluate_same_layer(
        samples_dir,
    ):
        port_list = [(2,2),
                     (3,3),
                     (4,4),
                     (5,5),
                     (6,6),
                     (7,7),
                     (8,8)]
        title = 'Same Layer Difference'
        name = 'same_layer_difference'
        # evaluate_multilayer_plot(samples_dir, port_list, title, name)

        nF = 500
        f = np.arange(nF)
        oberserve_number = 10

        sr_diff_total = np.zeros((len(port_list), len(port_list)))
        si_diff_total = np.zeros((len(port_list), len(port_list)))

        sr_average_diff = np.zeros((oberserve_number))
        si_average_diff = np.zeros((oberserve_number))

        average_array = np.zeros((oberserve_number, 2, nF))
        for i in range(oberserve_number):
            for j, port in enumerate(port_list):
                df = pd.read_pickle(os.path.join(samples_dir, f'{i+1}', f'{port[0]}_{port[1]}.zip'))
                sr_cols = [f'SR({port[0]},{port[1]})_{k}' for k in range(nF)]
                si_cols = [f'SI({port[0]},{port[1]})_{k}' for k in range(nF)]

                average_array[i, 0, ...] += df[sr_cols].to_numpy().reshape(-1)
                average_array[i, 1, ...] += df[si_cols].to_numpy().reshape(-1)

        average_array /= len(port_list)

        for i in range(oberserve_number):
            
            array = np.zeros((len(port_list), 2, nF))
            sr_diff = np.zeros((len(port_list), len(port_list)))
            si_diff = np.zeros((len(port_list), len(port_list)))
            
            for j, port in enumerate(port_list):
            
                df = pd.read_pickle(os.path.join(samples_dir, f'{i+1}', f'{port[0]}_{port[1]}.zip'))

                sr_cols = [f'SR({port[0]},{port[1]})_{k}' for k in range(nF)]
                si_cols = [f'SI({port[0]},{port[1]})_{k}' for k in range(nF)]

                array[j, 0, ...] = df[sr_cols].to_numpy()
                array[j, 1, ...] = df[si_cols].to_numpy()

            # calculate sr difference
            
            for t in range(len(port_list)):
                sr_average_diff[i] += np.mean(np.abs(array[t:(t+1), 0, :] - average_array[i, 0, ...])) / np.ptp(array[t:(t+1), 0, :])
                si_average_diff[i] += np.mean(np.abs(array[t:(t+1), 1, :] - average_array[i, 1, ...])) / np.ptp(array[t:(t+1), 1, :])
                for k in range(len(port_list)):
                    sr_diff[t, k] = np.mean(np.abs(array[t:(t+1), 0, :] - array[k:(k+1), 0, :])) / np.ptp(array[t:(t+1), 0, :])
                    sr_diff_total[t, k] += sr_diff[t, k]
                    si_diff[t, k] = np.mean(np.abs(array[t:(t+1), 1, :] - array[k:(k+1), 1, :])) / np.ptp(array[t:(t+1), 1, :])
                    si_diff_total[t, k] += si_diff[t, k]

            print(f'sr_diff of case {i+1}:')
            print(sr_diff)
            print(f'si_diff of case {i+1}:')
            print(si_diff)
            
        sr_diff = sr_diff_total / (oberserve_number)
        si_diff = si_diff_total / (oberserve_number)
            
        print(f'average abs of sr_diff:')
        print(sr_diff)
        print(f'average abs of si_diff:')
        print(si_diff)

        np.savetxt('sr_diff.csv', sr_diff, delimiter=',')
        np.savetxt('si_diff.csv', si_diff, delimiter=',')

        sr_average_diff /= len(port_list)
        si_average_diff /= len(port_list)

        print(f'average abs of sr_average_diff:')
        print(sr_average_diff)
        print(f'average abs of si_average_diff:')
        print(si_average_diff)

        np.savetxt('sr_average_diff.csv', sr_average_diff, delimiter=',')
        np.savetxt('si_average_diff.csv', si_average_diff, delimiter=',')

    def evaluate_neighbours(
        samples_dir
    ):
        port_list = [(1,2),
                     (1,3),
                     (1,4),
                     (1,5),
                     (1,6)]
        
        title = 'Neighbours'
        name = 'neighbours'
        evaluate_multilayer_plot(samples_dir, port_list, title, name)

    # evaluate_up_and_down(samples_dir)
    # evaluate_same_layer(samples_dir)
    # evaluate_neighbours(samples_dir)
    evaluate_up_and_down_back(samples_dir)

def evaluate_ws():

    def ws_sweep():
        # create a linespace with step
        w = np.arange(3.0, 7.1, 0.1)
        s = np.arange(1.0, 2.025, 0.025)
        ws = np.array(np.meshgrid(w, s)).T.reshape(-1, 2)
        ws = ws.round(2)
        return w, s, ws

    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)

    ds = configs.dataset_generation.name
    output_col = '(1,1)'

    df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{configs.case}/{ds}/{ds}_{output_col}_concat.zip'), compression='zip') 
    s_np = df.to_numpy()
    w, s, para_np = ws_sweep()

    mean_s_np = np.mean(s_np).reshape(1, 1, 1)
    s_np = s_np.reshape(41, 41, 1000)

    # total sum of squares
    t_ss = np.sum((s_np - mean_s_np) ** 2)
    t_mss = t_ss / 41 / 41 / 1000
    print('t_mss: ', t_mss)

    # w sum of squares
    w_ss = np.sum((np.mean(s_np, axis=(1, 2), keepdims=True) - mean_s_np) ** 2) * 1000 * 41
    w_mss = w_ss / 40 
    print('w_mss: ', w_mss)

    # s sum of squres
    s_ss = np.sum((np.mean(s_np, axis=(0, 2), keepdims=True) - mean_s_np) ** 2) * 1000 * 41
    s_mss = s_ss / 40 
    print('s_mss: ', s_mss)

    # plot frequency maps
    
    fig, ax = plt.subplots(11, 2, figsize=(24, 33), constrained_layout=True)

    return
    
    # plot average value
    fig = plt.figure(figsize=(5, 10), constrained_layout=True)

    X, Y = np.meshgrid(w, s)
    sri = ['SR', 'SI']
    for i in range(2):

        # set up a figure twice as wide as it is tall
        ax = fig.add_subplot(2, 1, i + 1, projection='3d')
        Z = np.mean(s_np.reshape(41, 41, 2, 500), axis=-1)[:, :, i]
        
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
        ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
        ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
        ax.set_title(f'Average {sri[i]}{output_col} ')
        ax.set(xlabel='W', ylabel='S', zlabel='S-parameter')

    plt.savefig('../Figures/pdf/Final_Cases_ws_sweep_1012/evaluate_ws_average.pdf')
    plt.savefig('../Figures/png/Final_Cases_ws_sweep_1012/evaluate_ws_average.png')

    
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(25, 10), constrained_layout=True)

    X, Y = np.meshgrid(w, s)
    sri = ['SR', 'SI']
    for i in range(2):
        for j in range(5):

            # set up a figure twice as wide as it is tall
            ax = fig.add_subplot(2, 5, i * 5 + j + 1, projection='3d')
            Z = s_np[:, :, i * 500 + 100 * j]
            
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
            ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
            ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
            ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
            ax.set_title(f'{sri[i]}{output_col} at frequency {j / 5} GHz')
            ax.set(xlabel='W', ylabel='S', zlabel='S-parameter')

    plt.savefig('../Figures/pdf/Final_Cases_ws_sweep_1012/evaluate_ws.pdf')
    plt.savefig('../Figures/png/Final_Cases_ws_sweep_1012/evaluate_ws.png')


def load_dataframe(configs: Config, layer_configs: MultiLayerInputFeatures, output_col: str):

    nF = layer_configs.nF
    ds = configs.dataset_generation.name
    
    df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{configs.case}/{ds}/{ds}_{output_col}_concat.zip'), compression='zip')
    return df, df.columns

def evaluate_ws_2():
    def ws_sweep():
        # create a linespace with step
        w_1_ = np.arange(20, 55, 5)
        w_2_ = np.arange(20, 55, 5)
        w_3_ = np.arange(20, 55, 5)
        w = np.array(np.meshgrid(w_1_, w_2_, w_3_)).T.reshape(-1, 3)
        return w
    
    nF = 500
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    w = ws_sweep()
    f = np.arange(nF)
    line = 7

    # plot 7, 2
    def load_dataframe(output_col):
        ds = configs.dataset_generation.name
        sri = ['SR', 'SI']
        columns = []
        for i in range(nF):
            for j in sri:
                columns.append(f'{j}{output_col}_{i}')
        df = pd.read_pickle(os.path.join(os.getcwd(), f'../Data/Dataset/{configs.case}/{ds}/{ds}_{output_col}_concat.zip'), compression='zip')
        return df, columns
    output_col = '(1,4)'
    df, columns = load_dataframe(output_col)
    fig, ax = plt.subplots(line, 2, figsize=(16, 5 * line), constrained_layout=True)

    for i in range(line):
        for j in range(line):
            index = i * (line ** 2) + i * line + j

            s_np = df.loc[(df['index_2_W'] == w[index, 0]) & (df['index_3_W'] == w[index, 1]) & (df['index_4_W'] == w[index, 2])].loc[:, columns].to_numpy().reshape(nF, 2)


            ax[i, 0].plot(f, s_np[:, 0], label=f'SR{output_col}, w1={w[index, 0]}, w2={w[index, 1]}, w3={w[index, 2]}')
            ax[i, 1].plot(f, s_np[:, 1], label=f'SI{output_col}, w1={w[index, 0]}, w2={w[index, 1]}, w3={w[index, 2]}')

    for i in range(line):
        for j in range(2):
            ax[i, j].legend(loc='upper right')
            if j < 2:
                ax[i, j].set_xlabel('Frequency (Hz)')
            ax[i, j].set_ylabel('S-parameter (dB)')

    plt.savefig('../Figures/pdf/Final_Cases_ws_sweep_1024/evaluate_ws_1_3_2.pdf')
    plt.savefig('../Figures/png/Final_Cases_ws_sweep_1024/evaluate_ws_1_3_2.png') 

    output_col = '(1,4)'
    df, columns = load_dataframe(output_col)

    fig, ax = plt.subplots(line, 2, figsize=(16, 5 * line), constrained_layout=True)
    for i in range(line):
        for j in range(line):

            index = j * (line ** 2) + i + i * line

            s_np = df.loc[(df['index_2_W'] == w[index, 0]) & (df['index_3_W'] == w[index, 1]) & (df['index_4_W'] == w[index, 2])].loc[:, columns].to_numpy().reshape(nF, 2)

            ax[i, 0].plot(f, s_np[:, 0], label=f'SR{output_col}, w1={w[index, 0]}, w2={w[index, 1]}, w3={w[index, 2]}')
            ax[i, 1].plot(f, s_np[:, 1], label=f'SI{output_col}, w1={w[index, 0]}, w2={w[index, 1]}, w3={w[index, 2]}')

    for i in range(line):
        for j in range(2):
            ax[i, j].legend(loc='upper right')
            if j < 2:
                ax[i, j].set_xlabel('Frequency (Hz)')
            ax[i, j].set_ylabel('S-parameter (dB)')

    plt.savefig('../Figures/pdf/Final_Cases_ws_sweep_1024/evaluate_ws_1_2.pdf')
    plt.savefig('../Figures/png/Final_Cases_ws_sweep_1024/evaluate_ws_1_2.png') 

def evaluate_ws_3():
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    configs.trial = get_filename(args.setting)[0]
    
    # define fig dir as (pdf_dir, png_dir)
    pdf_dir = os.path.join(configs.fig.dir, 'pdf', configs.trial)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join(configs.fig.dir, 'png', configs.trial)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir) 

    stackups = load_data(configs, configs.dataset_generation.name)
    layer_configs = MultiLayerInputFeatures(configs.config.dir)

    for i in range(10):
        w = stackups[i * 10].layers[3].w_list[0]
        fig, ax = plt.subplots(10, 10, figsize=(80, 50), constrained_layout=True)
        for pi in range(10):
            for pj in range(10):
                output_col = f'(1,{pi * 10 + pj + 1})'
                df, columns = load_dataframe(configs, layer_configs, output_col)
                for j in range(10):
                    s = stackups[i * 10 + j].layers[3].spaces[0]

                    index = f'{configs.dataset_generation.name}_0_{i * 10 + j + 1}'
                    y = df.loc[index, :].to_numpy().reshape(-1)[: layer_configs.nF]

                    ax[pi, pj].plot(layer_configs.frequency_np, y, label=f'w={w}, s={s}')
                    ax[pi, pj].legend(loc='upper right')
                    ax[pi, pj].set_xlabel('Frequency (Hz)')
                    ax[pi, pj].set_ylabel('S-parameter (dB)')
                    
        plt.savefig(f'{pdf_dir}/W{w}MNS.pdf')
        plt.savefig(f'{png_dir}/W{w}MNS.png')
        print(f'W{w}MNS.pdf is saved')

def evaluate_ws_4():
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    configs.trial = get_filename(args.setting)[0]
    
    # define fig dir as (pdf_dir, png_dir)
    pdf_dir = os.path.join(configs.fig.dir, 'pdf', configs.trial)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join(configs.fig.dir, 'png', configs.trial)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir) 

    input_features = MultiLayerInputFeatures(configs.config.dir)
    stackup_writer = StackupWriter.load_pickle(configs, f'{configs.dataset_generation.name}')
    stackups = stackup_writer.stackups

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)

    ws = np.arange(5,55,5)
    ss = np.arange(10,110,10)
    X, Y = np.meshgrid(ws,ss)

    sri = ['SR', 'SI']
    output_col = '(1,2)'
    df, _ = load_dataframe(configs, input_features, output_col)
    for i in range(2):
        for j in range(5):
            Z = np.zeros([10, 10])
            for wi in range(10):
                for si in range(10):
                    w = stackups[wi * 10].layers[3].w_list[0]
                    s = stackups[wi * 10 + si].layers[3].spaces[0]
                    index = f'{configs.dataset_generation.name}_0_{wi * 10 + si + 1}'
                    y = df.loc[index, :].to_numpy().reshape(-1)
                    Z[wi, si] = y[input_features.nF // 5 * j + i * input_features.nF]

            # set up a figure twice as wide as it is tall
            ax = fig.add_subplot(2, 5, i * 5 + j + 1, projection='3d')

            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='royalblue', lw=0.5, rstride=1, cstride=1, alpha=0.3)
            # ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
            ax.set_title(f'{sri[i]}{output_col} at frequency {j / 5 * 100} GHz')
            ax.set(xlabel='W', ylabel='S', zlabel='S-parameter')

    plt.savefig(f'{pdf_dir}/evaluate_ws_{output_col}.pdf')
    plt.savefig(f'{png_dir}/evaluate_ws_{output_col}.png')

def evaluate_w():
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    configs.trial = get_filename(args.setting)[0]
    
    # define fig dir as (pdf_dir, png_dir)
    pdf_dir = os.path.join(configs.fig.dir, 'pdf', configs.trial)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join(configs.fig.dir, 'png', configs.trial)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir) 

    input_features = MultiLayerInputFeatures(configs.config.dir)
    stackup_writer = StackupWriter.load_pickle(configs, f'{configs.datasets.datasets}')
    stackups = stackup_writer.stackups

    show_point = 20
    
    # # calculate w to S-parameter
    # ws = np.array(input_features.W.values)
    # ws_num = ws.shape[0]
    # for col in range(configs.node):
    #     output_col = f'(1,{col+1})'
    #     sri = ['SR', 'SI']
    #     df, _ = load_dataframe(configs, input_features, output_col)
    #     fig = plt.figure(figsize=(8, 15), constrained_layout=True)
    #     for i in range(2):
    #         # set up a figure twice as wide as it is tall
    #         ax = fig.add_subplot(2, 1, i+1)
    #         for j in range(show_point):
    #             Z = np.zeros([ws_num])
    #             for wi in range(ws_num):
    #                 w = stackups[wi].layers[3].w_list[0]
    #                 index = f'{configs.dataset_generation.name}_0_{wi+1}'
    #                 y = df.loc[index, :].to_numpy().reshape(-1)
    #                 Z[wi] = y[input_features.nF // show_point * j + i * input_features.nF]
    #             ax.plot(ws, Z, label=f'{sri[i]}{output_col} at frequency {int(j / show_point * 100)} GHz')
    #         ax.set(xlabel='W', ylabel='S-parameter')
    #         ax.legend(loc='upper right')
    #     fig.suptitle(f'Evaluation of S-parameter at different W')
    #     plt.savefig(f'{pdf_dir}/evaluate_w_{output_col}.pdf')
    #     plt.savefig(f'{png_dir}/evaluate_w_{output_col}.png')
    #     plt.close()

    # # calculate delta 
    # ws = np.array(input_features.W.values)
    # ws_num = ws.shape[0]
    # for col in range(configs.node):
    #     output_col = f'(1,{col+1})'
    #     sri = ['SR', 'SI']
    #     df, _ = load_dataframe(configs, input_features, output_col)
    #     fig = plt.figure(figsize=(8, 15), constrained_layout=True)
    #     for i in range(2):
    #         # set up a figure twice as wide as it is tall
    #         ax = fig.add_subplot(2, 1, i+1)
    #         for j in range(show_point):
    #             Z = np.zeros([ws_num-2])
    #             for wi in range(1, ws_num-1):
    #                 index_0 = f'{configs.dataset_generation.name}_0_{wi}'
    #                 index_1 = f'{configs.dataset_generation.name}_0_{wi+1}'
    #                 index_2 = f'{configs.dataset_generation.name}_0_{wi+2}'
    #                 y_0 = df.loc[index_0, :].to_numpy().reshape(-1)
    #                 y_1 = df.loc[index_1, :].to_numpy().reshape(-1)
    #                 y_2 = df.loc[index_2, :].to_numpy().reshape(-1)
    #                 y = (np.abs(y_1 - y_0) + np.abs(y_1 - y_2)) / 2
    #                 Z[wi-1] = y[input_features.nF // show_point * j + i * input_features.nF]
    #             ax.plot(ws[1:-1], Z, label=f'Delta of {sri[i]}{output_col} at frequency {int(j / show_point * 100)} GHz')
    #         ax.set(xlabel='W', ylabel='S-parameter')
    #         ax.legend(loc='upper right')
    #     fig.suptitle(f'Evaluation of delta S-parameter at different W')
    #     plt.savefig(f'{pdf_dir}/evaluate_delta_w_{output_col}.pdf')
    #     plt.savefig(f'{png_dir}/evaluate_delta_w_{output_col}.png')
    #     plt.close()

    # # calculate average delta
    # ws = np.array(input_features.W.values)
    # ws_num = ws.shape[0]
    # for col in range(configs.node):
    #     output_col = f'(1,{col+1})'
    #     sri = ['SR', 'SI']
    #     df, _ = load_dataframe(configs, input_features, output_col)
    #     fig = plt.figure(figsize=(8, 15), constrained_layout=True)
    #     for i in range(2):
    #         # set up a figure twice as wide as it is tall
    #         ax = fig.add_subplot(2, 1, i+1)
    #         Z = np.zeros([ws_num-2])
    #         for wi in range(1, ws_num-1):
    #             index_0 = f'{configs.dataset_generation.name}_0_{wi}'
    #             index_1 = f'{configs.dataset_generation.name}_0_{wi+1}'
    #             index_2 = f'{configs.dataset_generation.name}_0_{wi+2}'
    #             y_0 = df.loc[index_0, :].to_numpy().reshape(-1)
    #             y_1 = df.loc[index_1, :].to_numpy().reshape(-1)
    #             y_2 = df.loc[index_2, :].to_numpy().reshape(-1)
    #             y = (np.abs(y_1 - y_0) + np.abs(y_1 - y_2)) / 2
    #             Z[wi-1] = np.mean(y[i * input_features.nF : (i+1) * input_features.nF])
    #         ax.plot(ws[1:-1], Z, label=f'Average delta of S-parameters at port {sri[i]}{output_col}')
    #         ax.set(xlabel='W', ylabel='S-parameter')
    #         ax.legend(loc='upper right')
        
    #     fig.suptitle(f'Evaluation of delta S-parameter at different W')
    #     plt.savefig(f'{pdf_dir}/evaluate_mean_delta_w_{output_col}.pdf')
    #     plt.savefig(f'{png_dir}/evaluate_mean_delta_w_{output_col}.png')
    #     plt.close()

    # use SR/SI (1,1) and (1,11) as reference
    ws = np.array(input_features.W.values)
    ws_num = ws.shape[0]
    fig, ax = plt.subplots(constrained_layout=True)
    Z = np.zeros([ws_num-2])
    for wi in range(1, ws_num-1):
        index_0 = f'{configs.dataset_generation.name}_0_{wi}'
        index_1 = f'{configs.dataset_generation.name}_0_{wi+1}'
        index_2 = f'{configs.dataset_generation.name}_0_{wi+2}'
        y = 0
        for port in ['(1,1)', '(1,11)']:
            df, _ = load_dataframe(configs, input_features, port)
            y_0 = df.loc[index_0, :].to_numpy().reshape(-1)
            y_1 = df.loc[index_1, :].to_numpy().reshape(-1)
            y_2 = df.loc[index_2, :].to_numpy().reshape(-1)
            Z[wi-1] += np.mean(np.abs(y_1 - y_0) + np.abs(y_1 - y_2)) / 2
    print(ws[1:-1].shape)
    print(Z.shape)
    ax.plot(ws[1:-1], Z, label=f'Average delta of S-parameters at port (1,1) and (1,11)')
    ax.set(xlabel='W', ylabel='S-parameter')
    ax.legend(loc='upper right')

    fig.suptitle(f'Evaluation of delta S-parameter at different W')
    plt.savefig(f'{pdf_dir}/evaluate_ref_delta_w.pdf')
    plt.savefig(f'{png_dir}/evaluate_ref_delta_w.png')
    plt.close()

def evaluate():
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    configs.trial = get_filename(args.setting)[0]
    
    # define fig dir as (pdf_dir, png_dir)
    pdf_dir = os.path.join(configs.fig.dir, 'pdf', configs.trial)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join(configs.fig.dir, 'png', configs.trial)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir) 

    input_features = MultiLayerInputFeatures(configs.config.dir)
    stackup_writer = StackupWriter.load_pickle(configs, f'{configs.dataset_generation.name}')
    stackups = stackup_writer.stackups
   
    port_num = configs.node

    # evaluate w and s sweep
    w_num = len(input_features.W.values)
    s_num = len(input_features.S.values)

    x_coord = np.zeros([w_num, s_num, port_num, port_num])
    y_coord = np.zeros([w_num, s_num, port_num, port_num])
    max_s = np.zeros([w_num, s_num, port_num, port_num])
    interested_freq_num = 10
    interested_freq = np.arange(0, input_features.nF, interested_freq_num)
    s_at_certain_frequency = np.zeros([interested_freq_num, w_num, s_num, port_num, port_num])

    for i in range(1):
        for j in range(i, 100):
            output_col = f'({i+1},{j+1})'
            df, columns = load_dataframe(configs, input_features, output_col)
            for wi in range(10):
                for si in range(10):
                    # the first signal layer, also the third in the stackup defines the width and space
                    w = stackups[wi * 10].layers[3].w_list[0]
                    s = stackups[wi * 10 + si].layers[3].spaces[0]
                    index = f'{configs.dataset_generation.name}_0_{wi * 10 + si + 1}'
                    y = df.loc[index, :].to_numpy().reshape(-1)
                    y_abs = np.sqrt(y[: input_features.nF] ** 2 + y[input_features.nF:] ** 2)
                    # calculate max_s
                    max_s[wi, si, i, j] = np.max(y_abs)
                    # calculate w_s
                    xi, yi = i % 50, i // 50
                    xj, yj = j % 50, j // 50
                    x_coord[wi, si, i, j] = (xj - xi) * (w + s)
                    y_coord[wi, si, i, j] = (yj - yi)
                    # calculate intereted frequency points
                    for k in range(interested_freq_num):
                        s_at_certain_frequency[k, wi, si, i, j] = y[interested_freq[k]]
    
    # plot max_s to port
    fig, ax = plt.subplots(w_num, s_num, figsize=(80, 50), constrained_layout=True)
    for i in range(w_num):
        for j in range(s_num):
            print(x_coord[i, j, 0, :])
            ax[i, j].plot(x_coord[i, j, 0, : port_num // 2], max_s[i, j, 0, : port_num // 2])
            ax[i, j].set_xlim([0, 750])
            ax[i, j].set_yscale('log')
            ax[i, j].set_title(f'w={input_features.W.values[i]}, s={input_features.S.values[j]}')
            ax[i, j].set_xlabel('distance (um)')
            ax[i, j].set_ylabel('S-parameter (dB)')
    plt.savefig(f'{pdf_dir}/max_s_to_port.pdf')
    plt.savefig(f'{png_dir}/max_s_to_port.png')

    # plot 

if __name__ == "__main__":
    evaluate_w()
    