import os
from cores import *
from utils.config import Config
from utils.miscellaneous import get_filename
def preprocess_config(
    args,
    write_results = False
):
    configs = Config()
    configs.load(args.setting, recursive=True)
    rank = 0
    case = configs.case
    configs.trial = get_filename(args.setting)[0]

    # model path
    current_dir = os.getcwd()
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if not os.path.exists(os.path.join(home_dir, 'Models')) and (not configs.distributed or rank == 0):
        os.makedirs(os.path.join(home_dir, 'Models'))
    model_dir = os.path.join(home_dir, 'Models', case)
    if not os.path.exists(model_dir) and (not configs.distributed or rank == 0):
        os.makedirs(model_dir) 
    configs.model_dir = model_dir
    
    # define fig dir as (pdf_dir, png_dir)
    if not os.path.exists(configs.fig.dir) and (not configs.distributed or rank == 0):
        os.makedirs(configs.fig.dir)
    if not os.path.exists(os.path.join(configs.fig.dir, 'pdf')) and (not configs.distributed or rank == 0):
        os.makedirs(os.path.join(configs.fig.dir, 'pdf'))
    if not os.path.exists(os.path.join(configs.fig.dir, 'png')) and (not configs.distributed or rank == 0):
        os.makedirs(os.path.join(configs.fig.dir, 'png'))

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

    if write_results:
        if not os.path.exists(os.path.join(home_dir, 'Results')) and (not configs.distributed or rank == 0):
            os.makedirs(os.path.join(home_dir, 'Results'))
        results_dir = os.path.join(home_dir, 'Results', case)
        if not os.path.exists(results_dir) and (not configs.distributed or rank == 0):
            os.makedirs(results_dir) 
        configs.results_dir = results_dir
    
    return configs, input_features
