from random import sample
import argparse
import pandas as pd


from typing import List
import numpy as np
from utils import *
from cores.datasets.multilayer_input_features import *
from cores.builder import *
from cores.samplers import *
# supported_metis_variables = ["W", "S", "H", "T"]
# supported_metis_variables += ["W_1", "W_2", "W_3", "W_4", "W_5"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', type=str, help='program setting file')
    parser.add_argument('-m', '--mode', default='metis', type=str, help='generate samples for metis or tml')
    return parser.parse_args()

def main():
    args = parse_args()
    configs = Config()
    configs.load(args.setting, recursive=True)
    if args.mode == 'metis':
        generate_metis(configs)
    elif args.mode == 'metis_iter':
        generate_metis_iter(configs)
    elif args.mode == 'tml':
        pass

def generate_metis_iter(configs: Config):
    # TODO
    stackups = []
    input_features = MultiLayerInputFeatures(configs.config.dir)

    # stackup writer
    for loaded_dataset in configs.dataset_generation.loaded_datasets:
        stackup_writer = StackupWriter.load_pickle(configs, loaded_dataset)
        stackups.extend(stackup_writer.stackups)

    # create numpy array of input features from loaded stackups
    samples_np = create_samples_from_stackups(stackups=stackups,
                                              input_features=input_features,
                                              method=configs.differentline if hasattr(configs, 'differentline') else 'same_line')
    # create numpy array of input features from loaded stackups
    if hasattr(configs.dataset_generation, 'train'):
        sampler = make_sampler(input_features=input_features, 
                               method=configs.dataset_generation.train.sampling_method)
        sampling_num = configs.dataset_generation.train.sampling_num
        
        for i in range(sampling_num):
            to_sample = True
            while to_sample:
                if isinstance(sampler, UniformSampler):
                    new_sample = sampler.sample(sample_num=1)
                elif isinstance(sampler, GuidedUniformSampler):
                    new_sample = sampler.sample(sample_num=1, trial_num=10)
                else:
                    raise NotImplementedError
                if np.where((samples_np == new_sample).all(axis=1))[0].size == 0:
                    samples_np = np.concatenate((samples_np, new_sample))
                    to_sample = False
                else:
                    pass
        
        stackups = create_stackups_from_samples(input_features=input_features,
                                                samples_np=samples_np[-sampling_num:],
                                                layer_num=input_features.layer_num,
                                                method=configs.differentline if hasattr(configs, 'differentline') else 'same_line')
        write_to_files(configs=configs,
                        stackups=stackups,
                        postfix='_train')
        
    if hasattr(configs.dataset_generation, 'test'):
        samples_np = make_samples(input_features=input_features, 
                                    method=configs.dataset_generation.test.sampling_method,
                                    sampling_num=configs.dataset_generation.test.sampling_num,
                                    trial_num=configs.dataset_generation.test.trial_num if hasattr(configs.dataset_generation.test, 'trial_num') else 50)
        if samples_np.ndim == 3:
            samples_np = samples_np.reshape([-1, samples_np.shape[-1]])
        stackups = create_stackups_from_samples(input_features=input_features,
                                                samples_np=samples_np,
                                                layer_num=input_features.layer_num,
                                                method=configs.differentline if hasattr(configs, 'differentline') else 'same_line')
        write_to_files(configs=configs,
                        stackups=stackups,
                        postfix='_test')

def generate_metis(configs: Config):
    input_features = MultiLayerInputFeatures(configs.config.dir)
    
    # for variable in input_features.sampled_variables:
        # if not variable in supported_metis_variables:
        #     raise ValueError(f"Variables {variable} not supported.")
        
    stackups = []
    # generate the entire sample spaces, regardless of train/test data seperation
    if configs.dataset_generation.complete:
        meshs = []
        samples_np = make_samples(input_features=input_features, 
                                method='sweep',)
        stackups = create_stackups_from_samples(input_features,
                                                samples_np,
                                                layer_num=input_features.layer_num,
                                                method=configs.differentline if hasattr(configs, 'differentline') else 'same_line')
        write_to_files(configs=configs,
                       stackups=stackups)
    else:
        if hasattr(configs.dataset_generation, 'train'):
            samples_np = make_samples(input_features=input_features, 
                                      method=configs.dataset_generation.train.sampling_method,
                                      sampling_num=configs.dataset_generation.train.sampling_num,
                                      trial_num=configs.dataset_generation.train.trial_num if hasattr(configs.dataset_generation.train, 'trial_num') else 50)
            if samples_np.ndim == 3:
                samples_np = samples_np.reshape([-1, samples_np.shape[-1]])
            stackups = create_stackups_from_samples(input_features=input_features,
                                                    samples_np=samples_np,
                                                    layer_num=input_features.layer_num,
                                                    method=configs.differentline if hasattr(configs, 'differentline') else 'same_line')
            write_to_files(configs=configs,
                           stackups=stackups,
                           postfix='_train')
        if hasattr(configs.dataset_generation, 'test'):
            samples_np = make_samples(input_features=input_features, 
                                      method=configs.dataset_generation.test.sampling_method,
                                      sampling_num=configs.dataset_generation.test.sampling_num,
                                      trial_num=configs.dataset_generation.test.trial_num if hasattr(configs.dataset_generation.test, 'trial_num') else 50)
            if samples_np.ndim == 3:
                samples_np = samples_np.reshape([-1, samples_np.shape[-1]])
            stackups = create_stackups_from_samples(input_features=input_features,
                                                    samples_np=samples_np,
                                                    layer_num=input_features.layer_num,
                                                    method=configs.differentline if hasattr(configs, 'differentline') else 'same_line')
            write_to_files(configs=configs,
                           stackups=stackups,
                           postfix='_test')

def write_to_files(
    configs: Config,
    stackups: List[Stackup],
    postfix: str = '',
):
    stackup_writer = StackupWriter(stackups)
    stackup_writer.write_xlsx(configs, saved_name=f'{configs.dataset_generation.name}{postfix}')
    stackup_writer.to_pickle(configs, saved_name=f'{configs.dataset_generation.name}{postfix}')
    return

if __name__ == '__main__':
    main()

