from ..datasets import MultiLayerInputFeatures

import numpy as np
from scipy.stats import uniform
import pyDOE2
class UniformSampler():
    """
    This implements a uniform distribution sampler.

    Attributes:
        input_features: InputFeatures
    
    """
    def __init__(self, 
                 input_features) -> None:
        
        self.input_features = input_features

    def sample(self, 
               sample_num: int) -> np.ndarray:
        """
        Args:
            sample_num: int
        
        Return:
            np.ndarray (sample_num, feature_num)
        """

        feature_num = len(self.input_features.sampled_variables)

        # create samples for each variable
        if self.input_features.combination == "Dot":
            sample_np = np.zeros([sample_num, feature_num])
            for i in range(feature_num):
            # define trunc normal sampler
                # uniformly sample from the value list
                values = getattr(self.input_features, self.input_features.sampled_variables[i]).values
                rv = uniform(loc=0, scale=len(values))
                r = rv.rvs(size=sample_num)
                r_int = r.astype(int)
                values_np = np.array(values)
                sample_np[:, i] = values_np[r_int]

        elif self.input_features.combination == "Pair":
            combination_num = self.input_features.get_combination_num()
            sample_np = np.zeros([combination_num, sample_num, feature_num])
            samples_np_list = []
            for i in range(combination_num):
                sub_meshes = []
                for j in range(feature_num):
                    variable_parameter = getattr(self.input_features, self.input_features.sampled_variables[j])
                    if variable_parameter.is_list():
                        values = variable_parameter.values[i]
                    else:
                        values = variable_parameter.values
                    rv = uniform(loc=0, scale=len(values))
                    r = rv.rvs(size=sample_num)
                    r_int = r.astype(int)
                    values_np = np.array(values)
                    sample_np[i, :, j] = values_np[r_int]
        elif self.input_features.combination == "Mesh":
            # TODO
            raise NotImplementedError
        elif self.input_features.combination == "BBDesign":
            # TODO
            # create a box behnken design of experiments
            combination_num = self.input_features.get_combination_num()
            if combination_num != 3:
                raise ValueError(f'For Box-Behnken DOE, the combination number must be 3 instead of {combination_num}.')
            variable_num = len(self.input_features.sampled_variables)
            mat = (pyDOE2.bbdesign(variable_num, center=1) + 1).astype(int)
            sample_np = np.zeros([mat.shape[0], sample_num, feature_num])
            samples_np_list = []
            for i in range(mat.shape[0]):
                sub_meshes = []
                for j in range(feature_num):
                    variable_parameter = getattr(self.input_features, self.input_features.sampled_variables[j])
                    if variable_parameter.is_list():
                        values = variable_parameter.values[mat[i, j]]
                        print(i, j, values)
                    else:
                        values = variable_parameter.values
                    rv = uniform(loc=0, scale=len(values))
                    r = rv.rvs(size=sample_num)
                    r_int = r.astype(int)
                    values_np = np.array(values)
                    sample_np[i, :, j] = values_np[r_int]
        else:
            raise ValueError(f"Combination {self.input_features.combination} not supported.")
        return sample_np
