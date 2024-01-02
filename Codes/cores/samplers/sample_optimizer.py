import numpy as np
from torch import norm
from scipy.spatial.distance import pdist

from ..datasets.multilayer_input_features import MultiLayerInputFeatures
class SampleOptimizer():
    def __init__(self,
                 input_features: MultiLayerInputFeatures):
        self.input_features = input_features

    def range_uniformity(self, samples: np.ndarray) -> float:
        # return the sum of minimum occurance of every feature
        min_ratio = 0
        for idx, variable in enumerate(self.input_features.sampled_variables):
            unique, counts = np.unique(samples[:, idx], return_counts=True)
            min_ratio += (np.min(counts) / samples.shape[0]) * len(getattr(self.input_features, variable).values)  
        return min_ratio / len(self.input_features.sampled_variables)

    # not helpful
    def distance_uniformity(self, samples: np.ndarray) -> float:
        norm_samples = samples.copy()
        for idx, variable in enumerate(self.input_features.sampled_variables):
            # normalize samples
            v_min = getattr(self.input_features, variable).min
            v_max = getattr(self.input_features, variable).max
            norm_samples[:, idx] = (norm_samples[:, idx] - v_min) / (v_max - v_min)
        # calculate distances between every two rows in norm_samples
        norm_samples_dists = pdist(norm_samples, 'euclid').min()
        return norm_samples_dists
        

