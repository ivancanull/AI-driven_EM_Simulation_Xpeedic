from ..datasets import MultiLayerInputFeatures
from .sample_optimizer import *
from .uniform_sampler import *
from scipy.stats import uniform

class GuidedUniformSampler():
    """
    This implements a guided uniform distribution sampler.

    Attributes:
        input_features: InputFeatures | MultiLayerInputFeatures
    
    """
    def __init__(self, 
                 input_features) -> None:
        
        self.input_features = input_features
        self.sample_optimizer = SampleOptimizer(self.input_features)
    def sample(self, 
               sample_num: int,
               trial_num = 50) -> np.ndarray:
        """
        Args:
            sample_num: int
        
        Return:
            np.ndarray (sample_num, feature_num)
        """

        uniform_sampler = UniformSampler(self.input_features)
        candidate_samples_np = None
        max_uniformity = 0
        for i in range(trial_num):
            sampled_parameters = uniform_sampler.sample(sample_num)
            range_uniformity = self.sample_optimizer.range_uniformity(sampled_parameters)
            if range_uniformity > max_uniformity:
                max_uniformity = range_uniformity
                candidate_samples_np = sampled_parameters


        return candidate_samples_np

    
    # feature 1: select from generated data
    # feature 2: generate independent data
    # feature 3: ignore existed data
    # TODO
