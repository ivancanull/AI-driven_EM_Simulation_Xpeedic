from ..datasets import InputFeatures
from .base_sampler import *
from scipy.stats import truncnorm

__all__ = ['NormalSampler']
class NormalSampler(BaseSampler):
    """
    This implements a normal distribution sampler.

    Attributes:
        input_features: InputFeatures
    
    """
    def __init__(self, 
                 input_features: InputFeatures) -> None:
        
        super().__init__(input_features)
        self.nominals = self.input_features.variable_nominals

    def sample(self, 
               sample_num: int,) -> np.ndarray:
        """
        Args:
            sample_num: int
        
        Return:
            np.ndarray (sample_num, feature_num)
        """

        sample_np = np.zeros([sample_num, self.feature_num])

        # Create samples for each variable
        for i in range(self.feature_num):
            
            # Define trunc normal sampler
            start = self.limits[i, 0]
            nominal = self.nominals[i]
            end = self.limits[i, 1]
            
            # 3-sigma
            sd = max(nominal - start, end - nominal) / 3.0
            rv = get_truncated_normal(mean=nominal, sd=sd, low=start, upp=end)

            r = rv.rvs(size=sample_num)
            r = np.around(r, decimals=self.decimals[i])
            sample_np[:, i] = r
        
        return sample_np

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

