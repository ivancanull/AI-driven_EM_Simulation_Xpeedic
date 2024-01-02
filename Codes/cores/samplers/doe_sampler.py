from ..datasets import MultiLineInputFeatures
from .base_sampler import *

class DoESampler(BaseSampler):
    """
    This is a base calss for sample DoE cases.
    
    """

    def __init__(self, 
                 input_features: MultiLineInputFeatures) -> None:
        
        super().__init__(input_features)
        self.nominals = self.input_features.variable_nominals

    def sample_WS(self,
                  step: int = 10) -> np.ndarray:
        
        steps_np = np.linspace(0.0, 1.0, num=step+1)

        # Fix the central W
        samples_np = np.zeros(((step+1) ** 2, self.feature_num)) 

        for i, v in enumerate(self.sampled_variables):
            if v == 'W_0' or v == f'W_{self.input_features.nline - 1}':
                r = np.repeat(steps_np, [step+1])
                r = r * (self.limits[i, 1] - self.limits[i, 0]) + self.limits[i, 0]
                samples_np[:, i]  = np.around(r, decimals=self.decimals[i])
            elif v == 'S_0' or v == f'S_{self.input_features.nline - 2}':
                r = np.tile(steps_np, [step+1])
                r = r * (self.limits[i, 1] - self.limits[i, 0]) + self.limits[i, 0]
                samples_np[:, i]  = np.around(r, decimals=self.decimals[i])
            else:
                samples_np[:, i] = self.input_features.variable_nominals[i]
        
        return samples_np