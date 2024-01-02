from ..datasets import InputFeatures
from .base_sampler import *




class LhsSampler(BaseSampler):
    """
    This implements an lhs sampler.

    Attributes:
        input_features: InputFeatures
        interval_num: int
    
    """
    def __init__(self, 
                 input_features: InputFeatures,
                 interval_num: int) -> None:
        
        super().__init__(input_features)
        self.interval_num = interval_num

    def sample(self, 
               sample_num: int, 
               random_state: str) -> np.ndarray:
        """
        Args:
            sample_num: int
            random_state: str = random | center
        
        Return:
            np.ndarray (sample_num, feature_num)
        """

        if sample_num % self.interval_num != 0:
            raise ValueError("Sampling number must be multiple times of interval number!")
             
        sample_np = np.zeros([sample_num, self.feature_num])
        for i in range(sample_num // self.interval_num):
            # Generate the intervals
            cut = np.linspace(0, 1, self.interval_num + 1)    
            
            # Fill points uniformly in each interval
            if random_state == "random":
                u = np.random.rand(self.interval_num, self.feature_num)
            elif random_state == "center":
                u = np.ones(self.interval_num, self.feature_num) * 0.5
            else:
                raise NotImplementedError

            a = cut[ : self.interval_num]
            b = cut[1 : self.interval_num + 1]
            rdpoints = np.zeros_like(u)

            for j in range(self.feature_num):
                rdpoints[:, j] = u[:, j]*(b-a) + a
            
            # Make the random pairings
            H = np.zeros_like(rdpoints)
            for j in range(self.feature_num):
                order = np.random.permutation(range(self.interval_num))
                H[:, j] = rdpoints[order, j]

            
            for kx in range(self.feature_num):
                H[:, kx] = np.around(self.limits[kx, 0] + H[:, kx] * (self.limits[kx, 1] - self.limits[kx, 0]), decimals=self.decimals[kx])
            
            sample_np[i * self.interval_num : (i + 1) * self.interval_num, :] = H

        return sample_np

