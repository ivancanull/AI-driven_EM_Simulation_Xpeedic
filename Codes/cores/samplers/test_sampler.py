from ..datasets import *
from .base_sampler import *

class TestSampler(BaseSampler):
    """
    This sampler implements generation of samples near the covered training data.

    Attributes:
        input_features: InputFeatures
        train_samples_df: pd.DataFrame of sampled training data  
    """

    def __init__(self,
                 input_features: InputFeatures,
                 train_samples_df: pd.DataFrame) -> None:
        super().__init__(input_features)
        self.train_samples_df = train_samples_df
        # Use the sample() method to randomly select 1000 rows
        

    def sample_m1(self,
                  sample_num: int,
                  train_sample_num: int) -> np.ndarray:

        # Sample num muse be multiples of 1000
        if sample_num % train_sample_num != 0:
            raise ValueError(f'Sample num {sample_num} is not integer multiples of train sample num {train_sample_num}.')

        sampled_df = self.train_samples_df.sample(n=train_sample_num, random_state=42)[self.sampled_variables]
        num_copies = sample_num // train_sample_num

        # Create an index array with custom indices
        new_indices = ["Stripline" + str(i + 1) for i in range(sample_num)]

        # Use pd.concat() to copy the rows and set custom indices
        test_np = np.concatenate([sampled_df.to_numpy()] * num_copies)        

        # Generate a column of random integers (-1, 0, 1)
        
        random_values = np.random.choice([-1, 0, 1], size=[sample_num, self.feature_num])
        random_values = random_values * 1 / 10 ** (np.array(self.decimals, dtype=float).T)
        
        test_np = test_np + random_values

        test_np = np.clip(test_np, self.limits.T[0, :], self.limits.T[1, :])
        for i in range(self.feature_num):
            test_np[:, i] = np.around(test_np[:, i], decimals=self.decimals[i])

        return test_np