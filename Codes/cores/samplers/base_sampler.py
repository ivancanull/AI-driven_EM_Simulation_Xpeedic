from ..datasets import *

class BaseSampler:
    """
    This is a base class for samplers.

    Attributes:
        input_features: InputFeatures

    
    """
    def __init__(self, 
                 input_features: InputFeatures) -> None:
        self.input_features = input_features

        # Get feature num
        self.feature_num = self.input_features.variable_num
        # Sample variables within the range
        self.sampled_variables = self.input_features.sampled_variables
        self.limits = self.input_features.limits
        self.decimals = self.input_features.decimals
