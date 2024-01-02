import pandas as pd
import torch

def analyze_error(
    truth: torch.Tensor,
    prediction: torch.Tensor,
):
    """
    truth shape: (N, freq_num, 2)
    prediction shape: the same
    Analyze error with the following aspects:
    - MSE
    - mean relative error
    - max relative error
    """

    truth_max, _ = torch.max(truth, dim=1, keepdim=True)
    truth_min, _ = torch.min(truth, dim=1, keepdim=True)
    
    truth_range = truth_max - truth_min

    error = torch.abs(truth - prediction)
    
    sr_mean_relative_error = torch.mean(error[..., 0]/ truth_range[..., 0])
    si_mean_relative_error = torch.mean(error[..., 1]/ truth_range[..., 1])
    sr_max_relative_error = torch.max(error[..., 0]/ truth_range[..., 0])
    si_max_relative_error = torch.max(error[..., 1]/ truth_range[..., 1])

    print('SR Mean Error: %.5f, SI Mean Error: %.5f, SR Max Error: %.5f, SI Max Error: %.5f' 
          % (sr_mean_relative_error.item(), si_mean_relative_error.item(), sr_max_relative_error.item(), si_max_relative_error.item()))
