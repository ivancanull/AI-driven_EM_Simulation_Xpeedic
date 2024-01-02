import torch
from torch import Tensor

class LossLogger():
    # current loss
    # update loss
    def __init__(self, range=1.0):
        self._range = range
        self.clean()
    
    def clean(self):
        self._loss = 0.0
        self._total_loss = 0.0
        self._total_size = 0
        self._total_mean_error = 0.0
        self._mean_error = 0.0
        self._max_error = 0.0
    
    def update_error(
        self, 
        loss: torch.Tensor,
        truth: torch.Tensor,
        prediction: torch.Tensor,
        batch_size: int,
    ):
        error = torch.abs(truth - prediction)
        mean_relative_error = torch.mean(error / self._range).item()
        max_relative_error = torch.max(error / self._range).item()
        self._total_size += batch_size
        self._total_loss += loss.item() * batch_size
        self._total_mean_error += mean_relative_error * batch_size
        self._max_error = max(self._max_error, max_relative_error)
    
    @property
    def mean_error(self):
        self._mean_error = self._total_mean_error / self._total_size
        return self._mean_error
    
    @property
    def max_error(self):
        return self._max_error
    
    @property
    def loss(self):
        self._loss = self._total_loss / self._total_size
        return self._loss

class PhysicsLossLogger(LossLogger):

    def __init__(self, mode):
        super().__init__()
        self._sr_mean_error = 0.0
        self._si_mean_error = 0.0
        self._mean_error = 0.0
        self._total_sr_mean_error = 0.0
        self._total_si_mean_error = 0.0
        self._total_mean_error = 0.0
        self._sr_max_error = 0.0
        self._si_max_error = 0.0
        self._max_error = 0.0
        self.mode = mode
    
    def clean(self):
        super().clean()
        self._sr_mean_error = 0.0
        self._si_mean_error = 0.0
        self._mean_error = 0.0
        self._total_sr_mean_error = 0.0
        self._total_si_mean_error = 0.0
        self._total_mean_error = 0.0
        self._sr_max_error = 0.0
        self._si_max_error = 0.0
        self._max_error = 0.0

    def update_error(
        self, 
        truth: torch.Tensor,
        prediction: torch.Tensor,
        batch_size: int
    ):
        error = torch.abs(truth - prediction)

        if self.mode == 'dc':
            sr_mean_absolute_error = torch.mean(error[:, 0]).item()
            si_mean_absolute_error = torch.mean(error[:, 1]).item()
            sr_max_absolute_error = torch.max(error[:, 0]).item()
            si_max_absolute_error = torch.max(error[:, 1]).item()
            self._total_sr_mean_error += sr_mean_absolute_error * batch_size
            self._total_si_mean_error += si_mean_absolute_error * batch_size
            self._sr_max_error = max(self._sr_max_error, sr_max_absolute_error)
            self._si_max_error = max(self._si_max_error, si_max_absolute_error)
        elif 'fft' in self.mode:
            mean_absolute_error = torch.mean(error).item()
            max_absolute_error = torch.max(error).item()
            self._total_mean_error += mean_absolute_error * batch_size
            self._max_error = max(self._max_error, max_absolute_error)
        elif self.mode == 'pulse_location':
            mean_absolute_error = torch.mean(error).item() 
            max_absolute_error = torch.max(error).item()
            self._total_mean_error += mean_absolute_error * batch_size
            self._max_error = max(self._max_error, max_absolute_error) 
        elif 'pulse_peak' in self.mode:
            mean_absolute_error = torch.mean(error).item() 
            max_absolute_error = torch.max(error).item()
            self._total_mean_error += mean_absolute_error * batch_size
            self._max_error = max(self._max_error, max_absolute_error)
        elif self.mode == 'pulse':
            mean_absolute_error = torch.mean(error).item() 
            max_absolute_error = torch.max(error).item()
            self._total_mean_error += mean_absolute_error * batch_size
            self._max_error = max(self._max_error, max_absolute_error)
    @property
    def error(self):
        if self.mode == 'dc':
            self._sr_mean_error = self._total_sr_mean_error / self._total_size
            self._si_mean_error = self._total_si_mean_error / self._total_size
            return self._sr_mean_error, self._si_mean_error, self._sr_max_error, self._si_max_error
        elif 'fft' in self.mode:
            self._mean_error = self._total_mean_error / self._total_size
            return self._mean_error, self._max_error
        elif self.mode == 'pulse_location':
            self._mean_error = self._total_mean_error / self._total_size
            return self._mean_error, self._max_error
        elif 'pulse_peak' in self.mode:
            self._mean_error = self._total_mean_error / self._total_size
            return self._mean_error, self._max_error
        elif self.mode == 'pulse':
            self._mean_error = self._total_mean_error / self._total_size
            return self._mean_error, self._max_error

class ValLossLogger(LossLogger):

    def __init__(self, range):
        super().__init__()
    
    def clean(self):
        super().clean()

    def update_error(
        self, 
        truth: torch.Tensor,
        prediction: torch.Tensor,
        batch_size: int
    ):
        truth_max, _ = torch.max(truth, dim=-1, keepdim=True)
        truth_min, _ = torch.min(truth, dim=-1, keepdim=True)
        truth_range = truth_max - truth_min
        error = torch.abs(truth - prediction)

        sr_mean_relative_error = torch.mean(error[:, 0, :]/ truth_range[:, 0, :]).item()
        si_mean_relative_error = torch.mean(error[:, 1, :]/ truth_range[:, 1, :]).item()
        sr_max_relative_error = torch.max(error[:, 0, :]/ truth_range[:, 0, :]).item()
        si_max_relative_error = torch.max(error[:, 1, :]/ truth_range[:, 1, :]).item()

        self._total_sr_mean_error += sr_mean_relative_error * batch_size
        self._total_si_mean_error += si_mean_relative_error * batch_size
        self._sr_max_error = max(self._sr_max_error, sr_max_relative_error)
        self._si_max_error = max(self._si_max_error, si_max_relative_error)
    
    @property
    def error(self):
        self._sr_mean_error = self._total_sr_mean_error / self._total_size
        self._si_mean_error = self._total_si_mean_error / self._total_size
        return self._sr_mean_error, self._si_mean_error, self._sr_max_error, self._si_max_error

class CustomMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.sqrt(torch.mean((output - target) ** 2, dim=-1)))
