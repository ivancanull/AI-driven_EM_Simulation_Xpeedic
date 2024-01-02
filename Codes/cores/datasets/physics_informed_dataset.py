import torch
import math
import numpy as np

from .data_loader import *
from scipy.signal import hilbert

from scipy.optimize import curve_fit

from scipy.interpolate import interp1d, barycentric_interpolate
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from typing import List

class PhysicsInformedDataset(CustomDataset):
    def __init__(self, df, input_cols, output_cols, nF, indices, config_file, device):
        super().__init__(df, input_cols, output_cols, nF, indices, config_file, device)
        self.DC_value = None
        self.output_ifft = None
        self.__mode = "DC"

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode):
        self.__mode = mode

    def get_DC_value(self):
        self.DC_value = torch.mean(self.output, dim=-1, keepdim=False)
        return self.DC_value
    
    def get_ifft(self):
        output_complex = self.output[:, 0, :] + self.output[:, 1, :] * 1j
        self.output_ifft = torch.fft.irfft(output_complex)
        return self.output_ifft

    def get_ifft_pulse_location(self):
        length = self.output_ifft.shape[-1]
        start = int(0.05 * length)
        end = int(0.9 * length)
        _, location = torch.max(torch.abs(self.output_ifft[:, start:end]), dim=1, keepdim=True)
        location = location + start
        # location = torch.cat((torch.range(0, self.output_ifft.shape[0]), location), dim=1)
        value = torch.take_along_dim(self.output_ifft, location, 1)
        positive = value / torch.sqrt(value ** 2)
        self.length = length
        self.output_ifft_pulse = torch.cat((location, value), dim=1)
        return self.output_ifft_pulse

    # Derive the peak location and the values around it from -7 to 7, 15 points in total
    def get_ifft_pulse_approaximation(self):
        length = self.output_ifft.shape[-1]
        start = int(0.05 * length)
        end = int(0.9 * length)
        _, location = torch.max(torch.abs(self.output_ifft[:, start:end]), dim=1, keepdim=True)
        location_center = location + start
        location = torch.arange(-7, 8).int().to(self.output_ifft.device) + location_center
        value = torch.take_along_dim(self.output_ifft, location, 1)
        self.length = length
        self.output_ifft_pulse = torch.cat((location_center, value), dim=1)
        return self.output_ifft_pulse
    
    def get_ifft_pulse_curve_fitting(self):

        def normal_distribution_func(x, a):
            return np.exp(- x ** 2 / a)

        peak = self.output_ifft_pulse[:, 8:9].cpu().numpy()
        value = self.output_ifft_pulse[:, 1:].cpu().numpy() / peak

        a = []
        x = np.arange(-7, 8).astype(int)
        for i in range(peak.shape[0]):
            popt, pconv = curve_fit(normal_distribution_func, x, value[i])
            a.append(popt[0])

        # plot
        # fig, ax = plt.subplots(5, 1, figsize=(16, 5 * 5), constrained_layout=True)
        # for i in range(5):
        #     ax[i].plot(x, value[i], 'r--')
        #     ax[i].plot(x, normal_distribution_func(x, a[i]), 'g--')
        # plt.savefig('../test_normal_distribution.png')

        a = torch.Tensor(a).reshape((self.output_ifft.shape[0], 1))
        self.output_ifft_pulse_a = a.to(self.output_ifft.device)
        return


    def get_ifft_pulse(self, pulse_start=None, pulse_end=None):
        dataset_num = self.output_ifft.shape[0]
        length = self.output_ifft.shape[-1]

        # intuatively set

        if pulse_start is None or pulse_end is None:
            threshold = 3e-4
            masked_length = int(0.3 * length)

            start = int(0.05 * length)
            end = int(0.9 * length)
            _, location = torch.max(torch.abs(self.output_ifft[:, start:end]), dim=1, keepdim=True)

            pulse_start = start
            pulse_end = int(torch.max(location).item()) + start
        
        self.pulse_start = pulse_start
        self.pulse_end = pulse_end

        return pulse_start, pulse_end, self.output_ifft[:, pulse_start: pulse_end]
        

        # # Number 1: include the first value
        # location = torch.range(0, length).reshape(1, -1).to(self.output_ifft.device).repeat(dataset_num, 1)

        # mask = torch.gt(torch.abs(self.output_ifft[0: 10, start:end]), threshold) 
        # mask_nonzero = torch.nonzero(mask, as_tuple=True)
        # min_location = torch.min(mask_nonzero[1])
        # max_location = torch.max(mask_nonzero[1])
        
        # print(min_location, max_location)
        
        
        # indices = torch.masked_fill(torch.cumsum(mask.int(), dim=1), ~mask, 0)
        # print(indices)
        # masked_location = torch.scatter(input=torch.ones_like(location), dim=1, index=indices, src=location)[:,1:]
        # masked_values = torch.scatter(input=torch.zeros_like(self.output_ifft), dim=1, index=indices, src=self.output_ifft)[:,1:]

        # masked_location = masked_location[:, 0: masked_length]
        # masked_values = masked_values[:, 0: masked_length]
        # print(masked_location.shape)
        # print(masked_location[0])
        # print(masked_values[0])
        # return

    def set_max_min_ifft_pulse(self, max=None, min=None):
        if max is None:
            self.ifft_pulse_max = torch.max(self.output_ifft_pulse[:,0])[0]
        else:
            self.ifft_pulse_max = max
        if min is None:
            self.ifft_pulse_min = torch.min(self.output_ifft_pulse[:,0])[0]
        else:
            self.ifft_pulse_min = min

    def __getitem__(self, idx):
        if self.__mode == "dc":
            return self.parameters[idx, ...], self.DC_value[idx, ...]
        elif self.__mode == "fft_0":
            return self.parameters[idx, ...], self.output_ifft[idx, 0:1]
        elif self.__mode == "fft_rest":
            return self.parameters[idx, ...], self.output_ifft[idx, 1:] * 15
        elif self.__mode == "fft":
            return self.parameters[idx, ...], self.output_ifft[idx, ...]
        elif self.__mode == "pulse_location":
            return self.parameters[idx, ...], self.output_ifft_pulse[idx, 0:1] / self.length
        elif self.__mode == "pulse_peak":
            return self.parameters[idx, ...], self.output_ifft_pulse[idx, 1:]
        elif self.__mode == "pulse_peak_max":
            return self.parameters[idx, ...], self.output_ifft_pulse[idx, 8:9]
        elif self.__mode == "pulse_peak_a":
            return self.parameters[idx, ...], self.output_ifft_pulse_a[idx, ...]
        elif self.__mode == "pulse":
            return self.parameters[idx, ...], self.output_ifft[idx, self.pulse_start: self.pulse_end] * 1e2
        else:
            raise NotImplementedError

def get_envelop(signal: torch.Tensor) -> List[np.ndarray]:

    # Define numbers and channels
    # Subtract the dc value
    DC_value = torch.mean(signal, dim=-1, keepdim=True)
    # signal = signal - DC_value
    
    signal = signal.detach().cpu().numpy()        
    n = signal.shape[0]
    c = signal.shape[1]
    x = np.arange(signal.shape[2])
    # Switch signal to numpy and create the output envelop array
    
    envelop_signal = [np.zeros_like(signal), np.zeros_like(signal)]

    for i in range(n):
        for j in range(c):

            ### Method 1 ###
            # Use hilbert transform
            # analytic_signal = hilbert(signal[i, j] - DC_value[i, j])
            # envelop_signal[i, j] = np.abs(analytic_signal) + DC_value[i, j, 0].detach().cpu().item()

            ### Method 2 ###
            s = signal[i, j]
            # locals min     
            lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
            # locals max
            lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 

            # global min of dmin-chunks of locals min 
            lmin = lmin[[i+np.argmin(s[lmin[i:i+1]]) for i in range(0,len(lmin),1)]]
            # global max of dmax-chunks of locals max 
            lmax = lmax[[i+np.argmax(s[lmax[i:i+1]]) for i in range(0,len(lmax),1)]]

            f_min = interp1d(lmin, s[lmin], fill_value='extrapolate')
            envelop_signal[0][i, j] = f_min(x)
            f_max = interp1d(lmax, s[lmax], fill_value='extrapolate')
            envelop_signal[1][i, j] = f_max(x)

            # envelop_signal[0][i, j] = barycentric_interpolate(lmin, s[lmin], x)
            # envelop_signal[1][i, j] = barycentric_interpolate(lmax, s[lmax], x)
    return envelop_signal

def get_hilbert(signal: torch.Tensor) -> List[np.ndarray]:
    signal = signal.detach().cpu().numpy()        
    n = signal.shape[0]
    c = signal.shape[1]
    x = np.arange(signal.shape[2])
    # Switch signal to numpy and create the output envelop array
    
    envelop_signal = np.zeros_like(signal)

    for i in range(n):
        for j in range(c):
            analytic_signal = hilbert(signal[i, j])
            envelop_signal[i, j] = analytic_signal
    return envelop_signal

def get_ifft(signal: torch.Tensor) -> np.ndarray:
    signal = signal.detach().cpu()
    signal_complex = signal[:, 0, :] + signal[:, 1, :] * 1j
    signal_ifft = torch.fft.irfft(signal_complex)
    return signal_ifft.numpy()

def get_fft(signal: np.ndarray) -> np.ndarray:
    signal_fft = np.fft.rfft(signal)
    return signal_fft
