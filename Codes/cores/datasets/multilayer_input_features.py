import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from openpyxl import load_workbook
from torch import combinations

from .input_features import InputFeatures
from utils import Config

class MultiLayerParameters():
    """
    A class to represent the parameters of a multilayer system. If the parameter is a list of values, then the attributes are lists of the corresponding values.

    Attributes:
    -----------
    min : float
        The minimum value of the parameter.
    nominal : float
        The nominal value of the parameter. 
    max : float
        The maximum value of the parameter.
    step : float
        The step size between values of the parameter.
    values : list
        A list of all possible values of the parameter between min and max with step size of step.
    """

    def __init__(
        self,
        key,
        value
    ):
        def create_values(min, nominal, max, step, endpoint=False):
            if step > 0:
                if endpoint:
                    n = round((max - min) / step) + 1
                else:
                    n = round((max - min) / step)
                return np.linspace(min, max, n, endpoint=endpoint).tolist()
            else:
                raise ValueError('The step size must be positive.')
            
        if isinstance(value, list):
            self.min = [vi['min'] for vi in value]
            self.nominal = [vi['nominal'] for vi in value]
            self.max = [vi['max'] for vi in value]
            self.step = []
            self.values = []
            for vi in value:
                if 'step' in vi:
                    self.step.append(vi['step'])
                    if 'endpoint' in vi:
                        endpoint = vi['endpoint']
                    else:
                        endpoint = False
                    self.values.append(create_values(vi['min'], vi['nominal'], vi['max'], vi['step'], endpoint=endpoint))
                else:
                    self.step.append(0)
                    self.values.append([vi['nominal']])
        else:
            if 'standard' in value:
                self.standard = value['standard']
            else:
                self.standard = None
                self.min = value['min']
                self.nominal = value['nominal']
                self.max = value['max']
                if 'step' in value:
                    if 'endpoint' in value:
                        self.endpoint = value['endpoint']
                    else:
                        self.endpoint = False
                    self.step = value['step']
                    self.values = create_values(self.min, self.nominal, self.max, self.step, self.endpoint)
                else:
                    self.step = 0
                    self.values = [self.nominal]
    
    def is_list(self):
        return isinstance(self.min, list)
    
    def __len__(self):
        if self.is_list():
            return len(self.min)
        else:
            return 1
        
class Line():
    """
    A class representing a line in a multilayer transmission line.

    Attributes:
    -----------
    w : float
        The width of the line.
    signal : bool, optional
        Whether this line is a signal line. Default is False.
    """
    def __init__(
            self, 
        w: float,
        signal: bool=False
    ):
        self.w = w
        self.signal = signal

    @property
    def x_coord(self):
        return self._x_coord
    
    @x_coord.setter
    def x_coord(self, x):
        self._x_coord = x


class Layer():
    """
    A class representing a single layer in a multilayer structure.

    Attributes:
    ----------
    thickness : float
        The thickness of the layer in micrometers.
    conductivity : float
        The electrical conductivity of the layer in Siemens per meters.
    er : float
        The relative permittivity of the layer.
    loss : float
        The loss tangent of the layer.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):  
        if len(args) == 1:
            if not isinstance(args[0], MultiLayerInputFeatures) :
                raise ValueError('The input argument is not a MultiLayerInputFeatures object.')
            thickness = args[0].T.nominal
            conductivity = args[0].Conductivity.nominal
            er = args[0].Er.nominal
            loss = args[0].Loss.nominal
        elif len(args) != 0:
            raise ValueError('The number of input arguments is not correct.')
        
        if 'nominal' in kwargs:
            thickness = kwargs['nominal'].T.nominal
            conductivity = kwargs['nominal'].Conductivity.nominal
            er = kwargs['nominal'].Er.nominal
            loss = kwargs['nominal'].Loss.nominal
        if 'thickness' in kwargs:
            thickness = kwargs['thickness']
        if 'conductivity' in kwargs:
            conductivity = kwargs['conductivity']
        if 'er' in kwargs:
            er = kwargs['er']
        if 'loss' in kwargs:
            loss = kwargs['loss']

        self.thickness = thickness
        self.conductivity = conductivity
        self.er = er
        self.loss = loss

        self.length = 0

class SignalLayer(Layer):
    """
    A class representing a signal layer in a multilayer input feature.

    Attributes:
    - n_line: The number of lines in the signal layer.
    - lines: A list of Line objects representing the lines in the signal layer.
    - spaces: A list of spacings between the lines in the signal layer.
    - length: The total length of the signal layer.
    - w_list: A list of widths of the lines in the signal layer.

    Methods:
    - __init__: Initializes a SignalLayer object.
        Input arguments:
            1.  nominal: A MultiLayerInputFeatures object.
            2.  w (optional): A float representing the width of the lines in the signal layer.
                s (optional): A float representing the spacing between the lines in the signal layer.
                pattern (optional): A string representing the pattern of the lines in the signal layer. 'G' represents a ground line and 'S' represents a signal line.
    """

    def __init__(
        self,
        *args,
        **kwargs):

        super().__init__(
            *args,
            **kwargs,
        )

        if len(args) == 1:
            if args[0] is not MultiLayerInputFeatures:
                raise ValueError('The input argument is not a MultiLayerInputFeatures object.')
            w = args[0].W.nominal
            s = args[0].S.nominal
        elif len(args) != 0:
            raise ValueError('The number of input arguments is not correct.')
        
        if 'nominal' in kwargs:
            if hasattr(kwargs['nominal'], 'W') and hasattr(kwargs['nominal'], 'S'):
                w = kwargs['nominal'].W.nominal
                s = kwargs['nominal'].S.nominal
            # specify W and S for each layer
            elif 'w' not in kwargs or 's' not in kwargs:
                raise ValueError('For layers with different W/S settings, W/S for each layer must be specified.')

        if 'w' in kwargs:
            w = kwargs['w']
        if 's' in kwargs:
            s = kwargs['s']

        if 'pattern' in kwargs:
            self.pattern = kwargs['pattern']
        else:
            self.pattern = 'GGSGG'
        self.n_line = len(self.pattern)
        
        
        
        # create lists of widths and spacings
        if not isinstance(w, list):
            w_list = [w] * self.n_line
        else:
            w_list = w
        if not isinstance(s, list):
            s_list = [s] * (self.n_line - 1)
        else:
            s_list = s

        if self.n_line != len(w_list):
            raise ValueError('The number of lines is not equal to the number of widths.')
        if self.n_line != (len(s_list) + 1):
            raise ValueError('The number of lines is not equal to the number of spacings.')
        if self.n_line != len(self.pattern):
            raise ValueError('The number of lines is not equal to the number of patterns.')
        # generate patterns
        self._is_signal = [self.pattern[i] == 'S' for i in range(self.n_line)]
        
        self.lines = [Line(w_list[i], self._is_signal[i]) for i in range(self.n_line)]
        self.spaces = s_list
        self.length = sum(w_list) + sum(s_list)

        # calculate the x coordinates of the lines
        x = 0
        self.lines[0].x_coord = x
        for i in range(self.n_line - 1):
            x += self.lines[i].w
            x += self.spaces[i]
            self.lines[i+1].x_coord = x

    @property
    def w_list(self):
        """
        Returns a list of widths of the lines in the signal layer.
        """
        return [line.w for line in self.lines]
            
class AirLayer(Layer):
    """
    A class representing an air layer in a multilayer input features object.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If the number of input arguments is not correct or if the input argument is not a MultiLayerInputFeatures object.

    Attributes:
        thickness (float): The thickness of the air layer.
        conductivity (float): The conductivity of the air layer.
        er (float): The relative permittivity of the air layer.
        loss (float): The loss tangent of the air layer.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if len(args) == 1:
            if args[0] is not MultiLayerInputFeatures:
                raise ValueError('The input argument is not a MultiLayerInputFeatures object.')
            thickness = args[0].T.nominal
        elif len(args) != 0:
            raise ValueError('The number of input arguments is not correct.')
        
        if 'nominal' in kwargs:
            thickness = kwargs['nominal'].T.nominal
        if 'thickness' in kwargs:
            thickness = kwargs['thickness']
    
        super().__init__(
            thickness=thickness,
            conductivity=0,
            er=1,
            loss=0
        )

class DieLayer(Layer):
    """
    A class representing a dielectric layer in a multilayer structure.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If the number of input arguments is not correct or the input argument is not a MultiLayerInputFeatures object.

    Attributes:
        thickness (float): The thickness of the dielectric layer.
        conductivity (float): The conductivity of the dielectric layer.
        er (float): The relative permittivity of the dielectric layer.
        loss (float): The loss tangent of the dielectric layer.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if len(args) == 1:
            if args[0] is not MultiLayerInputFeatures:
                raise ValueError('The input argument is not a MultiLayerInputFeatures object.')
            thickness = args[0].H.nominal
            er = args[0].Er.nominal
        elif len(args) != 0:
            raise ValueError('The number of input arguments is not correct.')
        
        if 'nominal' in kwargs:
            thickness = kwargs['nominal'].H.nominal
            er = kwargs['nominal'].Er.nominal
        if 'thickness' in kwargs:
            thickness = kwargs['thickness']
        if 'er' in kwargs:
            er = kwargs['er']

        super().__init__(
            thickness=thickness,
            conductivity=0,
            er=er,
            loss=0,
        )

class GroundLayer(Layer):
    """
    A class representing the ground layer in a multilayer input feature dataset.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        None

    Methods:
        __init__: Initializes the GroundLayer object.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Stackup():
    def __init__(
        self,
        layers: list = None,
    ):
        self.layers = layers

    def append(self, layer: Layer):
        if self.layers == [] and layer != AirLayer:
            raise ValueError('The first layer must be an air layer.')
        self.layers.append(Layer)

    def extend(self, layers: List[Layer]):
        if self.layers == [] and layers[0] != AirLayer:
            raise ValueError('The first layer must be an air layer.')
        for layer in layers:
            self.append(layer)
    
    def center_alignment(self):
        for layer in self.layers:
            if isinstance(layer, SignalLayer):
                indentation = (self.length - layer.length) / 2
                for line in layer.lines:
                    line.x_coord = line.x_coord + indentation

    def plot(self, save_dir=None, save_name=None, figsize=(10, 10), dpi=300):
        self.center_alignment()
        fig, ax = plt.subplots(dpi=dpi)
        ax.set_aspect(1)
        y = 0
        for layer in reversed(self.layers):
            
            if isinstance(layer, AirLayer):
                print('layer is air')
                rect = plt.Rectangle((0, y), self.length, layer.thickness, color='white')
                ax.add_patch(rect)
            elif isinstance(layer, DieLayer):
                print('layer is die')
                rect = plt.Rectangle((0, y), self.length, layer.thickness, color='green')
                ax.add_patch(rect)
            elif isinstance(layer, GroundLayer):
                print('layer is ground')
                rect = plt.Rectangle((0, y), self.length, layer.thickness, color='gold')
                ax.add_patch(rect)
            elif isinstance(layer, SignalLayer):
                print('layer is signal')
                rect = plt.Rectangle((0, y), self.length, layer.thickness, color='lightgreen')
                ax.add_patch(rect)
                for line in layer.lines:
                    rect = plt.Rectangle((line.x_coord, y), line.w, layer.thickness, color='red' if line.signal else 'gold')
                    ax.add_patch(rect)
            else:
                raise ValueError('The layer is not a valid layer.')
            
            y += layer.thickness
        plt.xlim(0, self.length)
        plt.ylim(0, y)
        plt.axis('off')
        if save_dir is not None and save_name is not None:
            plt.savefig(save_dir + f'{save_name}.png', dpi=dpi)
            plt.savefig(save_dir + f'{save_name}.pdf', dpi=dpi)

    @property
    def length(self):
        return max(self.layers, key=lambda x: x.length).length

    def thickness(self):
        return sum([layer.thickness for layer in self.layers])

    def __getitem__(self, index):
        return self.layers[index]
    
    def __len__(self):
        return len(self.layers)

class MultiLayerInputFeatures(InputFeatures):
    def __init__(
        self,
        config_dir
    ):
        with open(config_dir, encoding="utf-8") as f:
            config = json.load(f)

        self.frequency = config["Frequency"]
        self.layer_num = config["Layer"] 
        self.frequency_np, self.nF = self.get_frequency()
        for key, value in config["Parameters"].items():
            setattr(self, key, MultiLayerParameters(key, value))
        self.sampled_variables = config["Variables"]
        self.pattern = config["Pattern"]
        # support mode: Dot, Pair, Mesh
        self.differentlayer = config["DifferentLayer"]
        self.combination = config["Combination"]
        if self.combination not in ['Dot', 'Pair', 'Mesh', 'BBDesign']:
            raise ValueError('The combination mode is not supported.')

    def get_frequency(self):
        """
        Convert the frequency range specified in the input features to a numpy array of frequencies and the number of frequencies.

        Returns:
        frequency_np (numpy.ndarray): Array of frequencies.
        nF (int): Number of frequencies.
        """
        def convert_Hz_to_num(hz: str):
            if hz == "Hz":
                return 1
            elif hz == "KHz":
                return 1e3
            elif hz == "MHz":
                return 1e6
            elif hz == "GHz":
                return 1e9
            else:
                raise ValueError('hz unit not correct')        
        f_start = self.frequency[0] * convert_Hz_to_num(self.frequency[1])
        f_end = self.frequency[2] * convert_Hz_to_num(self.frequency[3])
        f_step = self.frequency[4] * convert_Hz_to_num(self.frequency[5])
        frequency_np = np.arange(f_start, f_end + f_step, f_step)
        nF = len(frequency_np)
        return frequency_np, nF

    def get_combination_num(self):
        if self.combination == "Pair" or self.combination == "BBDesign":
            combination_num = []
            for variable in self.sampled_variables:
                variable_parameter = getattr(self, variable)
                if variable_parameter.is_list():
                    combination_num.append(len(variable_parameter))
            # check if all the elements in the combination num list are the same
            if combination_num.count(combination_num[0]) != len(combination_num):
                raise ValueError('All parameters should have the same nominal numbers under the Pair mode.')
            else:
                return combination_num[0]
        else:
            return None
        
class StackupWriter():
    def __init__(
        self, 
        stackups
    ) -> None:
        if isinstance(stackups, str):
            self.stackups = pickle.load(open(stackups, 'rb'))
        elif isinstance(stackups, list):
            self.stackups = stackups
        else:
            raise ValueError('The input argument is not a list or a directory of saved stackps.')
            
        self.check_stackups()

        self.layer_names = self.layer_name()
        self.stackup_df, self.index_num = self.stackup_dataframe()
        self.parameters_df_list, self.start_row = self.parameter_dataframe_list()

    def layer_name(self):
        layer_names = []
        air_i = 1
        metal_i = 1
        die_i = 1
        for layer in self.stackups[0]:
            if isinstance(layer, AirLayer):
                layer_names.append(f'Air{air_i}')
            elif isinstance(layer, DieLayer):
                layer_names.append(f'Die{die_i}')
                die_i += 1
            elif isinstance(layer, SignalLayer) or isinstance(layer, GroundLayer):
                layer_names.append(f'Metal{metal_i}')
                metal_i += 1
            else:
                raise ValueError('The layer is not a valid layer.')
        return layer_names
    
    def check_stackups(self):
        for stackup in self.stackups:
            if not isinstance(stackup, Stackup):
                raise ValueError('The input argument is not a Stackup object.')
            if not isinstance(stackup[0], AirLayer):
                raise ValueError('The first layer of the stackup must be an air layer.')
            if not isinstance(stackup[-1], AirLayer):
                raise ValueError('The last layer of the stackup must be an air layer.')
            if not isinstance(stackup[1], GroundLayer):
                raise ValueError('The second layer of the stackup must be a ground layer.')
            if not isinstance(stackup[-2], GroundLayer):
                raise ValueError('The second to last layer of the stackup must be a ground layer.')
        # check all stackups have the same number of layers
        n_layer = len(self.stackups[0])
        for stackup in self.stackups:
            if len(stackup) != n_layer:
                raise ValueError('All stackups must have the same number of layers.')
    
    def stackup_dataframe(self):
        self.check_stackups()
        df = pd.DataFrame(columns=['Layer Name', 'Thickness', 'Conductivity', 'Dieletric Er', 'Dieletric Loss', 'Parameter Index', 'Ref Layer'])
        
        # Define the columns of the dataframe
        # Layer Name	Thickness	Conductivity	Dieletric Er	Dieletric Loss	Parameter Index	Ref Layer
        # add die layer to write
        metal_index = 2
        for i, layer in enumerate(self.stackups[0].layers):
            if isinstance(layer, AirLayer):
                parameter_index = None
                ref_layer = None
            elif isinstance(layer, GroundLayer):
                parameter_index = 1
                ref_layer = 1
            elif isinstance(layer, DieLayer):
                parameter_index = metal_index
                ref_layer = None
                metal_index += 1
            elif isinstance(layer, SignalLayer):
                parameter_index = metal_index
                ref_layer = None
                metal_index += 1
            else:
                raise ValueError('The layer is not a valid layer.')
            
            df = df.append({
                'Layer Name': self.layer_names[i],
                'Thickness': layer.thickness,
                'Conductivity': layer.conductivity,
                'Dieletric Er': layer.er,
                'Dieletric Loss': layer.loss,
                'Parameter Index': parameter_index,
                'Ref Layer': ref_layer
            }, ignore_index=True)

        index_num = metal_index - 1
        return df, index_num
    
    def parameter_dataframe_list(self):
        self.check_stackups()
        df_list = [pd.DataFrame() for _ in range(self.index_num)]
        
        # Define the columns of the dataframe
        # Pattern   W   S   Thickness   Con Er  Loss
        for stackup in self.stackups:
            # idx is the index of the dataframe in df_list
            idx = 0
            # start_row is the start row of the dataframe in the excel file
            start_row = [0]
            # ground_written and die_written are flags to indicate whether the ground layer and dielectric layer have been written
            ground_written = False
            # die_written = False
            for layer in stackup:
                if (isinstance(layer, GroundLayer) and not ground_written) or (isinstance(layer, DieLayer)):
                    df_list[idx] = df_list[idx].append({
                        'Thickness': layer.thickness,
                        'Conductivity': layer.conductivity,
                        'Er': layer.er,
                        'Loss': layer.loss
                    }, ignore_index=True)
                    start_row.append(start_row[-1] + 5)
                    idx += 1
                    if isinstance(layer, GroundLayer):
                        ground_written = True
                elif isinstance(layer, SignalLayer):
                    df_list[idx] = df_list[idx].append({
                        'Pattern': change_fake_signal_to_ground(layer.pattern),
                        'W': list_to_str(layer.w_list),
                        'S': list_to_str(layer.spaces),
                        'Thickness': layer.thickness,
                        'Conductivity': layer.conductivity,
                        'Er': layer.er,
                        'Loss': layer.loss
                    }, ignore_index=True)
                    start_row.append(start_row[-1] + 8)
                    idx += 1
        return df_list, start_row[:-1]
    
    def write_xlsx(self,
                   configs: Config,
                   saved_name: str):
        """
        Write the stackup dataframe and parameter dataframe to an excel file.
        """
        
        # sampling_num is the number of stackups
        sampling_num = len(self.stackups)
        batch_num = configs.dataset_generation.batch_num
        # split the parameters_df to batches with batch_num equals 
        batches = sampling_num // batch_num + 1 if sampling_num % batch_num != 0 else sampling_num // batch_num
        
        out_dir = os.path.abspath(os.path.join(os.getcwd(), configs.dataset_generation.dir, saved_name))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for b in range(batches):
            script_dir = os.path.join(out_dir, f'{saved_name}_{b}')
            filename = os.path.join(script_dir, f'{saved_name}_{b}.xlsx')
            # Save file
            if not os.path.isdir(script_dir):
                os.mkdir(script_dir)
            
            # write the stackup dataframe and parameter dataframe to the excel file
            with pd.ExcelWriter(
                filename,
                mode="w",
                engine="openpyxl"
            ) as writer:
                self.stackup_df.to_excel(writer, sheet_name='Stackup', index=False)
                for i in range(len(self.parameters_df_list)):
                    self.parameters_df_list[i].iloc[b * batch_num : min((b + 1) * batch_num, sampling_num)].T.to_excel(writer, sheet_name='Parameter', header=False, startrow=self.start_row[i])
                
        return
    
    def load_xlsx(self,
                  script_dir: str,
                  batch_num: int):
        # TODO
        return


    def to_pickle(self, 
                  configs,
                  saved_name):
        """
        Save the stackup dataframe and parameter dataframe to an excel file.
        """
        out_dir = os.path.join(os.getcwd(), configs.dataset_generation.dir, saved_name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        saved_name = os.path.join(out_dir, f'{saved_name}.pkl')
        pickle.dump(self.stackups, open(saved_name, 'wb'))
        return 
    
    @classmethod
    def load_pickle(cls, 
                    configs,
                    saved_name):
        """
        Load the stackup dataframe and parameter dataframe from an excel file.
        """
        out_dir = os.path.join(os.getcwd(), configs.datasets.dir, configs.case, saved_name)
        if not os.path.isdir(out_dir):
            raise ValueError('The directory does not exist.')
        saved_name = os.path.join(out_dir, f'{saved_name}.pkl')
        stackups = pickle.load(open(saved_name, 'rb'))
        return cls(stackups)
    
def change_fake_signal_to_ground(pattern: str) -> str:
    # for example
    # GGFGG -> GGGGG
    return pattern.replace('F', 'G')

def list_to_str(l: list):
    s = ''
    for i in l:
        s += str(i) + ','
    s = s[:-1]
    return s

def load_stackup(config_dir):
    with open(config_dir, encoding="utf-8") as f:
        config = json.load(f)
    
def stackup_example(config_dir):
    configs = MultiLayerInputFeatures(config_dir)
    pattern = 'GGSGGGGSGGGGSGG'

    Air1 = AirLayer(nominal=configs)
    Metal1 = GroundLayer(nominal=configs)
    Die1 = DieLayer(nominal=configs)
    Metal2 = SignalLayer(nominal=configs, pattern=pattern)
    Die2 = DieLayer(nominal=configs)
    Metal3 = GroundLayer(nominal=configs)
    Air2 = AirLayer(nominal=configs)

    stackup = Stackup([Air1, Metal1, Die1, Metal2, Die2, Metal3, Air2])
    return stackup

def create_stackups_from_samples(
    input_features: MultiLayerInputFeatures,
    samples_np: np.ndarray,
    layer_num: int,
    method: str = 'same_line'
) -> List[Stackup]:
    """
    Create a list of stackups from a numpy array of samples. Currently restrict samples_np to be (N, 2) where the first column represents W and the second one represents S.
    (N, variable_num)

    Args:
        input_featuers (MultiLayerInputFeatures): The input features for the stackups.
        samples_np (np.ndarray): A numpy array of samples.
        layer_num (int): The number of layers in each stackup.

    Returns:
        List[Stackup]: A list of stackups.
    """
    def _get_value(
        variable: str,
        input_features,
        samples_np: np.ndarray,
        index: int):
        if variable in input_features.sampled_variables:
            value = samples_np[index, input_features.sampled_variables.index(variable)]
        elif variable.split('_')[0] in input_features.sampled_variables:
            value = samples_np[index, input_features.sampled_variables.index(variable.split('_')[0])]
        else:
            if hasattr(input_features, variable):
                parameter = getattr(input_features, variable)
            elif hasattr(input_features, variable.split('_')[0]):
                parameter = getattr(input_features, variable.split('_')[0])
            else:
                raise ValueError(f'The variable {variable} is not a valid variable.')
            if parameter.standard is not None:
                standard_variable = parameter.standard
                value = _get_value(standard_variable, input_features, samples_np, index)
            else:
                value = parameter.nominal
        return value

    def _build_layer_from_sample(
        input_features,
        samples_np: np.ndarray,
        method: str
    ):
        stackups = []
        pattern = input_features.pattern
        signal_line_per_layer = pattern.count('S')
        indices_arr = np.arange(signal_line_per_layer)
        for i in range(samples_np.shape[0]):
            Air = AirLayer(nominal=input_features)
            Metal = GroundLayer(nominal=input_features)
            # Die 1
            Die = DieLayer(nominal=input_features,
                           thickness=_get_value('H_1', input_features, samples_np, i),
                           er=_get_value('Er_1', input_features, samples_np, i),
                           loss=_get_value('Loss_1', input_features, samples_np, i))
            layers = [Air, Metal, Die]
            if method == 'center_aligned_shuffled_w':
                indices_arr = np.random.permutation(indices_arr)
            for j in range(layer_num):
                if method == 'same_line':
                    Metal = SignalLayer(nominal=input_features, 
                                        pattern=pattern, 
                                        w=_get_value(f'W_{j+1}', input_features, samples_np, i), 
                                        s=_get_value(f'S_{j+1}', input_features, samples_np, i),
                                        thickness=_get_value(f'T_{j+1}', input_features, samples_np, i),
                                        er=_get_value(f'Er_2', input_features, samples_np, i),
                                        loss=_get_value(f'Loss_2', input_features, samples_np, i))
                    Die = DieLayer(nominal=input_features,
                                thickness=_get_value(f'H_{j+2}', input_features, samples_np, i),
                                er=_get_value('Er_1', input_features, samples_np, i),
                                loss=_get_value('Loss_1', input_features, samples_np, i))
                elif method == 'same_column':
                    w_list = []
                    s_list = []
                    signal_idx = 0
                    for k in range(len(pattern)):
                        if pattern[k] == 'G':
                            w_list.append(_get_value(f'W', input_features, samples_np, i))
                            s_list.append(_get_value(f'S', input_features, samples_np, i))
                        # S means signal line, F means fake signal line
                        elif pattern[k] == 'S' or 'F':
                            w_list.append(_get_value(f'W_{signal_idx+1}', input_features, samples_np, i))
                            s_list.pop()
                            s = _get_value(f'S_{signal_idx+1}', input_features, samples_np, i)
                            s_list.extend([s, s])
                            signal_idx += 1
                        else:
                            raise ValueError('pattern item must be ground or signal')
                    s_list.pop()
                    Metal = SignalLayer(nominal=input_features, 
                                        pattern=pattern, 
                                        w=w_list, 
                                        s=s_list,
                                        thickness=_get_value(f'T_{j+1}', input_features, samples_np, i),
                                        er=_get_value(f'Er_2', input_features, samples_np, i),
                                        loss=_get_value(f'Loss_2', input_features, samples_np, i))
                    Die = DieLayer(nominal=input_features,
                                thickness=_get_value(f'H_{j+2}', input_features, samples_np, i),
                                er=_get_value('Er_1', input_features, samples_np, i),
                                loss=_get_value('Loss_1', input_features, samples_np, i))
                elif 'center_aligned' in method:
                    # the first layer is air
                    w_list = []
                    s_list = []
                    signal_idx = 0
                    for k in range(len(pattern)):
                        if pattern[k] == 'G':
                            w_list.append(_get_value(f'W', input_features, samples_np, i))
                            s_list.append(_get_value(f'S', input_features, samples_np, i))
                        # S means signal line, F means fake signal line
                        elif pattern[k] == 'S' or 'F':
                            if method == 'center_aligned_shuffled_w':
                                numpy_idx = indices_arr[signal_idx]
                            else:
                                numpy_idx = signal_idx
                            w = _get_value(f'W_{numpy_idx+1}', input_features, samples_np, i) * (1 - 0.1 * j)
                            w_list.append(w)
                            s_list.pop()
                            s = (_get_value(f'S', input_features, samples_np, i) * 2  + _get_value(f'W', input_features, samples_np, i) - w) / 2
                            s_list.extend([s, s])
                            signal_idx += 1
                        else:
                            raise ValueError('pattern item must be ground or signal')
                    s_list.pop()
                    Metal = SignalLayer(nominal=input_features, 
                                        pattern=pattern, 
                                        w=w_list, 
                                        s=s_list,
                                        thickness=_get_value(f'T_{j+1}', input_features, samples_np, i),
                                        er=_get_value(f'Er_2', input_features, samples_np, i),
                                        loss=_get_value(f'Loss_2', input_features, samples_np, i))
                    Die = DieLayer(nominal=input_features,
                                   thickness=_get_value(f'H_{j+2}', input_features, samples_np, i),
                                   er=_get_value('Er_1', input_features, samples_np, i),
                                   loss=_get_value('Loss_1', input_features, samples_np, i))
                else:
                    raise NotImplementedError
                layers.extend([Metal, Die])
            Metal = GroundLayer(nominal=input_features)
            Air = AirLayer(nominal=input_features)
            layers.extend([Metal, Air])
            stackups.append(Stackup(layers))
        return stackups
    
    # samples_np, sampled_varaiables
    # support modes:
    # sameline: the lines in the same layer have the same W/S
    # samecolumn: the lines in each column have the same W/S

    return _build_layer_from_sample(input_features, samples_np, method)
    
def create_samples_from_stackups(
    stackups: List[Stackup],
    input_features: MultiLayerInputFeatures,
    method: str = 'same_line'
) -> np.ndarray:
    """
    Create a numpy array of samples from a list of stackups. Currently restrict samples_np to be (N, 2) where the first column represents W and the second one represents S.
    (N, variable_num)

    Args:
        stackups (List[Stackup]): A list of stackups.
        input_featuers (MultiLayerInputFeatures): The input features for the stackups.

    Returns:
        np.ndarray: A numpy array of samples.
    """

    parameters = np.zeros([len(stackups), len(input_features.sampled_variables)])
    for i in range(len(stackups)):
        for v_i, v in enumerate(input_features.sampled_variables):
            if v == "W":
                parameters[i, v_i] = stackups[i].layers[3].w_list[0]
            elif v == "S":
                parameters[i, v_i] = stackups[i].layers[3].spaces[0]
            elif v == "H":
                parameters[i, v_i] = stackups[i].layers[4].thickness
            elif v == "T":
                parameters[i, v_i] = stackups[i].layers[3].thickness
            elif v == "Er":
                parameters[i, v_i] = stackups[i].layers[3].er
            elif v == "Loss":
                parameters[i, v_i] = stackups[i].layers[3].loss
            elif v.split('_')[0] == "W":
                # support differentline mode: same_column, center_aligned
                if method != 'same_line':
                    layer_index = int(v.split('_')[-1])
                    parameters[i, v_i] = stackups[i].layers[3].w_list[layer_index * 5 - 3]
                else:
                    layer_index = int(v.split('_')[-1])
                    parameters[i, v_i] = stackups[i].layers[1 + layer_index * 2].w_list[0]
            elif v.split('_')[0] == "H":
                layer_index = int(v.split('_')[-1])
                parameters[i, v_i] = stackups[i].layers[layer_index * 2].thickness
            elif v.split('_')[0] == "T":
                layer_index = int(v.split('_')[-1])
                parameters[i, v_i] = stackups[i].layers[1 + layer_index * 2].thickness
            elif v.split('_')[0] == "Er":
                layer_index = int(v.split('_')[-1])
                parameters[i, v_i] = stackups[i].layers[1 + layer_index].er
            elif v.split('_')[0] == "Loss":
                layer_index = int(v.split('_')[-1])
                parameters[i, v_i] = stackups[i].layers[1 + layer_index].loss
            else:
                raise ValueError(f"Unsupported variable type {v}")
    return parameters
