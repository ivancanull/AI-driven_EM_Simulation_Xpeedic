import os
import json
import numpy as np
import pandas as pd

class Parameters:

    def __init__(self, key, value: dict):
        """Construct a parameters object
        :param key: the name of the parameter
        :param args: a dictionary contains, min value, norminal value, max value, and variable or not.
        """
        self.name = key
        # check if the value is a reference to other parameters
        if key == 'Pattern':
            self._type = 'str'
            self.nominal = generate_pattern(value['num'])
            self._variable = value['variable']
            self.has_standard = False
            if self._variable:
                raise NotImplementedError('Pattern can\'t be variable.')
        else:
            self._type = 'float'
            if 'standard' in value:
                self._standard_value = value['standard']
                self.has_standard = True
            else:
                self._min = value['min']
                self.nominal = value['nominal']
                self._max = value['max']
                self._variable = value['variable']
                self.has_standard = False

                if self._variable:
                    self._decimals = value['decimals']
                else:
                    self._decimals = 0
            
    def get_standard(self):
        if self.has_standard:
            return (self._standard_value)
        else:
            raise ValueError('The value %s doesn\'t have a standard value.' % self.name)
    
    def get_decimals(self):
        return self._decimals
    
    def check_standard(self):
        if self.has_standard:
            raise ValueError('The value %s is a reference value.' % self.name)
        return
        
    def is_variable(self):
        if self.has_standard:
            return False
        else:
            return self._variable

    def min(self):
        self.check_standard()
        return self._min
    
    def max(self):
        self.check_standard()
        return self._max

    def range(self):
        self.check_standard()
        return [self._min, self._max]

class InputFeatures():
    def __init__(
        self,
        config_dir,
    ):
        self.parameter_dict = {}
        self.frequency = []
        if config_dir is not None:
            with open(config_dir, encoding="utf-8") as f:
                config = json.load(f)
            
            # read parameters from json files
            for key, value in config.items():
                if key == 'name':
                    pass
                elif key == 'Frequency':
                    self.frequency = value
                else:
                    self.parameter_dict[key] = Parameters(key, value)

        self.frequency_np, self.nF = self.get_frequency()
        self.variable_nominals = self.get_variable_nominal()
        self.sampled_variables, self.limits, self.decimals = self.get_variable_range()
        self.variable_num = len(self.sampled_variables)

        return
    
    def get_variable_nominal(
        self,
    ):
        """
        Returns the parameter dict list
        """
        nominals = []
        for key, p in self.parameter_dict.items():
            if p.is_variable():
                nominals.append(p.nominal)
        return np.array(nominals)
            

    def get_variable_range(
        self,
    ): 
        """
        Returns a list of variable parameters ranges
        :param parameters: a dict of parameters
        :return: an np.ndarray of variable parameters ranges
        sampled_variables, ranges, decimals
        """
        ranges = []
        sampled_variables = []
        decimals = []
        for key, p in self.parameter_dict.items():
            if p.is_variable():
                sampled_variables.append(key)
                ranges.append(p.range())
                decimals.append(p.get_decimals())
        return sampled_variables, np.array(ranges), decimals
    
    def get_frequency(self):
        f_start = self.frequency[0] * convert_Hz_to_num(self.frequency[1])
        f_end = self.frequency[2] * convert_Hz_to_num(self.frequency[3])
        f_step = self.frequency[4] * convert_Hz_to_num(self.frequency[5])
        frequency_np = np.arange(f_start, f_end + f_step, f_step)
        nF = len(frequency_np)
        return frequency_np, nF
    
    
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

class MultiLineInputFeatures(InputFeatures):

    def __init__(self, config_dir):

        self.parameter_dict = {}
        self.frequency = []
        if config_dir is not None:
            with open(config_dir, encoding="utf-8") as f:
                config = json.load(f)

            self.nline = config['N-Line']['nominal']
            # read parameters from json files
            for key, value in config.items():
                if key == 'name':
                    pass
                elif key == 'Frequency':
                    self.frequency = value
                else:
                    self.parameter_dict[key] = Parameters(key, value)
            
        self.frequency_np, self.nF = self.get_frequency()
        self.variable_nominals = self.get_variable_nominal()
        self.sampled_variables, self.limits, self.decimals = self.get_variable_range()
        self.variable_num = len(self.sampled_variables)

    def get_variable_nominal(
        self,
    ):
        """
        Returns the parameter dict list
        """
        nominals = []
        for key, p in self.parameter_dict.items():
            if p.is_variable():
                if key == 'S':
                    nominals += [p.nominal] * (self.nline - 1)
                elif key == 'W':
                    nominals += [p.nominal] * (self.nline)
                else:
                    nominals.append(p.nominal)
        return np.array(nominals)
            

    def get_variable_range(
        self,
    ): 
        """
        Returns a list of variable parameters ranges
        :param parameters: a dict of parameters
        :return: an np.ndarray of variable parameters ranges
        sampled_variables, ranges, decimals
        """
        ranges = []
        sampled_variables = []
        decimals = []
        for key, p in self.parameter_dict.items():
            if p.is_variable():
                # Write multiple keys for S and W
                if key == 'S':
                    sampled_variables += [f'S_{i}' for i in range(self.nline - 1)]
                    ranges += [p.range()] * (self.nline - 1)
                    decimals += [p.get_decimals()] * (self.nline - 1)
                elif key == 'W':
                    sampled_variables += [f'W_{i}' for i in range(self.nline)]
                    ranges += [p.range()] * (self.nline)
                    decimals += [p.get_decimals()] * (self.nline)
                else:
                    sampled_variables.append(key)
                    ranges.append(p.range())
                    decimals.append(p.get_decimals())
        return sampled_variables, np.array(ranges), decimals

# class Stackup():
#     def __init__(self, setting: dict):
#         self.columns = ["Layer Name", 
#                         "Thickness",
#                         "Conductivity",
#                         "Dielectric Er",
#                         "Dielectric Loss",
#                         "Parameter Index",
#                         "Ref Layer"]
#         self.stackup_df = pd.DataFrame(columns=self.columns)
#         for key, p in setting.items():
#             # add a row where Layer Name is key
#             self.stackup_df = self.stackup_df.append({'Layer Name': key}, ignore_index=True)
#             for inner_key, inner_p in p.items():
#                 self.stackup_df.at[self.stackup_df.index[-1], inner_key] = inner_p
        
#     @property
#     def stackup(self):
#         return self.stackup_df

# class ParameterIndex(InputFeatures):
#     def __init__(self, setting: dict):
#         # support 4 different settings now
#         columns_metal = ["Pattern",
#                          "W",
#                          "S",
#                          "Thickness",
#                          "Con",
#                          "Er",
#                          "Loss"]
#         columns_die = ["Thickness",
#                        "Con",
#                        "Er",
#                        "Loss"]
#         self.parameter_dict = {} # 4 different settings
#         self.wire_num = []
#         for key, p in setting.items():
#             self.parameter_dict[key] = {}
#             for inner_key, inner_p in p.items():
#                 self.parameter_dict[key][inner_key] = Parameters(inner_key, inner_p)
#                 if inner_key == 'Pattern':
#                     self.wire_num.append(len(self.parameter_dict[key][inner_key].nominal))
#         self.sampled_variables, self.limits, self.decimals, self.variable_nominals, self.varaiable_indices, self.constants_nominals, self.constants_indices = self.get_variable_range()
#         self.variable_num = len(self.sampled_variables)

#     def get_variable_range(
#         self,
#     ): 
#         """
#         Returns a list of variable parameters ranges
#         :param parameters: a dict of parameters
#         :return: an np.ndarray of variable parameters ranges
#         sampled_variables, ranges, decimals
#         """
#         ranges = []
#         sampled_variables = []
#         decimals = []
#         variable_nominals = []
#         varaiable_indices = [] # variable_indices locate the variable row number in xlsx file
        
#         constants_nominals = []
#         constants_indices = []
#         index = 0
#         for key, p in self.parameter_dict.items():

#             for inner_key, inner_p in p.items():

#                 if inner_p.is_variable():
#                     sampled_variables.append((key, inner_key))
#                     ranges.append(inner_p.range())
#                     variable_nominals.append(inner_p.nominal)
#                     decimals.append(inner_p.get_decimals())
#                     varaiable_indices.append(index)
#                 else:
#                     constants_nominals.append(inner_p.nominal)
#                     constants_indices.append(index)
#                 index += 1
#             index += 1
#         return sampled_variables, np.array(ranges), decimals, variable_nominals, varaiable_indices, constants_nominals, constants_indices 
        
        
# class MultiLayerInputFeatures_archive(InputFeatures):

#     def __init__(self, config_dir):

#         if config_dir is not None:
#             with open(config_dir, encoding="utf-8") as f:
#                 config = json.load(f)
#             self.frequency = config["Frequency"]
#             self.stackup = Stackup(config["Layer Name"])
#             self.parameter_index = ParameterIndex(config["Parameter"])


#         self.frequency_np, self.nF = self.get_frequency()
#         self.variable_num = self.parameter_index.variable_num
#         self.sampled_variables = []
#         for (index, p) in self.parameter_index.sampled_variables:
#             self.sampled_variables.append(f'index_{int(index)-1}_{p}')
#         self.sampled_variables += ['coord_delta_x', 'coord_delta_y']
#         self.train_variable_num = len(self.sampled_variables)

# def generate_pattern(
#     group: int
# ):
#     default_pattern = 'GGSGG'
#     # repeat default pattern 'group' times and concat then return
#     return ''.join([default_pattern for _ in range(group)])
