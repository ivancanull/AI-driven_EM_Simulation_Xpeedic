from ..datasets import MultiLayerInputFeatures

import numpy as np
class SweepSampler():
    def __init__(self, 
                 input_features) -> None:
        
        self.input_features = input_features

    def sample(self) -> np.ndarray:
        meshs = []
        for variable in self.input_features.sampled_variables:
            meshs.append(getattr(self.input_features, variable).values)

        if self.input_features.combination == "Dot":
            for i in range(len(meshs)):
                if isinstance(meshs[i][0], list):
                    raise ValueError(f'The components of parameter {self.input_features.sampled_variables[i]} have multiple values but combination is set to Dot.')
            samples_np = np.array(np.meshgrid(*meshs)).reshape([len(self.input_features.sampled_variables), -1]).T

        elif self.input_features.combination == "Pair":
            combination_num = self.input_features.get_combination_num()
            print(combination_num)
            samples_np_list = []
            for i in range(combination_num):
                sub_meshes = []
                for j in range(len(meshs)):
                    if isinstance(meshs[j][0], list):
                        sub_meshes.append(meshs[j][i])
                    else:
                        sub_meshes.append(meshs[j])
                samples_np = np.array(np.meshgrid(*sub_meshes)).reshape([len(self.input_features.sampled_variables), -1]).T
                samples_np_list.append(samples_np)
            samples_np = np.concatenate(samples_np_list, axis=0)

        elif self.input_features.combination == "Mesh":
            # TODO
            raise NotImplementedError

        else:
            raise ValueError(f'Combination typer {self.input_features.combination} not supported. Only Dot, Pair, and Mesh are availble.')
        
        return samples_np

class RatioSweepSampler():
    def __init__(self, 
                 input_features) -> None:
        
        self.input_features = input_features

    def sample(self) -> np.ndarray:
        meshs = []
        for variable in self.input_features.sampled_variables:
            meshs.append(getattr(self.input_features, variable).values)

        if self.input_features.combination == "Dot":

            for i in range(len(meshs)):
                # assert every list in meshs have the same length
                assert len(set([len(x) for x in meshs])) == 1
            samples_np = np.array(meshs).T

        elif self.input_features.combination == "Pair":
            # TODO
            raise NotImplementedError

        elif self.input_features.combination == "Mesh":
            # TODO
            raise NotImplementedError

        else:
            raise ValueError(f'Combination typer {self.input_features.combination} not supported. Only Dot, Pair, and Mesh are availble.')
        
        return samples_np

class TestSweepSampler():
    def __init__(self, 
                 input_features,
                 step_ratio = 0.5) -> None:
        
        self.input_features = input_features
        # create a densor sample space compared to original settings
        self.step_ratio = step_ratio

    def sample(self) -> np.ndarray:
        meshs = []
        for variable in self.input_features.sampled_variables:
            v = getattr(self.input_features, variable)
            new_values = np.linspace(v.min, v.max, round((v.max - v.min) / v.step / self.step_ratio), endpoint=False).tolist()
            meshs.append(new_values)

        if self.input_features.sampled_variables == ["W", "S"]:
            X, Y = np.meshgrid(*meshs)
            wn, sn = len(meshs[0]), len(meshs[1])
        else:
            X, Y = None, None

        if self.input_features.combination == "Dot":
            for i in range(len(meshs)):
                if isinstance(meshs[i][0], list):
                    raise ValueError(f'The components of parameter {self.input_features.sampled_variables[i]} have multiple values but combination is set to Dot.')
            samples_np = np.array(np.meshgrid(*meshs)).reshape([len(self.input_features.sampled_variables), -1]).T

        elif self.input_features.combination == "Pair":
            combination_num = self.input_features.get_combination_num()
            print(combination_num)
            samples_np_list = []
            for i in range(combination_num):
                sub_meshes = []
                for j in range(len(meshs)):
                    if isinstance(meshs[j][0], list):
                        sub_meshes.append(meshs[j][i])
                    else:
                        sub_meshes.append(meshs[j])
                samples_np = np.array(np.meshgrid(*sub_meshes)).reshape([len(self.input_features.sampled_variables), -1]).T
                samples_np_list.append(samples_np)
            samples_np = np.concatenate(samples_np_list, axis=0)

        elif self.input_features.combination == "Mesh":
            # TODO
            raise NotImplementedError

        else:
            raise ValueError(f'Combination typer {self.input_features.combination} not supported. Only Dot, Pair, and Mesh are availble.')
        
        return samples_np, X, Y, wn, sn