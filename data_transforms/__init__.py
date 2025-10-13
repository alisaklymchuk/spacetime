from .mean import MeanTransform, MeanInputTransform
from .standardize import StandardizeTransform
from .affine import InverseAffineTransform
from .affine import InversePercentageAffineTransform
from .affine import InverseMeanPercentageAffineTransform
from .last import LastAffineTransform
from .mean_percentage import MeanPercentageTransform
from .mean_percentage import PercentageTransform


def get_data_transforms(method, lag):
    supported_methods = ['mean', 'mean_input', 'last', 
                         'standardize', 'percentage', 'mean_percentage', 'none']
    if method == 'mean':
        input_transform = MeanTransform(lag)
        output_transform = InverseAffineTransform(input_transform)
    elif method == 'mean_input':
        input_transform = MeanInputTransform(lag)
        output_transform = InverseAffineTransform(input_transform)
    elif method == 'last':
        input_transform = LastAffineTransform(lag)
        output_transform = InverseAffineTransform(input_transform)   
    elif method == 'standardize':
        input_transform = StandardizeTransform(lag)
        output_transform = InverseAffineTransform(input_transform)
    elif method == 'percentage':
        input_transform = PercentageTransform(lag)
        output_transform = InversePercentageAffineTransform(input_transform)
    elif method == 'mean_percentage':
        input_transform = MeanPercentageTransform(lag)
        output_transform = None # InverseMeanPercentageAffineTransform(input_transform)
    elif method == 'none':
        input_transform = lambda x: x
        output_transform = lambda x: x
    else:
        raise NotImplementedError(f"Data transform method '{method}' not supported. Please choose from {supported_methods}.")
        
    return input_transform, output_transform