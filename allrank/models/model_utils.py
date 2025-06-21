from typing import Any

import numpy as np
import torch
import torch.nn as nn

from allrank.utils.file_utils import is_gs_path, copy_file_to_local
from allrank.utils.ltr_logging import get_logger

logger = get_logger()


def get_torch_device():
    """
    Getter for an available pyTorch device.
    :return: CUDA-capable GPU if available, CPU otherwise
    """
    print(torch.cuda.is_available())
    return torch.device("cuda:0") 
    
    #if torch.cuda.is_available() else torch.device("cpu")
    #return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_num_params(model: nn.Module) -> int:
    """
    Calculation of the number of nn.Module parameters.
    :param model: nn.Module
    :return: number of parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params  # type: ignore


def log_num_params(num_params: int) -> None:
    """
    Logging num_params to the global logger.
    :param num_params: number of parameters to log
    """
    logger.info("Model has {} trainable parameters".format(num_params))
    

def get_model_parameters(model, parameter_type=None):
    """
    Get parameters from model based on type, handling both parallel and non-parallel cases
    
    Args:
        model: PyTorch model (either wrapped in DataParallel/DistributedDataParallel or not)
        parameter_type: str or None, options: 
            - 'shared_parameters' for shared parameters
            - 'task_specific_parameters' for task-specific parameters
            - 'last_shared_parameters' for last shared parameters
            - None to get all parameters
    
    Returns:
        list: Model parameters based on specified type
    
    Raises:
        AttributeError: If the requested parameter_type method doesn't exist in the model
    """
    # Get the base model (handles both parallel and non-parallel cases)
    base_model = model.module if hasattr(model, 'module') else model
    
    if parameter_type is None:
        return list(base_model.parameters())
    
    # Validate that the requested method exists
    if not hasattr(base_model, parameter_type):
        raise AttributeError(f"Model does not have method: {parameter_type}")
    
    # Get and return the parameters
    return list(getattr(base_model, parameter_type)())

    
        
class CustomDataParallel(nn.DataParallel):
    """
    Wrapper for scoring with nn.DataParallel object containing LTRModel.
    """

    def score(self, x, mask, indices):
        """
        Wrapper function for a forward pass through the whole LTRModel and item scoring.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        return self.module.score(x, mask, indices)  # type: ignore


def load_state_dict_from_file(path: str, device: Any):
    if is_gs_path(path):
        path = copy_file_to_local(path)

    return torch.load(path, map_location=device)
