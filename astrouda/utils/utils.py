import yaml
import torch

def load_config(config_path: str):
    """
    Load a configuration file in YAML format.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # TODO Run validation

    return config

def make_logger(config:dict):
    """
    Create a logger based on the configuration.

    Args:
        config (dict): Configuration parameters.

    Returns:
        logging.Logger: Configured logger instance.
    """
    import logging

    logging.basicConfig(level=config.get('log_level', 'INFO'))

    return logging.getLogger(__name__) 

def get_device(config: dict):
    """
    Get the device to be used for computations based on the configuration.

    Args:
        config (dict): Configuration parameters.

    Returns:
        str: Device name ('cpu' or 'cuda').
    """
    if config.get('use_cuda', True) and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def calculate_accuracy(predictions, labels):
    """
    Calculate classwise accuracy, returning a dictionary with accuracy for each class.
    """
    accuracy_dict = {}
    for _class in torch.unique(labels):
        class_mask = labels == _class
        class_accuracy = (predictions[class_mask] == labels[class_mask]).float().mean().item()
        accuracy_dict[_class.item()] = class_accuracy
        
    return accuracy_dict


def plot_accuracy(accuracy_history: dict, config: dict) -> None:
    "Plot accuracy of the different classes and datasets as a function of epoch"

def plot_loss(loss_history: dict, config: dict) -> None: 
    "Plot the multiple components of loss as a function of epoch"

def plot_auc():
    ""