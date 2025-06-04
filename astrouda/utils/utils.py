import yaml


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