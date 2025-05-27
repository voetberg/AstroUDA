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