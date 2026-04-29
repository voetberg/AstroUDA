from astrouda.models.train import TrainDA
from astrouda.utils import utils
import sys

args = sys.argv
if len(args) > 1:
    config = args[1]
else:
    raise ValueError("Please provide a config file path as a command line argument. Example: python inference.py config.yaml")

if __name__ == "__main__":
    config = utils.load_config(config)
    train_da = TrainDA(config)
    train_da.test()