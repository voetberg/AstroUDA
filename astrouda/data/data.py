import os
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class DataLoader(ABC): 
    def __init__(self, config, source_dir, training_name, validation_name, test_name):
        self.source_dir = source_dir
        self.config = config
        
        self.train = self.get_data(training_name)
        self.val = self.get_data(validation_name)
        self.test = self.get_data(test_name)

    @abstractmethod
    def load_path(self, data_path) -> DataLoader: 
        """
        Returns the dataloader object for a given data path.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_data(self, name):
        data_path = f"{self.source_dir}/{name}"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        
        return self.load_path(data_path)