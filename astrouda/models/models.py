

from abc import ABC, abstractmethod


class Model(ABC): 
    def __init__(self, config, data_loader, output_path): 
        ""

    def _load_loss(self, loss_name):
        ""

    def _load_optimizer(self, optimizer_name):
        ""
    
    def _load_scheduler(self, scheduler_name):
        ""

    def _evaluate(self): 
        ""

    def train(self):
        ""

    def inference(self): 
        ""

    @abstractmethod
    def model_forward(self): 
        ""

    @abstractmethod
    def load_model(self): 
        ""

    def save_checkpoint(self): 
        ""

    def save_results(): 
        ""