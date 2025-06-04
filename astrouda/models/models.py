

from abc import ABC, abstractmethod
import json
import os
from torchvision.models import resnet50
import torch

from astrouda.utils import utils


class Model(ABC): 
    def __init__(self, config, data_loader, output_path): 
        self.config = config
        self.logger = utils.make_logger(config)
        self.device = utils.get_device(config)
        self.data_loader = data_loader
        self.output_path = output_path

        self.loss_history = {"val": [], "train": []},
        self.best_checkpoint = 0

        self.optimizer = self._load_optimizer(config.get("optimizer", "adam"))
        self.scheduler = self._load_scheduler(config.get("scheduler", "step_lr"))
        self.loss_fn = self._load_loss(config.get("loss", "cross_entropy"))

        self.model = self.load_model()
        self.load_checkpoint()

    def _load_loss(self, loss_name):
        ""

    def _load_optimizer(self, optimizer_name):
        ""
    
    def _load_scheduler(self, scheduler_name):
        ""

    def train(self):
        epochs = self.config.get("epochs", 10)
        self.logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            ""
            self.model.train()
            train_loss = self.model_forward(self.data_loader.train)
            train_loss.backward()

            self.model.eval()
            val_loss = self.model_forward(self.data_loader.val)
            self.logger.info(f"Ending epoch {epoch} with train loss {train_loss} and val loss {val_loss}")

            self.optimizer.step()
            self.scheduler.step()

            self.loss_history["train"].append(train_loss)
            self.loss_history["val"].append(val_loss)
            self.save_checkpoint()

        self.logger.info("Training complete. Saving final model.")
        self.save_results()

    def inference(self): 
        ""
        self.model.eval()

    @abstractmethod
    def model_forward(self, data_loader) -> float: 
        """Takes a data loader, iterates through batch by batch, and returns the total loss per epoch. Does not step th optimizer or scheduler."""

    @abstractmethod
    def load_model(self): 
        "Initialize a base model based on configuration"

    def save_checkpoint(self): 
        "Save a checkpoint if the validation is better than the best val. Save the loss history."
        with open(f"{self.output_path}/history.json", "w") as f:
            json.dump(self.loss_history, f)

        torch.save(
            {
                "optimizer": self.optimizer.state_dict(), 
                "schedule": self.scheduler.state_dict()}, f"{self.output_path}/training_state.pth"
        )
        if self.loss_history["val"][-1] > self.loss_history["val"][self.best_checkpoint]:
            self.logger.info(f"Saving new best checkpoint with validation loss: {self.loss_history['val'][-1]}")
            torch.save(self.model.state_dict(), f"{self.output_path}/best_val.pth")
            self.best_checkpoint = len(self.loss_history["val"])

    def save_results(self): 
        self.save_checkpoint()
        torch.save(self.model.state_dict(), f"{self.output_path}/final.pth")

    def load_checkpoint(self):
        "If the output dir already has a checkpoint, load it (if retrain=False)"
        if self.config.get("retrain", False): 
            self.logger.info("Retraining is enabled, skipping checkpoint loading.")
        else: 
            checkpoint = f"{self.output_path}/best_val.pth"
            if not os.path.exists(checkpoint):
                self.logger.info("No checkpoint found, starting from scratch.")
                return
            self.logger.info(f"Loading checkpoint from {checkpoint}")
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

            self.optimizer.load_state_dict(
                torch.load(f"{self.output_path}/training_state.pth", map_location=self.device)["optimizer"]
            )
            self.scheduler.load_state_dict(
                torch.load(f"{self.output_path}/training_state.pth", map_location=self.device)["schedule"]
            )
        
            self.loss_history = json.load(open(f"{self.output_path}/history.json", "r"))


class ResNet50(Model): 
    def __init__(self, config, data_loader, output_path):
        super().__init__(config, data_loader, output_path)

    def model_forward(self, data_loader):
        loss = []
        for batch in data_loader:
            ""

        loss = []
        return loss

    def load_model(self):
        # Reference: https://github.com/pytorch/vision/blob/966da7e46f65d6d49df3e31214470a4fe5cc8e66/torchvision/models/resnet.py#L166
        model = resnet50(
            num_classes="", 
            device=self.device
        )
        return model
