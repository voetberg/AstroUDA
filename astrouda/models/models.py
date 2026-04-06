

from abc import ABC, abstractmethod
import os
from torchvision.models import resnet50, resnet18, ResNet18_Weights, ResNet50_Weights
import torch

from astrouda.utils import utils


class Model(ABC): 
    def __init__(self, config): 
        self.config = config
        self.output_path = config.get("output_path", "./outputs")
        self.logger = utils.make_logger(config)
        self.device = utils.get_device(config)
        
        self.feature_model = self.init_feature_model()
        self.classifier_model = self.init_classifier_model()

        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()


    def load_optimizer(self) -> torch.optim.Optimizer:
        optim_types = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD
        }
        optim_type = self.config.get("optimizer", "adam").lower()
        if optim_type not in optim_types:
            raise ValueError(f"Optimizer {optim_type} not supported. Choose from {list(optim_types.keys())}")
        return optim_types[optim_type](self.classifier_model.parameters(), lr=self.config.get("learning_rate", 1e-3))
    
    def load_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        sched_types = {
            "steplr": torch.optim.lr_scheduler.StepLR,
            "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
            "inverse_decay": torch.optim.lr_scheduler.LambdaLR
        }
        sched_type = self.config.get("scheduler", "steplr").lower()
        if sched_type not in sched_types:
            raise ValueError(f"Scheduler {sched_type} not supported. Choose from {list(sched_types.keys())}")

        return sched_types[sched_type](self.optimizer, step_size=self.config.get("scheduler_step_size", 10), gamma=self.config.get("scheduler_gamma", 0.1))


    def save_checkpoint(self) -> None: 
        """Save optimizer, scheduler, feature and classifier model states"""
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(), 
                "schedule": self.scheduler.state_dict(),
                "feature_model": self.feature_model.state_dict(),
                "classifier_model": self.classifier_model.state_dict()
            }, 
            f"{self.output_path}/training_state.pth"
        )

    def save_results(self) -> None: 
        "Save a checkpoint with all training data, and set the classifier model as a final result"
        self.save_checkpoint()
        torch.save(self.classifier_model.state_dict(), f"{self.output_path}/final.pth")

    def load_checkpoint(self) -> None:
        "If the output dir already has a checkpoint, load it (if retrain=False)"
        if self.config.get("retrain", False): 
            self.logger.info("Retraining is enabled, skipping checkpoint loading.")
        else: 
            checkpoint = f"{self.output_path}/training_state.pth"
            if not os.path.exists(checkpoint):
                self.logger.info("No checkpoint found, starting from scratch.")
                return
            self.logger.info(f"Loading checkpoint from {checkpoint}")
            checkpoint_data = torch.load(checkpoint, map_location=self.device)

            self.feature_model.load_state_dict(checkpoint_data["feature_model"])
            self.classifier_model.load_state_dict(checkpoint_data["classifier_model"])
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            self.scheduler.load_state_dict(checkpoint_data["schedule"])

    @abstractmethod
    def init_feature_model(self) -> torch.nn.Module:
        raise NotImplementedError
    
    @abstractmethod
    def init_classifier_model(self) -> torch.nn.Module:
        raise NotImplementedError


class AdaptiveModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def init_feature_model(self):
        class FeatureExtractor(torch.nn.Module):
            def __init__(self, feature_extractor="resnet18"):
                super().__init__()
                if feature_extractor == "resnet18":
                    self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
                else: 
                    self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.resnet_out = torch.nn.Linear(1000, 2048)

            def forward(self, x):
                return self.resnet_out(self.resnet(x))
            
        return FeatureExtractor(feature_extractor=self.config.get("feature_extractor", "resnet50").lower())

    def init_classifier_model(self):
        class Classifier(torch.nn.Module):
            def __init__(self, num_classes=10, feature_extractor="resnet50"):
                super().__init__()
                if feature_extractor == "resnet18":
                    self.feature_extractor = resnet18(weights=None, num_classes=2048)
                else: 
                    self.feature_extractor = resnet50(weights=None, num_classes=2048)
                self.fc = torch.nn.Linear(2048, num_classes)

            def forward(self, x, bypass_feature_extractor=False):
                if not bypass_feature_extractor:
                    x = self.feature_extractor(x)

                return self.fc(x)
            
        return Classifier(num_classes=self.config.get("num_classes", 10), feature_extractor=self.config.get("feature_extractor", "resnet50").lower())