import logging

import torch
from torch import nn

from astrouda.config import Config

logger: logging.Logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.SGD:
    "Paper: SGD with (Nesterov) momentum and weight decay."
    optimizer: torch.optim.SGD = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        nesterov=config.use_nesterov,
        weight_decay=config.weight_decay,
    )
    logger.debug(
        f"SGD lr={config.learning_rate} momentum={config.momentum} nesterov={config.use_nesterov} "
        f"weight_decay={config.weight_decay}"
    )

    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config) -> torch.optim.lr_scheduler.StepLR:
    "Paper: learning rate multiplied by gamma every step_size epochs. The trainer steps it once per epoch."
    scheduler: torch.optim.lr_scheduler.StepLR = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.learning_rate_decay_every_epochs,
        gamma=config.learning_rate_decay_factor,
    )
    logger.debug(
        f"StepLR step_size={config.learning_rate_decay_every_epochs} gamma={config.learning_rate_decay_factor}"
    )

    return scheduler
