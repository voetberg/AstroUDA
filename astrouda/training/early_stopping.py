import logging
import math
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Stops when a metric (higher is better) has not strictly improved for patience_epochs consecutive epochs.
    The first update always counts as an improvement.
    """

    def __init__(self, metric_name: str, patience_epochs: int) -> None:
        self.metric_name: str = metric_name
        self.patience_epochs: int = patience_epochs
        self.best_value: float = -math.inf
        self.best_epoch: int = 0
        self.epochs_without_improvement: int = 0

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience_epochs

    def update(self, metric_value: float, epoch: int) -> bool:
        "Record one epoch. Returns True when this epoch is the new best."
        is_improvement: bool = metric_value > self.best_value

        if is_improvement:
            self.best_value = float(metric_value)
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        logger.debug(
            f"Early stopping on {self.metric_name}: value={metric_value:.6f} best={self.best_value:.6f} "
            f"(epoch {self.best_epoch}) without_improvement={self.epochs_without_improvement}/{self.patience_epochs}"
        )

        return is_improvement

    def state_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "patience_epochs": self.patience_epochs,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_value = float(state["best_value"])
        self.best_epoch = int(state["best_epoch"])
        self.epochs_without_improvement = int(state["epochs_without_improvement"])
        logger.debug(f"Early stopping restored: best={self.best_value} at epoch {self.best_epoch}")
