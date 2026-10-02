import torch
import torch.nn.functional as functional
from torch import Tensor

from astrouda.config import Config
from astrouda.logging_utils import get_logger

logger = get_logger(__name__, Config())


def compute_class_weights(source_labels: Tensor, number_of_classes: int) -> Tensor:
    """
    Eq. 4 weights w_k = N_s / (K * n_k) from integer source labels of shape (N_s,).

    A class with zero samples (n_k = 0) gets weight 0 instead of infinity. It can never appear
    as a source label, so the weight only matters for keeping the returned tensor finite.
    """
    class_counts: Tensor = torch.bincount(source_labels.long(), minlength=number_of_classes).to(torch.float32)
    logger.debug(f"Source class counts: {class_counts.tolist()}")

    missing_classes: list[int] = torch.nonzero(class_counts == 0).flatten().tolist()
    if missing_classes:
        logger.warning(f"Classes {missing_classes} have no source samples, giving them weight 0")

    safe_counts: Tensor = torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts))
    class_weights: Tensor = torch.where(
        class_counts > 0,
        source_labels.numel() / (number_of_classes * safe_counts),
        torch.zeros_like(class_counts),
    )
    logger.debug(f"Class weights: {class_weights.tolist()}")

    return class_weights


def weighted_cross_entropy_loss(logits: Tensor, labels: Tensor, class_weights: Tensor) -> Tensor:
    """
    Eq. 4 on source samples. Per-sample losses are scaled by w_{y_i} and divided by the sum of
    w_{y_i} over the batch, the same normalisation as torch's weighted CrossEntropyLoss.
    With equal weights this is plain mean cross entropy.
    """
    class_weights_on_device: Tensor = class_weights.to(device=logits.device, dtype=logits.dtype)
    loss: Tensor = functional.cross_entropy(logits, labels.long(), weight=class_weights_on_device, reduction="mean")
    logger.debug(f"Weighted cross entropy over {labels.numel()} samples: {loss.item():.6f}")

    return loss
