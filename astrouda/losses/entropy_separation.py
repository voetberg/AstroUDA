import torch
from torch import Tensor

from astrouda.config import Config
from astrouda.logging_utils import get_logger

logger = get_logger(__name__, Config())


def entropy(probabilities: Tensor, probability_epsilon: float) -> Tensor:
    """
    Shannon entropy (natural log) per row of a (batch, K) softmax probability tensor.
    Probabilities are clamped at probability_epsilon inside the log, so p = 0 contributes 0 and
    the gradient stays finite.
    """
    clamped_probabilities: Tensor = probabilities.clamp(min=probability_epsilon)

    return -(probabilities * torch.log(clamped_probabilities)).sum(dim=1)


def entropy_separation_loss(
    probabilities: Tensor,
    entropy_boundary: float,
    confidence_margin: float,
    probability_epsilon: float,
) -> Tensor:
    """
    Eqs. 2-3 on target softmax probabilities (not logits): per sample -|H(p) - rho| when
    |H(p) - rho| > m, else 0, averaged over the batch. rho is entropy_boundary, m is confidence_margin.
    """
    sample_entropy: Tensor = entropy(probabilities, probability_epsilon)
    distance_from_boundary: Tensor = torch.abs(sample_entropy - entropy_boundary)
    per_sample_loss: Tensor = torch.where(
        distance_from_boundary > confidence_margin,
        -distance_from_boundary,
        torch.zeros_like(distance_from_boundary),
    )
    loss: Tensor = per_sample_loss.mean()
    logger.debug(
        f"Entropy separation: rho={entropy_boundary:.4f} m={confidence_margin:.4f} "
        f"active_fraction={(distance_from_boundary > confidence_margin).float().mean().item():.3f} loss={loss.item():.6f}"
    )

    return loss
