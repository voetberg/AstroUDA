import logging
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)


def total_loss(
    cross_entropy_loss: Tensor,
    adaptive_clustering_loss: Tensor,
    entropy_separation_loss: Tensor,
    domain_adaptation_weight: float,
) -> Tensor:
    "L = L_CE + lambda * (L_AC + L_ES)."
    combined_loss: Tensor = cross_entropy_loss + domain_adaptation_weight * (
        adaptive_clustering_loss + entropy_separation_loss
    )
    logger.debug(
        f"Total loss {combined_loss.item():.6f} = CE {cross_entropy_loss.item():.6f} + "
        f"{domain_adaptation_weight} * (AC {adaptive_clustering_loss.item():.6f} + ES {entropy_separation_loss.item():.6f})"
    )

    return combined_loss
