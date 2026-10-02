import logging
from typing import Any, Optional

import torch
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)


class ProbabilityBank:
    """
    FIFO memory bank of detached softmax probability vectors, capacity counted in samples.

    Design choices (paper silent): it stores probabilities rather than features, holds source and
    target samples together, and carries no gradient, so only the current target batch is trained
    through L_AC. Oldest samples are evicted first.
    """

    def __init__(self, capacity: int, number_of_classes: int, device: Optional[torch.device] = None) -> None:
        self.capacity: int = capacity
        self.number_of_classes: int = number_of_classes
        self._stored_probabilities: Tensor = torch.empty(0, number_of_classes, device=device)
        logger.debug(f"Created probability bank, capacity={capacity}, classes={number_of_classes}, device={device}")

    @property
    def probabilities(self) -> Tensor:
        return self._stored_probabilities

    def __len__(self) -> int:
        return self._stored_probabilities.shape[0]

    def add(self, probabilities: Tensor) -> None:
        "Append a (n, K) batch, then drop the oldest samples beyond capacity."
        incoming_probabilities: Tensor = probabilities.detach().to(
            device=self._stored_probabilities.device, dtype=self._stored_probabilities.dtype
        )
        combined_probabilities: Tensor = torch.cat([self._stored_probabilities, incoming_probabilities], dim=0)
        self._stored_probabilities = combined_probabilities[-self.capacity :].clone()
        logger.debug(f"Bank added {incoming_probabilities.shape[0]} samples, now holds {len(self)}/{self.capacity}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "number_of_classes": self.number_of_classes,
            "probabilities": self._stored_probabilities.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.number_of_classes = int(state["number_of_classes"])
        self._stored_probabilities = state["probabilities"].to(self._stored_probabilities.device).clone()
        logger.debug(f"Bank restored with {len(self)} samples, capacity={self.capacity}")


def ordered_top_k_similarity_labels(probabilities_a: Tensor, probabilities_b: Tensor, top_k: int) -> Tensor:
    """
    (n_a, n_b) binary matrix, 1 where the top_k classes of two samples are identical in the same rank order.

    Each ordered top-k index tuple is encoded as one integer in base K, so the comparison is a single
    (n_a, n_b) equality instead of a (n_a, n_b, k) broadcast. Ties in probability are broken by torch.topk.
    """
    number_of_classes: int = probabilities_a.shape[1]
    positional_base: Tensor = number_of_classes ** torch.arange(
        top_k, device=probabilities_a.device, dtype=torch.long
    )

    codes_a: Tensor = (probabilities_a.topk(top_k, dim=1).indices * positional_base).sum(dim=1)
    codes_b: Tensor = (probabilities_b.topk(top_k, dim=1).indices * positional_base).sum(dim=1)
    similarity_labels: Tensor = (codes_a[:, None] == codes_b[None, :]).to(probabilities_a.dtype)
    logger.debug(f"Top-{top_k} similarity labels {tuple(similarity_labels.shape)}, positive fraction={similarity_labels.mean().item():.4f}")

    return similarity_labels


def adaptive_clustering_loss(
    target_probabilities: Tensor,
    bank_probabilities: Tensor,
    top_k: int,
    probability_epsilon: float,
) -> Tensor:
    """
    Eq. 1: binary cross entropy between s_ij and shat_ij = p_i . p_j for every (bank i, target j) pair.

    Reduction is the mean over pairs (the paper sums), so the magnitude does not depend on bank fill.
    shat is clamped to [eps, 1 - eps]. Bank entries are detached, so gradients reach only the target
    batch. The bank must not contain the current batch. An empty bank gives a zero loss that keeps the graph.
    """
    if bank_probabilities.shape[0] == 0:
        logger.debug("Bank empty, adaptive clustering loss is zero")
        return target_probabilities.sum() * 0.0

    detached_bank: Tensor = bank_probabilities.detach().to(target_probabilities.device)
    similarity_labels: Tensor = ordered_top_k_similarity_labels(detached_bank, target_probabilities.detach(), top_k)

    predicted_similarity: Tensor = (detached_bank @ target_probabilities.T).clamp(
        min=probability_epsilon, max=1.0 - probability_epsilon
    )
    pairwise_loss: Tensor = -(
        similarity_labels * torch.log(predicted_similarity)
        + (1.0 - similarity_labels) * torch.log(1.0 - predicted_similarity)
    )
    loss: Tensor = pairwise_loss.mean()
    logger.debug(f"Adaptive clustering over {tuple(pairwise_loss.shape)} pairs: {loss.item():.6f}")

    return loss
