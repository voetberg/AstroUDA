import math

import pytest
import torch
from torch import Tensor

from astrouda.losses import entropy, entropy_separation_loss, total_loss

EPSILON: float = 1e-8


def test_entropy_of_uniform_and_one_hot() -> None:
    probabilities: Tensor = torch.tensor([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])

    sample_entropy: Tensor = entropy(probabilities, EPSILON)

    assert sample_entropy[0].item() == pytest.approx(math.log(4), rel=1e-6)
    assert sample_entropy[1].item() == pytest.approx(0.0, abs=1e-6)


def test_zero_inside_margin() -> None:
    probabilities: Tensor = torch.full((4, 3), 1 / 3)

    loss: Tensor = entropy_separation_loss(probabilities, entropy_boundary=math.log(3) - 0.1, confidence_margin=0.4, probability_epsilon=EPSILON)

    assert loss.item() == 0.0


def test_negative_distance_outside_margin() -> None:
    probabilities: Tensor = torch.tensor([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    entropy_boundary: float = 0.5

    loss: Tensor = entropy_separation_loss(probabilities, entropy_boundary, confidence_margin=0.4, probability_epsilon=EPSILON)

    one_hot_term: float = -abs(0.0 - entropy_boundary)
    uniform_term: float = -abs(math.log(3) - entropy_boundary)
    assert loss.item() == pytest.approx((one_hot_term + uniform_term) / 2, rel=1e-5)


def test_symmetric_about_boundary() -> None:
    entropy_boundary: float = 0.6
    margin: float = 0.1
    candidate_leading_probabilities: Tensor = torch.linspace(0.34, 0.999, 20000)
    candidate_probabilities: Tensor = torch.stack(
        [
            candidate_leading_probabilities,
            (1 - candidate_leading_probabilities) / 2,
            (1 - candidate_leading_probabilities) / 2,
        ],
        dim=1,
    )
    candidate_entropies: Tensor = entropy(candidate_probabilities, EPSILON)
    below_index: int = int((candidate_entropies - (entropy_boundary - 0.3)).abs().argmin())
    above_index: int = int((candidate_entropies - (entropy_boundary + 0.3)).abs().argmin())

    below_loss: Tensor = entropy_separation_loss(candidate_probabilities[below_index : below_index + 1], entropy_boundary, margin, EPSILON)
    above_loss: Tensor = entropy_separation_loss(candidate_probabilities[above_index : above_index + 1], entropy_boundary, margin, EPSILON)

    assert below_loss.item() == pytest.approx(-0.3, abs=1e-3)
    assert above_loss.item() == pytest.approx(-0.3, abs=1e-3)


def test_gradients_finite_with_one_hot_probabilities() -> None:
    probabilities: Tensor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)

    loss: Tensor = entropy_separation_loss(probabilities, entropy_boundary=0.5, confidence_margin=0.1, probability_epsilon=EPSILON)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(probabilities.grad).all()


def test_gradients_finite_through_softmax_of_extreme_logits() -> None:
    logits: Tensor = torch.tensor([[200.0, -200.0, -200.0]], requires_grad=True)

    loss: Tensor = entropy_separation_loss(torch.softmax(logits, dim=1), 0.5, 0.1, EPSILON)
    loss.backward()

    assert torch.isfinite(logits.grad).all()


def test_total_loss_combination() -> None:
    cross_entropy_value: Tensor = torch.tensor(1.0)
    clustering_value: Tensor = torch.tensor(2.0)
    separation_value: Tensor = torch.tensor(-0.5)

    combined: Tensor = total_loss(cross_entropy_value, clustering_value, separation_value, domain_adaptation_weight=0.1)

    assert combined.item() == pytest.approx(1.0 + 0.1 * 1.5)
