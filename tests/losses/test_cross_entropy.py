import math

import pytest
import torch
import torch.nn.functional as functional
from torch import Tensor

from astrouda.losses import compute_class_weights, weighted_cross_entropy_loss


def test_class_weights_follow_equation_four() -> None:
    source_labels: Tensor = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])

    class_weights: Tensor = compute_class_weights(source_labels, number_of_classes=3)

    expected_weights: Tensor = torch.tensor([8 / (3 * 4), 8 / (3 * 2), 8 / (3 * 2)])
    assert torch.allclose(class_weights, expected_weights)


def test_missing_class_gets_zero_weight_and_stays_finite() -> None:
    source_labels: Tensor = torch.tensor([0, 0, 2])

    class_weights: Tensor = compute_class_weights(source_labels, number_of_classes=3)

    assert torch.isfinite(class_weights).all()
    assert class_weights[1].item() == 0.0
    assert class_weights[0].item() == pytest.approx(3 / (3 * 2))


def test_weighted_loss_matches_hand_computed_value() -> None:
    logits: Tensor = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    labels: Tensor = torch.tensor([0, 1])
    class_weights: Tensor = torch.tensor([1.0, 3.0])

    loss: Tensor = weighted_cross_entropy_loss(logits, labels, class_weights)

    first_sample_loss: float = math.log(1 + math.exp(-2.0))
    second_sample_loss: float = math.log(1 + math.exp(-1.0))
    expected_loss: float = (1.0 * first_sample_loss + 3.0 * second_sample_loss) / (1.0 + 3.0)
    assert loss.item() == pytest.approx(expected_loss, rel=1e-6)


def test_weighted_loss_matches_torch_module() -> None:
    torch.manual_seed(0)
    logits: Tensor = torch.randn(16, 3)
    labels: Tensor = torch.randint(0, 3, (16,))
    class_weights: Tensor = compute_class_weights(labels, number_of_classes=3)

    loss: Tensor = weighted_cross_entropy_loss(logits, labels, class_weights)

    reference_loss: Tensor = torch.nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
    assert torch.allclose(loss, reference_loss)


def test_balanced_classes_reduce_to_plain_cross_entropy() -> None:
    torch.manual_seed(0)
    logits: Tensor = torch.randn(9, 3)
    labels: Tensor = torch.tensor([0, 1, 2] * 3)
    class_weights: Tensor = compute_class_weights(labels, number_of_classes=3)

    loss: Tensor = weighted_cross_entropy_loss(logits, labels, class_weights)

    assert torch.allclose(class_weights, torch.ones(3))
    assert torch.allclose(loss, functional.cross_entropy(logits, labels))
