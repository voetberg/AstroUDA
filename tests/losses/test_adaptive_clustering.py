import math

import pytest
import torch
from torch import Tensor

from astrouda.losses import ProbabilityBank, adaptive_clustering_loss, ordered_top_k_similarity_labels

EPSILON: float = 1e-8


def test_order_matters_for_full_top_k() -> None:
    probabilities_a: Tensor = torch.tensor([[0.5, 0.3, 0.2]])
    probabilities_b: Tensor = torch.tensor([[0.5, 0.2, 0.3], [0.6, 0.25, 0.15]])

    labels: Tensor = ordered_top_k_similarity_labels(probabilities_a, probabilities_b, top_k=3)

    assert labels.tolist() == [[0.0, 1.0]]


def test_top_one_ignores_lower_ranks() -> None:
    probabilities_a: Tensor = torch.tensor([[0.5, 0.3, 0.2]])
    probabilities_b: Tensor = torch.tensor([[0.5, 0.2, 0.3], [0.1, 0.8, 0.1]])

    labels: Tensor = ordered_top_k_similarity_labels(probabilities_a, probabilities_b, top_k=1)

    assert labels.tolist() == [[1.0, 0.0]]


def test_top_two_requires_same_first_two_ranks() -> None:
    probabilities_a: Tensor = torch.tensor([[0.5, 0.3, 0.2]])
    probabilities_b: Tensor = torch.tensor([[0.5, 0.35, 0.15], [0.3, 0.5, 0.2], [0.5, 0.2, 0.3]])

    labels: Tensor = ordered_top_k_similarity_labels(probabilities_a, probabilities_b, top_k=2)

    assert labels.tolist() == [[1.0, 0.0, 0.0]]


def test_full_k_is_not_trivially_all_ones() -> None:
    torch.manual_seed(0)
    probabilities: Tensor = torch.softmax(torch.randn(50, 3), dim=1)

    labels: Tensor = ordered_top_k_similarity_labels(probabilities, probabilities, top_k=3)

    assert 0.0 < labels.mean().item() < 1.0
    assert torch.equal(labels.diagonal(), torch.ones(50))


def test_ten_classes_top_seven_codes_do_not_collide() -> None:
    probabilities_a: Tensor = torch.softmax(torch.arange(10.0)[None, :], dim=1)
    probabilities_b: Tensor = torch.softmax(torch.cat([torch.arange(8.0), torch.tensor([9.0, 8.0])])[None, :], dim=1)

    labels: Tensor = ordered_top_k_similarity_labels(probabilities_a, probabilities_b, top_k=7)

    assert labels.item() == 0.0


def test_loss_hand_checked_on_tiny_case() -> None:
    bank_probabilities: Tensor = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    target_probabilities: Tensor = torch.tensor([[0.8, 0.2]])

    loss: Tensor = adaptive_clustering_loss(target_probabilities, bank_probabilities, top_k=1, probability_epsilon=EPSILON)

    expected_loss: float = (-math.log(0.74) - math.log(1 - 0.26)) / 2
    assert loss.item() == pytest.approx(expected_loss, rel=1e-6)


def test_gradient_flows_only_through_target() -> None:
    bank_probabilities: Tensor = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
    target_probabilities: Tensor = torch.tensor([[0.8, 0.2]], requires_grad=True)

    loss: Tensor = adaptive_clustering_loss(target_probabilities, bank_probabilities, top_k=1, probability_epsilon=EPSILON)
    loss.backward()

    assert target_probabilities.grad is not None and torch.isfinite(target_probabilities.grad).all()
    assert target_probabilities.grad.abs().sum().item() > 0
    assert bank_probabilities.grad is None


def test_empty_bank_gives_zero_loss_that_backpropagates() -> None:
    target_probabilities: Tensor = torch.tensor([[0.8, 0.2]], requires_grad=True)
    empty_bank: ProbabilityBank = ProbabilityBank(capacity=4, number_of_classes=2)

    loss: Tensor = adaptive_clustering_loss(target_probabilities, empty_bank.probabilities, top_k=1, probability_epsilon=EPSILON)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(target_probabilities.grad, torch.zeros(1, 2))


def test_bank_evicts_oldest_first_and_respects_capacity() -> None:
    bank: ProbabilityBank = ProbabilityBank(capacity=3, number_of_classes=2)

    bank.add(torch.tensor([[1.0, 0.0], [0.9, 0.1]]))
    bank.add(torch.tensor([[0.8, 0.2], [0.7, 0.3]]))

    assert len(bank) == 3
    assert bank.probabilities[:, 0].tolist() == pytest.approx([0.9, 0.8, 0.7])


def test_bank_add_larger_than_capacity_keeps_newest() -> None:
    bank: ProbabilityBank = ProbabilityBank(capacity=2, number_of_classes=2)

    bank.add(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]))

    assert bank.probabilities[:, 0].tolist() == pytest.approx([0.9, 0.8])


def test_bank_is_detached() -> None:
    bank: ProbabilityBank = ProbabilityBank(capacity=4, number_of_classes=2)
    probabilities: Tensor = torch.softmax(torch.randn(2, 2, requires_grad=True), dim=1)

    bank.add(probabilities)

    assert not bank.probabilities.requires_grad


def test_bank_state_dict_round_trip() -> None:
    bank: ProbabilityBank = ProbabilityBank(capacity=4, number_of_classes=2)
    bank.add(torch.tensor([[0.6, 0.4], [0.2, 0.8]]))
    restored_bank: ProbabilityBank = ProbabilityBank(capacity=1, number_of_classes=2)

    restored_bank.load_state_dict(bank.state_dict())

    assert restored_bank.capacity == 4
    assert torch.equal(restored_bank.probabilities, bank.probabilities)


def test_bank_stored_before_loss_means_no_self_pairs() -> None:
    target_probabilities: Tensor = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    bank: ProbabilityBank = ProbabilityBank(capacity=8, number_of_classes=2)
    bank.add(torch.tensor([[0.9, 0.1]]))

    loss: Tensor = adaptive_clustering_loss(target_probabilities, bank.probabilities, top_k=1, probability_epsilon=EPSILON)
    pair_count_before_update: int = len(bank) * target_probabilities.shape[0]
    bank.add(target_probabilities)

    assert pair_count_before_update == 2
    assert len(bank) == 3
    assert torch.isfinite(loss)
