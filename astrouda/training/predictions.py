from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionSet:
    """
    Model outputs on one split ('validation' or 'test'), in loader order.

    *_labels: int64, shape (N,). *_predicted_classes: int64, shape (N,).
    *_probabilities: float32 softmax, shape (N, number_of_classes).
    Target labels are used for evaluation only. The target domain index follows the validation rule:
    1 in domain adaptation mode, 0 in the source-only baseline.
    """

    source_labels: np.ndarray
    source_predicted_classes: np.ndarray
    source_probabilities: np.ndarray
    target_labels: np.ndarray
    target_predicted_classes: np.ndarray
    target_probabilities: np.ndarray

    @property
    def source_accuracy(self) -> float:
        return float((self.source_labels == self.source_predicted_classes).mean())

    @property
    def target_accuracy(self) -> float:
        return float((self.target_labels == self.target_predicted_classes).mean())
