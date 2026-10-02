import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from numpy import ndarray
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class RocCurve:
    false_positive_rates: list[float]
    true_positive_rates: list[float]


@dataclass
class ClassificationReport:
    """
    Metrics of one evaluation, as plain Python numbers and lists so it serialises to JSON directly.

    Classes absent from the labels (support 0) have no defined recall, so:
    recall and F1 are reported as 0.0 (precision is 0.0 as well, since every prediction of them is wrong),
    no one-vs-rest ROC exists, so their AUC and ROC curve are None and they are excluded from macro AUC,
    and they are excluded from the macro averages of precision, recall and F1. A warning is logged.
    Weighted averages are unaffected because their weight is zero.
    The binary case (K == 2) reports the AUC of class 1 as the macro AUC, which equals the class 0 AUC.
    """

    number_of_classes: int
    number_of_samples: int
    accuracy: float
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    normalised_confusion_matrix: list[list[float]]
    roc_auc: list[Optional[float]]
    macro_roc_auc: Optional[float]
    roc_curves: list[Optional[RocCurve]]

    def to_dictionary(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dictionary(cls, dictionary: dict[str, Any]) -> "ClassificationReport":
        report_fields: dict[str, Any] = dict(dictionary)
        report_fields["roc_curves"] = [
            None if curve is None else RocCurve(**curve) for curve in dictionary["roc_curves"]
        ]
        return cls(**report_fields)


def _validate_inputs(labels: ndarray, predicted_classes: ndarray, probabilities: ndarray, number_of_classes: int) -> None:
    if labels.ndim != 1 or predicted_classes.shape != labels.shape:
        raise ValueError(f"labels and predicted_classes must be 1D and equal in shape, got {labels.shape} and {predicted_classes.shape}")
    if probabilities.shape != (labels.shape[0], number_of_classes):
        raise ValueError(f"probabilities must have shape {(labels.shape[0], number_of_classes)}, got {probabilities.shape}")
    if labels.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty set of samples")
    if labels.min() < 0 or labels.max() >= number_of_classes or predicted_classes.min() < 0 or predicted_classes.max() >= number_of_classes:
        raise ValueError(f"labels and predicted_classes must lie in [0, {number_of_classes - 1}]")


def compute_classification_report(
    labels: ndarray,
    predicted_classes: ndarray,
    probabilities: ndarray,
    number_of_classes: int,
) -> ClassificationReport:
    _validate_inputs(labels, predicted_classes, probabilities, number_of_classes)
    class_indices: list[int] = list(range(number_of_classes))
    logger.debug(f"Evaluating {labels.shape[0]} samples over {number_of_classes} classes")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted_classes, labels=class_indices, average=None, zero_division=0
    )
    present_mask: ndarray = support > 0
    absent_classes: list[int] = [int(index) for index in np.flatnonzero(~present_mask)]
    if absent_classes:
        logger.warning(f"Classes {absent_classes} are absent from the labels: zero recall/F1, no AUC, excluded from macro averages")

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predicted_classes, labels=class_indices, average="weighted", zero_division=0
    )

    confusion_counts: ndarray = confusion_matrix(labels, predicted_classes, labels=class_indices)
    row_totals: ndarray = confusion_counts.sum(axis=1, keepdims=True)
    normalised_confusion: ndarray = np.divide(
        confusion_counts, row_totals, out=np.zeros(confusion_counts.shape, dtype=float), where=row_totals > 0
    )
    logger.debug(f"Confusion matrix:\n{confusion_counts}")

    per_class_auc: list[Optional[float]] = []
    per_class_curves: list[Optional[RocCurve]] = []
    for class_index in class_indices:
        binary_labels: ndarray = (labels == class_index).astype(int)
        has_both_outcomes: bool = 0 < binary_labels.sum() < binary_labels.shape[0]
        if not has_both_outcomes:
            logger.debug(f"Class {class_index} has a single outcome in the labels, no ROC")
            per_class_auc.append(None)
            per_class_curves.append(None)
            continue

        class_scores: ndarray = probabilities[:, class_index]
        false_positive_rates, true_positive_rates, _ = roc_curve(binary_labels, class_scores)
        area_under_curve: float = float(roc_auc_score(binary_labels, class_scores))
        logger.debug(f"Class {class_index} one-vs-rest AUC={area_under_curve:.4f}")
        per_class_auc.append(area_under_curve)
        per_class_curves.append(RocCurve(false_positive_rates.tolist(), true_positive_rates.tolist()))

    defined_aucs: list[float] = [value for value in per_class_auc if value is not None]
    macro_roc_auc: Optional[float] = float(np.mean(defined_aucs)) if defined_aucs else None

    if number_of_classes == 2 and per_class_auc[1] is not None:
        macro_roc_auc = per_class_auc[1]
    logger.debug(f"Macro AUC={macro_roc_auc}")

    return ClassificationReport(
        number_of_classes=number_of_classes,
        number_of_samples=int(labels.shape[0]),
        accuracy=float(np.mean(labels == predicted_classes)),
        precision=precision.tolist(),
        recall=recall.tolist(),
        f1=f1.tolist(),
        support=[int(value) for value in support],
        macro_precision=float(precision[present_mask].mean()),
        macro_recall=float(recall[present_mask].mean()),
        macro_f1=float(f1[present_mask].mean()),
        weighted_precision=float(weighted_precision),
        weighted_recall=float(weighted_recall),
        weighted_f1=float(weighted_f1),
        confusion_matrix=confusion_counts.tolist(),
        normalised_confusion_matrix=normalised_confusion.tolist(),
        roc_auc=per_class_auc,
        macro_roc_auc=macro_roc_auc,
        roc_curves=per_class_curves,
    )


def save_report(report: ClassificationReport, output_path: str) -> None:
    with open(output_path, "w") as output_file:
        json.dump(report.to_dictionary(), output_file, indent=2)
    logger.debug(f"Saved classification report to {output_path}")


def load_report(input_path: str) -> ClassificationReport:
    with open(input_path, "r") as input_file:
        dictionary: dict[str, Any] = json.load(input_file)
    logger.debug(f"Loaded classification report from {input_path}")
    return ClassificationReport.from_dictionary(dictionary)
