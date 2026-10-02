import logging
import os
from pathlib import Path

import numpy as np
import pytest
from numpy import ndarray
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from astrouda.evaluation import (
    AggregatedReport,
    ClassificationReport,
    aggregate_reports,
    comparison_table,
    compute_classification_report,
    load_report,
    plot_confusion_matrix,
    plot_roc_curves,
    save_report,
)


def one_hot_probabilities(predicted_classes: ndarray, number_of_classes: int) -> ndarray:
    return np.eye(number_of_classes)[predicted_classes]


def known_three_class_report() -> ClassificationReport:
    labels: ndarray = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    predicted_classes: ndarray = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
    return compute_classification_report(labels, predicted_classes, one_hot_probabilities(predicted_classes, 3), 3)


def test_perfect_classifier() -> None:
    labels: ndarray = np.array([0, 1, 2, 0, 1, 2])
    report: ClassificationReport = compute_classification_report(labels, labels, one_hot_probabilities(labels, 3), 3)

    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.macro_roc_auc == 1.0
    assert report.confusion_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert report.normalised_confusion_matrix == [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]


def test_all_wrong_classifier() -> None:
    labels: ndarray = np.array([0, 1, 2, 0, 1, 2])
    predicted_classes: ndarray = (labels + 1) % 3
    report: ClassificationReport = compute_classification_report(
        labels, predicted_classes, one_hot_probabilities(predicted_classes, 3), 3
    )

    assert report.accuracy == 0.0
    assert report.precision == [0.0, 0.0, 0.0]
    assert report.recall == [0.0, 0.0, 0.0]
    assert report.f1 == [0.0, 0.0, 0.0]


def test_known_three_class_confusion() -> None:
    report: ClassificationReport = known_three_class_report()

    assert report.accuracy == pytest.approx(0.7)
    assert report.confusion_matrix == [[3, 1, 0], [0, 2, 1], [1, 0, 2]]
    assert report.support == [4, 3, 3]
    assert report.precision == pytest.approx([3 / 4, 2 / 3, 2 / 3])
    assert report.recall == pytest.approx([3 / 4, 2 / 3, 2 / 3])
    assert report.f1 == pytest.approx([3 / 4, 2 / 3, 2 / 3])
    assert report.macro_f1 == pytest.approx((3 / 4 + 2 / 3 + 2 / 3) / 3)
    assert report.weighted_f1 == pytest.approx((4 * 3 / 4 + 3 * 2 / 3 + 3 * 2 / 3) / 10)
    assert report.normalised_confusion_matrix[0] == pytest.approx([0.75, 0.25, 0.0])


def test_agreement_with_sklearn() -> None:
    generator: np.random.Generator = np.random.default_rng(0)
    labels: ndarray = generator.integers(0, 4, size=200)
    probabilities: ndarray = generator.dirichlet(np.ones(4), size=200)
    predicted_classes: ndarray = probabilities.argmax(axis=1)

    report: ClassificationReport = compute_classification_report(labels, predicted_classes, probabilities, 4)
    expected_precision, expected_recall, expected_f1, _ = precision_recall_fscore_support(
        labels, predicted_classes, average="macro"
    )

    assert report.accuracy == pytest.approx(accuracy_score(labels, predicted_classes))
    assert report.macro_precision == pytest.approx(expected_precision)
    assert report.macro_recall == pytest.approx(expected_recall)
    assert report.macro_f1 == pytest.approx(expected_f1)


def test_binary_auc_hand_value() -> None:
    labels: ndarray = np.array([0, 0, 1, 1])
    positive_scores: ndarray = np.array([0.1, 0.4, 0.35, 0.8])
    probabilities: ndarray = np.stack([1 - positive_scores, positive_scores], axis=1)

    report: ClassificationReport = compute_classification_report(labels, probabilities.argmax(axis=1), probabilities, 2)

    assert report.macro_roc_auc == pytest.approx(0.75)
    assert report.roc_auc[1] == pytest.approx(0.75)
    assert report.roc_curves[1].false_positive_rates[0] == 0.0
    assert report.roc_curves[1].true_positive_rates[-1] == 1.0


def test_missing_class_is_handled(caplog: pytest.LogCaptureFixture) -> None:
    labels: ndarray = np.array([0, 0, 1, 1])
    predicted_classes: ndarray = np.array([0, 1, 1, 1])
    probabilities: ndarray = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])

    with caplog.at_level(logging.WARNING):
        report: ClassificationReport = compute_classification_report(labels, predicted_classes, probabilities, 3)

    assert "absent" in caplog.text
    assert report.support == [2, 2, 0]
    assert report.recall[2] == 0.0
    assert report.f1[2] == 0.0
    assert report.roc_auc[2] is None
    assert report.roc_curves[2] is None
    assert report.macro_roc_auc == pytest.approx(np.mean([report.roc_auc[0], report.roc_auc[1]]))
    assert report.macro_recall == pytest.approx(np.mean(report.recall[:2]))
    assert report.normalised_confusion_matrix[2] == [0.0, 0.0, 0.0]
    assert not np.isnan(np.array(report.normalised_confusion_matrix)).any()


def test_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError):
        compute_classification_report(np.array([0, 1]), np.array([0, 1]), np.ones((2, 3)), 2)


def test_aggregation_matches_numpy() -> None:
    generator: np.random.Generator = np.random.default_rng(1)
    reports: list[ClassificationReport] = []
    for _ in range(5):
        labels: ndarray = generator.integers(0, 3, size=100)
        probabilities: ndarray = generator.dirichlet(np.ones(3), size=100)
        reports.append(compute_classification_report(labels, probabilities.argmax(axis=1), probabilities, 3))

    aggregated: AggregatedReport = aggregate_reports(reports)
    accuracies: list[float] = [report.accuracy for report in reports]
    class_one_recalls: list[float] = [report.recall[1] for report in reports]
    macro_aucs: list[float] = [report.macro_roc_auc for report in reports]

    assert aggregated.number_of_reports == 5
    assert aggregated.accuracy.mean == pytest.approx(np.mean(accuracies))
    assert aggregated.accuracy.std == pytest.approx(np.std(accuracies))
    assert aggregated.recall[1].mean == pytest.approx(np.mean(class_one_recalls))
    assert aggregated.macro_roc_auc.std == pytest.approx(np.std(macro_aucs))


def test_aggregate_rejects_empty() -> None:
    with pytest.raises(ValueError):
        aggregate_reports([])


def test_comparison_table_contents() -> None:
    labels: ndarray = np.array([0, 1, 2, 0, 1, 2])
    perfect: ClassificationReport = compute_classification_report(labels, labels, one_hot_probabilities(labels, 3), 3)
    wrong_classes: ndarray = (labels + 1) % 3
    wrong: ClassificationReport = compute_classification_report(
        labels, wrong_classes, one_hot_probabilities(wrong_classes, 3), 3
    )

    table: str = comparison_table(aggregate_reports([perfect, perfect]), aggregate_reports([wrong, wrong]), "LSST Y10")

    assert "LSST Y10" in table
    assert "| Accuracy | 1.000 ± 0.000 | 0.000 ± 0.000 |" in table
    assert "Domain adaptation" in table and "Source only" in table


def test_json_round_trip(tmp_path: Path) -> None:
    report: ClassificationReport = known_three_class_report()
    output_path: str = str(tmp_path / "report.json")

    save_report(report, output_path)
    loaded: ClassificationReport = load_report(output_path)

    assert loaded == report


def test_plots_produce_files(tmp_path: Path) -> None:
    report: ClassificationReport = known_three_class_report()
    confusion_path: str = str(tmp_path / "confusion.png")
    roc_path: str = str(tmp_path / "roc.png")

    plot_confusion_matrix(report, confusion_path, class_names=["a", "b", "c"])
    plot_roc_curves(report, roc_path)

    assert os.path.getsize(confusion_path) > 0
    assert os.path.getsize(roc_path) > 0


def test_plot_rejects_wrong_class_name_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_confusion_matrix(known_three_class_report(), str(tmp_path / "x.png"), class_names=["a"])
