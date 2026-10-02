import logging
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from astrouda.evaluation.report import ClassificationReport

logger: logging.Logger = logging.getLogger(__name__)


def _resolve_class_names(class_names: Optional[list[str]], number_of_classes: int) -> list[str]:
    if class_names is None:
        return [str(class_index) for class_index in range(number_of_classes)]
    if len(class_names) != number_of_classes:
        raise ValueError(f"Expected {number_of_classes} class names, got {len(class_names)}")
    return list(class_names)


def plot_confusion_matrix(
    report: ClassificationReport,
    output_path: str,
    class_names: Optional[list[str]] = None,
    title: str = "Confusion matrix",
) -> None:
    resolved_names: list[str] = _resolve_class_names(class_names, report.number_of_classes)
    normalised_matrix: np.ndarray = np.array(report.normalised_confusion_matrix)
    logger.debug(f"Plotting confusion matrix to {output_path}")

    figure: Figure
    figure, axes = plt.subplots(figsize=(1.2 * report.number_of_classes + 3, 1.2 * report.number_of_classes + 2.5))
    image = axes.imshow(normalised_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    figure.colorbar(image, ax=axes, label="Fraction of true class")

    axes.set_xticks(range(report.number_of_classes), labels=resolved_names, rotation=45, ha="right")
    axes.set_yticks(range(report.number_of_classes), labels=resolved_names)
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    axes.set_title(title)

    for row_index in range(report.number_of_classes):
        for column_index in range(report.number_of_classes):
            cell_fraction: float = normalised_matrix[row_index, column_index]
            axes.text(
                column_index,
                row_index,
                f"{cell_fraction:.2f}\n({report.confusion_matrix[row_index][column_index]})",
                ha="center",
                va="center",
                color="black" if cell_fraction > 0.5 else "white",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_roc_curves(
    report: ClassificationReport,
    output_path: str,
    class_names: Optional[list[str]] = None,
    title: str = "ROC curves (one vs rest)",
) -> None:
    resolved_names: list[str] = _resolve_class_names(class_names, report.number_of_classes)
    logger.debug(f"Plotting ROC curves to {output_path}")

    figure: Figure
    figure, axes = plt.subplots(figsize=(5.5, 5))
    axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Chance")

    for class_index, curve in enumerate(report.roc_curves):
        if curve is None:
            logger.debug(f"Skipping class {class_index} in ROC plot, no curve")
            continue
        axes.plot(
            curve.false_positive_rates,
            curve.true_positive_rates,
            label=f"{resolved_names[class_index]} (AUC {report.roc_auc[class_index]:.3f})",
        )

    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(title)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
