import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from astrouda.evaluation.report import ClassificationReport

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MeanStd:
    mean: float
    std: float


@dataclass
class AggregatedReport:
    """
    Mean and population standard deviation (numpy default, ddof=0) over reports from different seeds.
    AUC entries are None when no report defines them, otherwise they aggregate the reports that do.
    """

    number_of_reports: int
    number_of_classes: int
    accuracy: MeanStd
    macro_precision: MeanStd
    macro_recall: MeanStd
    macro_f1: MeanStd
    macro_roc_auc: Optional[MeanStd]
    precision: list[MeanStd]
    recall: list[MeanStd]
    f1: list[MeanStd]
    roc_auc: list[Optional[MeanStd]]


def _mean_std(values: list[Optional[float]]) -> Optional[MeanStd]:
    defined_values: list[float] = [value for value in values if value is not None]
    if not defined_values:
        return None
    return MeanStd(mean=float(np.mean(defined_values)), std=float(np.std(defined_values)))


def _required_mean_std(values: list[float]) -> MeanStd:
    aggregated: Optional[MeanStd] = _mean_std(list(values))
    assert aggregated is not None
    return aggregated


def aggregate_reports(reports: list[ClassificationReport]) -> AggregatedReport:
    if not reports:
        raise ValueError("Cannot aggregate an empty list of reports")
    number_of_classes: int = reports[0].number_of_classes
    if any(report.number_of_classes != number_of_classes for report in reports):
        raise ValueError("All reports must have the same number of classes")
    logger.debug(f"Aggregating {len(reports)} reports over {number_of_classes} classes")

    return AggregatedReport(
        number_of_reports=len(reports),
        number_of_classes=number_of_classes,
        accuracy=_required_mean_std([report.accuracy for report in reports]),
        macro_precision=_required_mean_std([report.macro_precision for report in reports]),
        macro_recall=_required_mean_std([report.macro_recall for report in reports]),
        macro_f1=_required_mean_std([report.macro_f1 for report in reports]),
        macro_roc_auc=_mean_std([report.macro_roc_auc for report in reports]),
        precision=[_required_mean_std([report.precision[class_index] for report in reports]) for class_index in range(number_of_classes)],
        recall=[_required_mean_std([report.recall[class_index] for report in reports]) for class_index in range(number_of_classes)],
        f1=[_required_mean_std([report.f1[class_index] for report in reports]) for class_index in range(number_of_classes)],
        roc_auc=[_mean_std([report.roc_auc[class_index] for report in reports]) for class_index in range(number_of_classes)],
    )


def _format_cell(value: Optional[MeanStd]) -> str:
    if value is None:
        return "n/a"
    return f"{value.mean:.3f} ± {value.std:.3f}"


def comparison_table(adapted: AggregatedReport, source_only: AggregatedReport, domain_name: str) -> str:
    metric_rows: list[tuple[str, Optional[MeanStd], Optional[MeanStd]]] = [
        ("Accuracy", adapted.accuracy, source_only.accuracy),
        ("Macro precision", adapted.macro_precision, source_only.macro_precision),
        ("Macro recall", adapted.macro_recall, source_only.macro_recall),
        ("Macro F1", adapted.macro_f1, source_only.macro_f1),
        ("Macro ROC-AUC", adapted.macro_roc_auc, source_only.macro_roc_auc),
    ]
    logger.debug(f"Rendering comparison table for {domain_name}")

    table_lines: list[str] = [
        f"Target domain: {domain_name} (mean ± std over {adapted.number_of_reports} DA runs, {source_only.number_of_reports} source-only runs)",
        "",
        "| Metric | Domain adaptation | Source only |",
        "| --- | --- | --- |",
    ]
    for metric_name, adapted_value, source_only_value in metric_rows:
        table_lines.append(f"| {metric_name} | {_format_cell(adapted_value)} | {_format_cell(source_only_value)} |")

    return "\n".join(table_lines)
