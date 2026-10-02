import json
import os
from dataclasses import asdict, replace
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astrouda.config import Config
from astrouda.evaluation import (
    AggregatedReport,
    ClassificationReport,
    aggregate_reports,
    comparison_table,
    plot_confusion_matrix,
)
from astrouda.experiment.runs import (
    ADAPTED_MODE,
    MODES,
    SOURCE_ONLY_MODE,
    RunSpecification,
    class_names_for,
    enumerate_run_specifications,
)
from astrouda.logging_utils import get_logger

DOMAIN_LABELS: tuple[str, str] = ("target", "source")
BAR_METRIC_NAMES: tuple[str, ...] = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc")


class MissingRunsError(RuntimeError):
    pass


def _load_run_report(specification: RunSpecification, domain_label: str) -> ClassificationReport:
    with open(specification.report_path, "r") as report_file:
        run_record: dict[str, Any] = json.load(report_file)

    return ClassificationReport.from_dictionary(run_record[domain_label])


def _plot_metric_bars(aggregates: dict[str, AggregatedReport], output_path: str, title: str) -> None:
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    bar_width: float = 0.38

    for mode_offset, mode in enumerate(MODES):
        means: list[float] = []
        standard_deviations: list[float] = []
        for metric_name in BAR_METRIC_NAMES:
            metric_value = getattr(aggregates[mode], metric_name)
            means.append(0.0 if metric_value is None else metric_value.mean)
            standard_deviations.append(0.0 if metric_value is None else metric_value.std)
        axes.bar(
            np.arange(len(BAR_METRIC_NAMES)) + (mode_offset - 0.5) * bar_width,
            means,
            bar_width,
            yerr=standard_deviations,
            capsize=3,
            label="Domain adaptation" if mode == ADAPTED_MODE else "Source only",
        )

    axes.set_xticks(range(len(BAR_METRIC_NAMES)), labels=[name.replace("_", " ") for name in BAR_METRIC_NAMES], rotation=30, ha="right")
    axes.set_ylim(0.0, 1.0)
    axes.set_ylabel("Mean over seeds (error bar: population std)")
    axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _mean_confusion_report(reports: list[ClassificationReport]) -> ClassificationReport:
    "A report whose confusion matrices are the seed means, for plotting only."
    mean_counts: np.ndarray = np.mean([report.confusion_matrix for report in reports], axis=0)
    mean_normalised: np.ndarray = np.mean([report.normalised_confusion_matrix for report in reports], axis=0)

    return replace(
        reports[0],
        confusion_matrix=np.rint(mean_counts).astype(int).tolist(),
        normalised_confusion_matrix=mean_normalised.tolist(),
    )


def aggregate_experiment(config: Config) -> dict[str, Any]:
    "Aggregate every saved run report, write aggregate.json, comparison_table.md and plots, print the tables."
    logger = get_logger(__name__, config)
    specifications: list[RunSpecification] = enumerate_run_specifications(config)

    missing_specifications: list[RunSpecification] = [
        specification for specification in specifications if not os.path.exists(specification.report_path)
    ]
    if missing_specifications:
        missing_description: str = "; ".join(
            f"run {specification.run_index} ({specification.mode}, seed {specification.seed}): {specification.report_path}"
            for specification in missing_specifications
        )
        raise MissingRunsError(
            f"{len(missing_specifications)} of {len(specifications)} runs have no report.json: {missing_description}"
        )

    aggregate_record: dict[str, Any] = {"number_of_seeds": config.number_of_seeds, "seeds": [], "domains": {}}
    aggregate_record["seeds"] = sorted({specification.seed for specification in specifications})
    table_sections: list[str] = []
    class_names: Optional[list[str]] = class_names_for(config)

    for domain_label, domain_name in zip(DOMAIN_LABELS, (config.target_domain_name, config.source_domain_name)):
        reports_by_mode: dict[str, list[ClassificationReport]] = {
            mode: [
                _load_run_report(specification, domain_label)
                for specification in specifications
                if specification.mode == mode
            ]
            for mode in MODES
        }
        aggregates: dict[str, AggregatedReport] = {mode: aggregate_reports(reports_by_mode[mode]) for mode in MODES}
        logger.debug(f"Aggregated {domain_label} reports: {[(mode, len(reports_by_mode[mode])) for mode in MODES]}")

        aggregate_record["domains"][domain_label] = {
            "domain_name": domain_name,
            ADAPTED_MODE: asdict(aggregates[ADAPTED_MODE]),
            SOURCE_ONLY_MODE: asdict(aggregates[SOURCE_ONLY_MODE]),
        }
        if domain_label == "target":
            table_sections.append(comparison_table(aggregates[ADAPTED_MODE], aggregates[SOURCE_ONLY_MODE], domain_name))

        _plot_metric_bars(
            aggregates,
            os.path.join(config.output_directory, f"aggregate_metrics_{domain_label}.png"),
            f"{domain_name} test metrics",
        )
        for mode in MODES:
            plot_confusion_matrix(
                _mean_confusion_report(reports_by_mode[mode]),
                os.path.join(config.output_directory, f"aggregate_confusion_matrix_{domain_label}_{mode}.png"),
                class_names,
                title=f"{domain_name} ({mode}), mean over {config.number_of_seeds} seeds",
            )

    os.makedirs(config.output_directory, exist_ok=True)
    with open(os.path.join(config.output_directory, "aggregate.json"), "w") as aggregate_file:
        json.dump(aggregate_record, aggregate_file, indent=2)

    table_text: str = "\n\n".join(table_sections)
    with open(os.path.join(config.output_directory, "comparison_table.md"), "w") as table_file:
        table_file.write(table_text + "\n")
    print(table_text)
    logger.debug(f"Wrote aggregate.json and comparison_table.md to {config.output_directory}")

    return aggregate_record


def aggregate_command(config_path: str, override_texts: list[str]) -> dict[str, Any]:
    from astrouda.cli import parse_override

    overrides: dict[str, Any] = dict(parse_override(override_text) for override_text in override_texts)

    return aggregate_experiment(Config(config_path, **overrides))
