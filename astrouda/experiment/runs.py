import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from astrouda.config import Config
from astrouda.evaluation import (
    ClassificationReport,
    compute_classification_report,
    plot_confusion_matrix,
    plot_roc_curves,
)
from astrouda.exit_codes import EX_TEMPFAIL
from astrouda.logging_utils import get_logger
from astrouda.training import PredictionSet, Trainer

ADAPTED_MODE: str = "adapted"
SOURCE_ONLY_MODE: str = "source_only"
MODES: tuple[str, str] = (ADAPTED_MODE, SOURCE_ONLY_MODE)
REPORT_FILE_NAME: str = "report.json"


class RunStoppedForTimeLimit(RuntimeError):
    "Raised by run_single when training stopped at the wall time limit. Rerunning the same run continues it."


@dataclass(frozen=True)
class RunSpecification:
    run_index: int
    seed_index: int
    seed: int
    mode: str
    output_directory: str

    @property
    def enable_domain_adaptation(self) -> bool:
        return self.mode == ADAPTED_MODE

    @property
    def report_path(self) -> str:
        return os.path.join(self.output_directory, REPORT_FILE_NAME)


def build_run_specification(config: Config, run_index: int) -> RunSpecification:
    "run_index = seed_index * 2 + (0 for adapted, 1 for source_only)."
    total_runs: int = len(MODES) * config.number_of_seeds
    if not 0 <= run_index < total_runs:
        raise ValueError(f"run_index must be in [0, {total_runs}), got {run_index}")

    seed_index: int = run_index // len(MODES)
    mode: str = MODES[run_index % len(MODES)]
    seed: int = config.random_seed + seed_index

    return RunSpecification(
        run_index=run_index,
        seed_index=seed_index,
        seed=seed,
        mode=mode,
        output_directory=os.path.join(config.output_directory, f"seed_{seed}", mode),
    )


def decode_run_index(config: Config, run_index: int) -> tuple[int, str]:
    specification: RunSpecification = build_run_specification(config, run_index)

    return specification.seed_index, specification.mode


def enumerate_run_specifications(config: Config) -> list[RunSpecification]:
    return [build_run_specification(config, run_index) for run_index in range(len(MODES) * config.number_of_seeds)]


def classification_reports_from_predictions(
    prediction_set: PredictionSet, number_of_classes: int
) -> tuple[ClassificationReport, ClassificationReport]:
    "Returns (target report, source report)."
    target_report: ClassificationReport = compute_classification_report(
        prediction_set.target_labels,
        prediction_set.target_predicted_classes,
        prediction_set.target_probabilities,
        number_of_classes,
    )
    source_report: ClassificationReport = compute_classification_report(
        prediction_set.source_labels,
        prediction_set.source_predicted_classes,
        prediction_set.source_probabilities,
        number_of_classes,
    )

    return target_report, source_report


def class_names_for(config: Config) -> Optional[list[str]]:
    if config.dataset_name.startswith("gz2") and len(config.gz2_class_names) == config.number_of_classes:
        return list(config.gz2_class_names)

    return None


def run_single(base_config: Config, run_index: int) -> Optional[RunSpecification]:
    "Train, reload the best weights, evaluate on the test split and save. Returns None when skipped. Raises RunStoppedForTimeLimit."
    specification: RunSpecification = build_run_specification(base_config, run_index)
    logger = get_logger(__name__, base_config)

    if os.path.exists(specification.report_path):
        if not base_config.overwrite_existing_runs:
            logger.info(f"Skipping run {run_index} ({specification.mode}, seed {specification.seed}): {specification.report_path} exists")

            return None

        checkpoint_path: str = os.path.join(specification.output_directory, "checkpoint.pt")
        logger.info(f"Overwriting run {run_index}: removing {specification.report_path} and {checkpoint_path}")
        os.remove(specification.report_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    logger.info(f"Run {run_index}: seed {specification.seed}, mode {specification.mode}, into {specification.output_directory}")
    run_values: dict[str, Any] = base_config.to_dictionary()
    run_values.update(
        random_seed=specification.seed,
        output_directory=specification.output_directory,
        enable_domain_adaptation=specification.enable_domain_adaptation,
        auto_resume=True,
    )
    trainer: Trainer = Trainer(Config(None, **run_values))
    trainer.train()

    if trainer.stopped_for_time_limit:
        logger.info(f"Run {run_index} stopped for the wall time limit at epoch {trainer.completed_epochs}, no report written")

        raise RunStoppedForTimeLimit(f"run {run_index} ({specification.mode}, seed {specification.seed})")

    trainer.load_best_model()

    prediction_set: PredictionSet = trainer.collect_predictions("test")
    target_report, source_report = classification_reports_from_predictions(
        prediction_set, trainer.config.number_of_classes
    )
    logger.info(
        f"Run {run_index} test: target accuracy {target_report.accuracy:.4f}, source accuracy {source_report.accuracy:.4f}"
    )

    class_names: Optional[list[str]] = class_names_for(trainer.config)
    for domain_label, domain_name, report in (
        ("target", trainer.config.target_domain_name, target_report),
        ("source", trainer.config.source_domain_name, source_report),
    ):
        plot_confusion_matrix(
            report,
            os.path.join(specification.output_directory, f"confusion_matrix_{domain_label}.png"),
            class_names,
            title=f"{domain_name} ({specification.mode}, seed {specification.seed})",
        )
        plot_roc_curves(
            report,
            os.path.join(specification.output_directory, f"roc_{domain_label}.png"),
            class_names,
            title=f"{domain_name} ROC ({specification.mode}, seed {specification.seed})",
        )

    with open(specification.report_path, "w") as report_file:
        json.dump(
            {
                "seed": specification.seed,
                "mode": specification.mode,
                "best_epoch": trainer.early_stopping.best_epoch,
                "completed_epochs": trainer.completed_epochs,
                "target": target_report.to_dictionary(),
                "source": source_report.to_dictionary(),
            },
            report_file,
            indent=2,
        )
    logger.debug(f"Wrote {specification.report_path}")

    return specification


def run_experiment(base_config: Config, run_index: Optional[int] = None) -> list[RunSpecification]:
    "All runs when run_index is None, otherwise only that one. Returns the specifications that actually ran."
    run_indices: list[int] = (
        list(range(len(MODES) * base_config.number_of_seeds)) if run_index is None else [run_index]
    )
    completed_specifications: list[RunSpecification] = []

    for current_run_index in run_indices:
        specification: Optional[RunSpecification] = run_single(base_config, current_run_index)
        if specification is not None:
            completed_specifications.append(specification)

    return completed_specifications


def experiment_command(config_path: str, override_texts: list[str], run_index: Optional[int]) -> int:
    "Returns 0 when every requested run completed, EX_TEMPFAIL at the first run stopped for the time limit."
    from astrouda.cli import parse_override
    from astrouda.experiment.aggregation import aggregate_experiment

    overrides: dict[str, Any] = dict(parse_override(override_text) for override_text in override_texts)
    config: Config = Config(config_path, **overrides)
    try:
        run_experiment(config, run_index)
    except RunStoppedForTimeLimit as stopped_run:
        get_logger(__name__, config).info(f"Stopped for the wall time limit in {stopped_run}, rerun the same command to continue")

        return EX_TEMPFAIL

    if run_index is None:
        aggregate_experiment(config)

    return 0
