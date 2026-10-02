import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from astrouda.cli import main
from astrouda.config import Config
from astrouda.evaluation import ClassificationReport, aggregate_reports, compute_classification_report
from astrouda.experiment import (
    MissingRunsError,
    aggregate_experiment,
    decode_run_index,
    enumerate_run_specifications,
    run_experiment,
)
from astrouda.training import Trainer

FAST_SET_ARGUMENTS: list[str] = ["--set", "maximum_epochs=1", "--set", "number_of_seeds=2"]


def relative_files(directory: Path) -> set[str]:
    return {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}


def test_run_index_mapping(make_tiny_config: Callable[..., Config]) -> None:
    config: Config = make_tiny_config(number_of_seeds=3, random_seed=10)

    decoded_runs: list[tuple[int, str]] = [decode_run_index(config, run_index) for run_index in range(6)]
    assert decoded_runs == [
        (0, "adapted"),
        (0, "source_only"),
        (1, "adapted"),
        (1, "source_only"),
        (2, "adapted"),
        (2, "source_only"),
    ]

    specifications = enumerate_run_specifications(config)
    assert [specification.seed for specification in specifications] == [10, 10, 11, 11, 12, 12]
    assert specifications[3].output_directory == os.path.join(config.output_directory, "seed_11", "source_only")
    assert specifications[2].enable_domain_adaptation and not specifications[3].enable_domain_adaptation

    with pytest.raises(ValueError):
        decode_run_index(config, 6)


def test_single_run_index_writes_only_its_files(tiny_config_path: Path, tmp_path: Path) -> None:
    exit_code: int = main(["experiment", "--config", str(tiny_config_path), "--run-index", "3", *FAST_SET_ARGUMENTS])

    assert exit_code == 0
    output_directory: Path = tmp_path / "output"
    assert {path.name for path in output_directory.iterdir()} == {"seed_1"}
    assert {path.name for path in (output_directory / "seed_1").iterdir()} == {"source_only"}

    written_files: set[str] = relative_files(output_directory / "seed_1" / "source_only")
    for expected_file in (
        "report.json",
        "history.json",
        "best_model.pt",
        "confusion_matrix_target.png",
        "roc_target.png",
        "confusion_matrix_source.png",
        "roc_source.png",
    ):
        assert expected_file in written_files

    with open(output_directory / "seed_1" / "source_only" / "report.json") as report_file:
        run_record = json.load(report_file)
    assert run_record["seed"] == 1 and run_record["mode"] == "source_only"
    assert set(run_record) >= {"target", "source"}


def test_completed_runs_are_skipped_unless_overwritten(make_tiny_config: Callable[..., Config]) -> None:
    config: Config = make_tiny_config(maximum_epochs=1, number_of_seeds=1)

    assert len(run_experiment(config, 1)) == 1
    report_path: str = os.path.join(config.output_directory, "seed_0", "source_only", "report.json")
    first_modification_time: int = os.stat(report_path).st_mtime_ns

    assert run_experiment(config, 1) == []
    assert os.stat(report_path).st_mtime_ns == first_modification_time

    overwriting_config: Config = make_tiny_config(maximum_epochs=1, number_of_seeds=1, overwrite_existing_runs=True)
    assert len(run_experiment(overwriting_config, 1)) == 1
    assert os.stat(report_path).st_mtime_ns > first_modification_time


def test_full_sequential_path_writes_aggregates(
    tiny_config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code: int = main(["experiment", "--config", str(tiny_config_path), *FAST_SET_ARGUMENTS])

    assert exit_code == 0
    output_directory: Path = tmp_path / "output"
    for seed in (0, 1):
        for mode in ("adapted", "source_only"):
            assert (output_directory / f"seed_{seed}" / mode / "report.json").exists()

    with open(output_directory / "aggregate.json") as aggregate_file:
        aggregate_record = json.load(aggregate_file)
    assert aggregate_record["seeds"] == [0, 1]
    assert aggregate_record["domains"]["target"]["adapted"]["number_of_reports"] == 2
    assert aggregate_record["domains"]["source"]["source_only"]["number_of_reports"] == 2

    table_text: str = (output_directory / "comparison_table.md").read_text()
    assert "| Accuracy |" in table_text
    assert table_text.strip() in capsys.readouterr().out
    assert (output_directory / "aggregate_metrics_target.png").exists()
    assert (output_directory / "aggregate_confusion_matrix_target_adapted.png").exists()


def test_aggregate_fails_clearly_on_missing_run(tiny_config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["experiment", "--config", str(tiny_config_path), "--run-index", "0", *FAST_SET_ARGUMENTS]) == 0

    config: Config = Config(str(tiny_config_path), number_of_seeds=2, maximum_epochs=1)
    with pytest.raises(MissingRunsError, match=r"3 of 4 runs.*run 1 \(source_only, seed 0\)"):
        aggregate_experiment(config)

    assert main(["aggregate", "--config", str(tiny_config_path), "--set", "number_of_seeds=2"]) == 1


def make_report(accuracy_pattern: np.ndarray) -> ClassificationReport:
    labels: np.ndarray = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    predicted: np.ndarray = np.where(accuracy_pattern, labels, (labels + 1) % 3)
    probabilities: np.ndarray = np.eye(3)[predicted] * 0.8 + 0.1

    return compute_classification_report(labels, predicted, probabilities, 3)


def test_population_standard_deviation_matches_numpy() -> None:
    reports: list[ClassificationReport] = [
        make_report(np.array([True] * 10)),
        make_report(np.array([True] * 7 + [False] * 3)),
        make_report(np.array([True] * 4 + [False] * 6)),
    ]
    accuracies: list[float] = [report.accuracy for report in reports]

    aggregated = aggregate_reports(reports)

    assert aggregated.accuracy.mean == pytest.approx(np.mean(accuracies))
    assert aggregated.accuracy.std == pytest.approx(np.std(accuracies, ddof=0))
    assert aggregated.accuracy.std != pytest.approx(np.std(accuracies, ddof=1))


def test_load_best_model_restores_best_epoch_weights(make_tiny_config: Callable[..., Config]) -> None:
    config: Config = make_tiny_config(maximum_epochs=2, early_stopping_patience_epochs=5)
    trainer: Trainer = Trainer(config)
    trainer.train()

    best_state: dict[str, torch.Tensor] = torch.load(trainer.best_model_path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)

    trainer.load_best_model()

    for name, restored_tensor in trainer.model.state_dict().items():
        assert torch.equal(restored_tensor.cpu(), best_state[name]), name


def test_load_best_model_without_training_raises(make_tiny_config: Callable[..., Config]) -> None:
    trainer: Trainer = Trainer(make_tiny_config())

    with pytest.raises(FileNotFoundError):
        trainer.load_best_model()
