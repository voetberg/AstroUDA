import json
import os
from pathlib import Path
from typing import Any

import pytest

from astrouda.cli import main
from astrouda.config import Config
from astrouda.optimization import OptimizationRunner, SearchSpace, TrialLog, TrialRecord
from astrouda.optimization import runner as runner_module

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
SMOKE_SEARCH_SPACE_PATH: Path = REPOSITORY_ROOT / "configs" / "search_space_smoke.json"


def build_runner(config_path: Path, number_of_trials: int = 3, **overrides: Any) -> OptimizationRunner:
    return OptimizationRunner(
        config_path=str(config_path),
        search_space=SearchSpace.from_json(str(SMOKE_SEARCH_SPACE_PATH)),
        number_of_trials=number_of_trials,
        config_overrides={"maximum_epochs": 1, **overrides},
    )


def make_record(trial_index: int, objective: float, status: str = "ok") -> TrialRecord:
    return TrialRecord(
        trial_index=trial_index,
        parameters={"learning_rate": 0.001 * (trial_index + 1)},
        objective=objective if status == "ok" else None,
        objective_metric="target_validation_accuracy",
        epochs_run=1,
        status=status,
        error=None if status == "ok" else "boom",
        wall_time_seconds=0.1,
        output_directory=f"trial_{trial_index}",
    )


def test_trial_log_keeps_last_record_and_picks_best(tmp_path: Path) -> None:
    trial_log: TrialLog = TrialLog(str(tmp_path / "optimization" / "trials.jsonl"))
    for record in [make_record(0, 0.4), make_record(1, 0.9), make_record(2, 0.9), make_record(3, 0.0, "failed")]:
        trial_log.append(record)
    trial_log.append(make_record(3, 0.95))

    assert trial_log.completed_indices() == {0, 1, 2, 3}
    best_record: TrialRecord | None = trial_log.best_record()
    assert best_record is not None and best_record.trial_index == 3

    trial_log.append(make_record(3, 0.0, "failed"))
    assert trial_log.completed_indices() == {0, 1, 2}
    assert trial_log.best_record().trial_index == 1


def test_runner_records_trials_and_best_config_round_trips(tiny_config_path: Path) -> None:
    runner: OptimizationRunner = build_runner(tiny_config_path, number_of_trials=2)
    records: list[TrialRecord] = runner.run()

    assert [record.status for record in records] == ["ok", "ok"]
    assert all(record.epochs_run == 1 and 0.0 <= record.objective <= 1.0 for record in records)
    assert all(os.path.exists(os.path.join(record.output_directory, "history.json")) for record in records)
    assert len({record.output_directory for record in records}) == 2

    best_record: TrialRecord = max(records, key=lambda record: (record.objective, -record.trial_index))
    best_config: Config = Config(runner.best_config_path)
    assert best_config.output_directory == best_record.output_directory
    for name, value in best_record.parameters.items():
        assert getattr(best_config, name) == pytest.approx(value)
    assert best_config.maximum_epochs == 1


def test_completed_trials_are_skipped_on_rerun(tiny_config_path: Path) -> None:
    first_runner: OptimizationRunner = build_runner(tiny_config_path, number_of_trials=2)
    first_runner.run(trial_index=0)
    assert first_runner.trial_log.completed_indices() == {0}

    second_runner: OptimizationRunner = build_runner(tiny_config_path, number_of_trials=2)
    new_records: list[TrialRecord] = second_runner.run()

    assert [record.trial_index for record in new_records] == [1]
    assert second_runner.run() == []
    with open(second_runner.trial_log.log_path) as log_file:
        assert len(log_file.readlines()) == 2


def test_failed_trial_is_recorded_and_sweep_continues(tiny_config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_trainer: type = runner_module.Trainer

    class FlakyTrainer(original_trainer):  # type: ignore[valid-type, misc]
        def train(self) -> list[dict[str, float]]:
            if self.config.output_directory.endswith("trial_0001"):
                raise RuntimeError("synthetic failure")
            return super().train()

    monkeypatch.setattr(runner_module, "Trainer", FlakyTrainer)
    runner: OptimizationRunner = build_runner(tiny_config_path, number_of_trials=3)
    records: list[TrialRecord] = runner.run()

    assert [record.status for record in records] == ["ok", "failed", "ok"]
    assert "synthetic failure" in records[1].error
    assert records[1].objective is None
    assert runner.trial_log.completed_indices() == {0, 2}
    assert os.path.exists(runner.best_config_path)

    monkeypatch.setattr(runner_module, "Trainer", original_trainer)
    retried_records: list[TrialRecord] = build_runner(tiny_config_path, number_of_trials=3).run()
    assert [(record.trial_index, record.status) for record in retried_records] == [(1, "ok")]


def test_trial_index_outside_range_and_clashing_overrides_raise(tiny_config_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        build_runner(tiny_config_path, number_of_trials=2).run(trial_index=2)

    with pytest.raises(ValueError, match="both searched and fixed"):
        build_runner(tiny_config_path, learning_rate=0.01)


def test_cli_smoke_sweep_of_three_trials(tiny_config_path: Path) -> None:
    exit_code: int = main(
        [
            "optimize",
            "--config", str(tiny_config_path),
            "--search-space", str(SMOKE_SEARCH_SPACE_PATH),
            "--trials", "3",
            "--set", "maximum_epochs=1",
        ]
    )
    assert exit_code == 0

    optimization_directory: str = os.path.join(Config(str(tiny_config_path)).output_directory, "optimization")
    with open(os.path.join(optimization_directory, "trials.jsonl")) as log_file:
        records: list[dict[str, Any]] = [json.loads(line) for line in log_file]

    assert [record["trial_index"] for record in records] == [0, 1, 2]
    assert all(record["status"] == "ok" and record["epochs_run"] == 1 for record in records)
    assert os.path.exists(os.path.join(optimization_directory, "best_config.json"))

    assert main(
        ["optimize", "--config", str(tiny_config_path), "--search-space", str(SMOKE_SEARCH_SPACE_PATH), "--trials", "3",
         "--trial-index", "1", "--set", "maximum_epochs=1"]
    ) == 0
    with open(os.path.join(optimization_directory, "trials.jsonl")) as log_file:
        assert len(log_file.readlines()) == 3


def test_cli_exits_non_zero_on_invalid_search_space(tiny_config_path: Path, tmp_path: Path) -> None:
    bad_search_space_path: Path = tmp_path / "bad.json"
    bad_search_space_path.write_text(json.dumps({"nope": {"type": "uniform", "low": 0, "high": 1}}))

    assert main(["optimize", "--config", str(tiny_config_path), "--search-space", str(bad_search_space_path), "--trials", "1"]) == 1


def test_output_directory_override_sets_optimisation_root(tiny_config_path: Path, tmp_path: Path) -> None:
    custom_root: Path = tmp_path / "custom_root"
    runner: OptimizationRunner = build_runner(tiny_config_path, number_of_trials=1, output_directory=str(custom_root))
    records: list[TrialRecord] = runner.run()

    assert records[0].status == "ok"
    assert records[0].output_directory == str(custom_root / "optimization" / "trial_0000")
