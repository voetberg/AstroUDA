from pathlib import Path

from astrouda.cli import main
from astrouda.exit_codes import EX_TEMPFAIL

TIME_LIMITED_ARGUMENTS: list[str] = [
    "--set", "maximum_epochs=2",
    "--set", "number_of_seeds=1",
    "--set", "maximum_wall_time_seconds=0",
]


def test_run_index_stopped_for_time_limit_writes_no_report_and_rerun_continues(
    tiny_config_path: Path, tmp_path: Path
) -> None:
    command: list[str] = ["experiment", "--config", str(tiny_config_path), "--run-index", "0", *TIME_LIMITED_ARGUMENTS]
    run_directory: Path = tmp_path / "output" / "seed_0" / "adapted"

    assert main(command) == EX_TEMPFAIL
    assert (run_directory / "checkpoint.pt").exists()
    assert not (run_directory / "report.json").exists()
    assert not (run_directory / "confusion_matrix_target.png").exists()

    assert main(command) == 0
    assert (run_directory / "report.json").exists()
    assert main(command) == 0


def test_sequential_experiment_stops_at_first_unfinished_run(tiny_config_path: Path, tmp_path: Path) -> None:
    command: list[str] = ["experiment", "--config", str(tiny_config_path), *TIME_LIMITED_ARGUMENTS]
    output_directory: Path = tmp_path / "output"

    assert main(command) == EX_TEMPFAIL
    assert not (output_directory / "seed_0" / "source_only").exists()
    assert not (output_directory / "aggregate.json").exists()

    exit_codes: list[int] = [main(command) for _ in range(4)]

    assert exit_codes[-1] == 0
    assert set(exit_codes) <= {0, EX_TEMPFAIL}
    assert (output_directory / "aggregate.json").exists()


def test_experiment_failure_exits_one(tiny_config_path: Path) -> None:
    assert main(["experiment", "--config", str(tiny_config_path), "--run-index", "99"]) == 1
