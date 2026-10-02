import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT: Path = Path(__file__).resolve().parent.parent
SLURM_DIRECTORY: Path = REPOSITORY_ROOT / "slurm"
HELPERS_PATH: Path = SLURM_DIRECTORY / "resubmit_helpers.sh"

FAKE_SCONTROL: str = '#!/bin/bash\necho "JobId=${SLURM_JOB_ID} Name=x RunTime=00:01:00 TimeLimit=${FAKE_TIME_LIMIT} Priority=1"\n'
FAKE_SBATCH: str = """#!/bin/bash
echo "$@" >> "${FAKE_BIN_DIRECTORY}/sbatch.log"
COUNTER_FILE="${FAKE_BIN_DIRECTORY}/sbatch.counter"
COUNT=$(cat "${COUNTER_FILE}" 2> /dev/null || echo 100)
echo $((COUNT + 1)) > "${COUNTER_FILE}"
echo $((COUNT + 1))
"""
FAKE_SQUEUE: str = '#!/bin/bash\necho "${FAKE_SQUEUE_OUTPUT:-}"\n'


@pytest.fixture
def fake_bin_directory(tmp_path: Path) -> Path:
    bin_directory: Path = tmp_path / "bin"
    bin_directory.mkdir()
    for name, text in (("scontrol", FAKE_SCONTROL), ("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE)):
        (bin_directory / name).write_text(text)
        (bin_directory / name).chmod(0o755)

    return bin_directory


def run_helpers(
    snippet: str, fake_bin_directory: Path, extra_environment: dict[str, str] | None = None, with_fake_binaries: bool = True
) -> subprocess.CompletedProcess[str]:
    path_prefix: str = f"{fake_bin_directory}:" if with_fake_binaries else ""
    environment: dict[str, str] = {
        "PATH": f"{path_prefix}/usr/bin:/bin",
        "FAKE_BIN_DIRECTORY": str(fake_bin_directory),
        "SLURM_JOB_ID": "555",
        "USER": "tester",
        **(extra_environment or {}),
    }

    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail\nsource "{HELPERS_PATH}"\n{snippet}'],
        env=environment,
        capture_output=True,
        text=True,
    )


def read_sbatch_log(fake_bin_directory: Path) -> list[str]:
    return (fake_bin_directory / "sbatch.log").read_text().splitlines()


@pytest.mark.parametrize(
    "time_limit_text, expected_seconds",
    [("08:00:00", 28800), ("1-02:03:04", 93784), ("30:00", 1800), ("00:00:05", 5), ("08:09:08", 29348)],
)
def test_time_limit_formats_are_parsed(fake_bin_directory: Path, time_limit_text: str, expected_seconds: int) -> None:
    completed = run_helpers("slurm_time_limit_seconds", fake_bin_directory, {"FAKE_TIME_LIMIT": time_limit_text})

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(expected_seconds)


@pytest.mark.parametrize("time_limit_text", ["UNLIMITED", "NOT_SET", ""])
def test_unparseable_time_limit_prints_nothing(fake_bin_directory: Path, time_limit_text: str) -> None:
    completed = run_helpers("wall_time_budget_seconds", fake_bin_directory, {"FAKE_TIME_LIMIT": time_limit_text})

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""


def test_budget_without_scontrol_prints_nothing(fake_bin_directory: Path) -> None:
    completed = run_helpers("wall_time_budget_seconds", fake_bin_directory, with_fake_binaries=False)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""


def test_budget_subtracts_used_seconds_and_margin(fake_bin_directory: Path) -> None:
    environment: dict[str, str] = {"FAKE_TIME_LIMIT": "08:00:00"}

    default_margin = run_helpers("SECONDS=100; wall_time_budget_seconds", fake_bin_directory, environment)
    custom_margin = run_helpers(
        "SECONDS=100; wall_time_budget_seconds", fake_bin_directory, {**environment, "TIME_MARGIN_SECONDS": "60"}
    )

    assert default_margin.stdout.strip() == str(28800 - 100 - 1200)
    assert custom_margin.stdout.strip() == str(28800 - 100 - 60)


def test_budget_is_never_negative(fake_bin_directory: Path) -> None:
    completed = run_helpers("wall_time_budget_seconds", fake_bin_directory, {"FAKE_TIME_LIMIT": "10:00"})

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0"


def test_resubmission_cap_is_enforced(fake_bin_directory: Path) -> None:
    below_cap = run_helpers("check_resubmission_allowed", fake_bin_directory, {"RESUBMISSION_COUNT": "19"})
    at_cap = run_helpers("check_resubmission_allowed", fake_bin_directory, {"RESUBMISSION_COUNT": "20"})
    custom_cap = run_helpers(
        "check_resubmission_allowed", fake_bin_directory, {"RESUBMISSION_COUNT": "3", "MAXIMUM_RESUBMISSIONS": "3"}
    )
    fresh_chain = run_helpers("check_resubmission_allowed", fake_bin_directory)

    assert below_cap.returncode == 0
    assert fresh_chain.returncode == 0
    assert at_cap.returncode != 0 and "resubmission cap reached (20 of 20)" in at_cap.stderr
    assert custom_cap.returncode != 0 and "MAXIMUM_RESUBMISSIONS" in custom_cap.stderr


def test_resubmit_training_job_passes_same_arguments_and_incremented_count(fake_bin_directory: Path) -> None:
    completed = run_helpers(
        'resubmit_training_job slurm/train.sbatch configs/x.json --set "random_seed=1"',
        fake_bin_directory,
        {"RESUBMISSION_COUNT": "2"},
    )

    assert completed.returncode == 0, completed.stderr
    assert read_sbatch_log(fake_bin_directory) == [
        "--parsable --export=ALL,RESUBMISSION_COUNT=3 slurm/train.sbatch configs/x.json --set random_seed=1"
    ]
    assert "job 101" in completed.stdout


def test_resubmit_training_job_refuses_past_the_cap(fake_bin_directory: Path) -> None:
    completed = run_helpers("resubmit_training_job slurm/train.sbatch cfg", fake_bin_directory, {"RESUBMISSION_COUNT": "20"})

    assert completed.returncode != 0
    assert not (fake_bin_directory / "sbatch.log").exists()


def test_resubmit_experiment_task_submits_one_task_and_a_dependent_aggregation(fake_bin_directory: Path) -> None:
    completed = run_helpers(
        "resubmit_experiment_task slurm/experiment_array.sbatch configs/x.json --set number_of_seeds=2",
        fake_bin_directory,
        {"SLURM_ARRAY_TASK_ID": "3"},
    )

    assert completed.returncode == 0, completed.stderr
    assert read_sbatch_log(fake_bin_directory) == [
        "--parsable --export=ALL,RESUBMISSION_COUNT=1 --array=3 slurm/experiment_array.sbatch configs/x.json --set number_of_seeds=2",
        "--parsable --dependency=afterok:101 slurm/aggregate.sbatch configs/x.json --set number_of_seeds=2",
    ]
    assert "aggregation job 102" in completed.stdout


@pytest.mark.parametrize(
    "aggregate_output, squeue_output, expect_deferred",
    [
        ("3 of 4 runs have no report.json: run 1", "123 astrouda_experiment", True),
        ("3 of 4 runs have no report.json: run 1", "", False),
        ("some other failure", "123 astrouda_experiment", False),
    ],
)
def test_aggregation_deferral(
    fake_bin_directory: Path, tmp_path: Path, aggregate_output: str, squeue_output: str, expect_deferred: bool
) -> None:
    output_file: Path = tmp_path / "aggregate_output.txt"
    output_file.write_text(aggregate_output)

    completed = run_helpers(
        f'aggregation_should_be_deferred "{output_file}"', fake_bin_directory, {"FAKE_SQUEUE_OUTPUT": squeue_output}
    )

    assert (completed.returncode == 0) is expect_deferred


def read_script(name: str) -> str:
    return (SLURM_DIRECTORY / name).read_text()


def test_helpers_file_only_defines_functions() -> None:
    completed = subprocess.run(["bash", "-n", str(HELPERS_PATH)], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert "set -euo pipefail" not in read_script("resubmit_helpers.sh")


@pytest.mark.parametrize("script_name", ["train.sbatch", "experiment_array.sbatch"])
def test_resubmitting_scripts_follow_the_exit_code_contract(script_name: str) -> None:
    script_text: str = read_script(script_name)

    assert "#SBATCH --time=08:00:00" in script_text
    assert 'ORIGINAL_ARGUMENTS=("$@")' in script_text
    assert script_text.index("ORIGINAL_ARGUMENTS=") < script_text.index("then shift")
    assert "source slurm/resubmit_helpers.sh" in script_text
    assert "--set auto_resume=true" in script_text
    assert "maximum_wall_time_seconds=" in script_text
    assert "-eq 75" in script_text
    assert "set +e" in script_text and "set -e" in script_text


def test_optimize_array_has_time_limit_but_no_resubmission() -> None:
    script_text: str = read_script("optimize_array.sbatch")

    assert "#SBATCH --time=08:00:00" in script_text
    assert "resubmit" not in script_text.replace("self-resubmission", "")
    assert "maximum_wall_time_seconds" not in script_text


def test_aggregate_script_defers_while_experiments_are_pending() -> None:
    script_text: str = read_script("aggregate.sbatch")

    assert "#SBATCH --time=01:00:00" in script_text
    assert "aggregation_should_be_deferred" in script_text
    assert "deferred to a later job" in script_text
