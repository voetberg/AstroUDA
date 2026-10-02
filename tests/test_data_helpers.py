import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT: Path = Path(__file__).resolve().parent.parent
HELPERS_PATH: Path = REPOSITORY_ROOT / "slurm" / "data_helpers.sh"

FAKE_PYTHON: str = """#!/bin/bash
echo "$@" >> "${FAKE_BIN_DIRECTORY}/python.log"
exit ${FAKE_PYTHON_EXIT_CODE:-0}
"""


@pytest.fixture
def fake_bin_directory(tmp_path: Path) -> Path:
    bin_directory: Path = tmp_path / "bin"
    bin_directory.mkdir()
    (bin_directory / "python").write_text(FAKE_PYTHON)
    (bin_directory / "python").chmod(0o755)

    return bin_directory


def run_prepare_data(fake_bin_directory: Path, extra_environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    environment: dict[str, str] = {
        "PATH": f"{fake_bin_directory}:/usr/bin:/bin",
        "FAKE_BIN_DIRECTORY": str(fake_bin_directory),
        **extra_environment,
    }
    snippet: str = f'''set -euo pipefail
source "{HELPERS_PATH}"
prepare_data
echo "ARGUMENT_COUNT=${{#DATA_ARGUMENTS[@]}}"
for argument in ${{DATA_ARGUMENTS[@]+"${{DATA_ARGUMENTS[@]}}"}}; do echo "ARGUMENT=$argument"; done
'''

    return subprocess.run(["bash", "-c", snippet], env=environment, capture_output=True, text=True)


def read_python_calls(fake_bin_directory: Path) -> list[str]:
    log_path: Path = fake_bin_directory / "python.log"

    return log_path.read_text().splitlines() if log_path.exists() else []


def test_nothing_happens_without_download_dataset(fake_bin_directory: Path) -> None:
    completed = run_prepare_data(fake_bin_directory, {})

    assert completed.returncode == 0, completed.stderr
    assert "ARGUMENT_COUNT=0" in completed.stdout
    assert read_python_calls(fake_bin_directory) == []


def test_download_goes_to_scratch_by_default(fake_bin_directory: Path, tmp_path: Path) -> None:
    completed = run_prepare_data(fake_bin_directory, {"DOWNLOAD_DATASET": "lsst", "SCRATCH": str(tmp_path / "scratch")})

    expected_directory: str = f"{tmp_path / 'scratch'}/astrouda_data/lsst"
    assert completed.returncode == 0, completed.stderr
    assert read_python_calls(fake_bin_directory) == [f"-m astrouda.cli download --dataset lsst --directory {expected_directory}"]
    assert "ARGUMENT_COUNT=2" in completed.stdout
    assert "ARGUMENT=--set" in completed.stdout
    assert f"ARGUMENT=data_directory={expected_directory}" in completed.stdout


def test_data_directory_overrides_scratch(fake_bin_directory: Path, tmp_path: Path) -> None:
    completed = run_prepare_data(
        fake_bin_directory, {"DOWNLOAD_DATASET": "gz2", "SCRATCH": str(tmp_path / "scratch"), "DATA_DIRECTORY": str(tmp_path / "elsewhere")}
    )

    assert completed.returncode == 0, completed.stderr
    assert read_python_calls(fake_bin_directory) == [f"-m astrouda.cli download --dataset gz2 --directory {tmp_path / 'elsewhere'}"]
    assert f"ARGUMENT=data_directory={tmp_path / 'elsewhere'}" in completed.stdout


def test_missing_scratch_without_data_directory_fails_clearly(fake_bin_directory: Path) -> None:
    completed = run_prepare_data(fake_bin_directory, {"DOWNLOAD_DATASET": "lsst"})

    assert completed.returncode != 0
    assert "SCRATCH must be set" in completed.stderr
    assert read_python_calls(fake_bin_directory) == []


def test_failed_download_fails_the_job(fake_bin_directory: Path, tmp_path: Path) -> None:
    completed = run_prepare_data(
        fake_bin_directory, {"DOWNLOAD_DATASET": "lsst", "SCRATCH": str(tmp_path), "FAKE_PYTHON_EXIT_CODE": "1"}
    )

    assert completed.returncode != 0
    assert "ARGUMENT_COUNT" not in completed.stdout
