import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest
import torch

from astrouda.config import Config
from astrouda.training import Trainer

ConfigFactory = Callable[..., Config]
REPOSITORY_ROOT: Path = Path(__file__).resolve().parent.parent
SLURM_DIRECTORY: Path = REPOSITORY_ROOT / "slurm"
SLURM_SCRIPT_PATHS: list[Path] = sorted(SLURM_DIRECTORY.glob("*.sbatch")) + [SLURM_DIRECTORY / "submit_experiment.sh"]

ALLOWED_FLAGS_BY_SUBCOMMAND: dict[str, set[str]] = {
    "train": {"--config", "--set"},
    "experiment": {"--config", "--set", "--run-index"},
    "aggregate": {"--config", "--set"},
    "optimize": {"--config", "--search-space", "--trials", "--trial-index", "--set"},
}
CLI_INVOCATION_PATTERN: re.Pattern[str] = re.compile(r"python -m astrouda\.cli (\w+)((?:[^\n]|\\\n)*)")
FLAG_PATTERN: re.Pattern[str] = re.compile(r"(--[a-z][a-z-]*)")


def read_cli_invocations(script_path: Path) -> list[tuple[str, set[str]]]:
    script_text: str = script_path.read_text()

    return [
        (match.group(1), set(FLAG_PATTERN.findall(match.group(2))))
        for match in CLI_INVOCATION_PATTERN.finditer(script_text)
    ]


@pytest.mark.parametrize("script_path", SLURM_SCRIPT_PATHS, ids=lambda path: path.name)
def test_script_has_valid_bash_syntax(script_path: Path) -> None:
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        ["bash", "-n", str(script_path)], capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert "set -euo pipefail" in script_path.read_text()


@pytest.mark.parametrize("script_path", SLURM_SCRIPT_PATHS, ids=lambda path: path.name)
def test_script_only_uses_known_cli_interfaces(script_path: Path) -> None:
    for subcommand, flags in read_cli_invocations(script_path):
        assert subcommand in ALLOWED_FLAGS_BY_SUBCOMMAND
        assert flags <= ALLOWED_FLAGS_BY_SUBCOMMAND[subcommand]


def test_every_cli_subcommand_is_submitted() -> None:
    used_subcommands: set[str] = {
        subcommand for script_path in SLURM_SCRIPT_PATHS for subcommand, _ in read_cli_invocations(script_path)
    }

    assert used_subcommands == set(ALLOWED_FLAGS_BY_SUBCOMMAND)


@pytest.mark.parametrize("script_path", sorted(SLURM_DIRECTORY.glob("*.sbatch")), ids=lambda path: path.name)
def test_sbatch_scripts_declare_resources(script_path: Path) -> None:
    script_text: str = script_path.read_text()

    for directive in ("--partition=", "--account=", "--time=", "--mem=", "--cpus-per-task=", "--output=slurm/logs/"):
        assert f"#SBATCH {directive}" in script_text
    if script_path.name != "aggregate.sbatch":
        assert "#SBATCH --gres=gpu:a100:1" in script_text


def test_experiment_array_covers_both_modes_for_every_seed() -> None:
    script_text: str = (SLURM_DIRECTORY / "experiment_array.sbatch").read_text()
    submit_text: str = (SLURM_DIRECTORY / "submit_experiment.sh").read_text()

    assert "--run-index" in script_text
    assert "--dependency=" in submit_text and "afterok" in submit_text
    assert "2 * NUMBER_OF_SEEDS - 1" in submit_text


def test_logs_directory_is_tracked_but_ignored() -> None:
    assert (SLURM_DIRECTORY / "logs" / ".gitkeep").exists()
    assert "slurm/logs/*" in (REPOSITORY_ROOT / ".gitignore").read_text()


@pytest.mark.parametrize(
    "config_name", ["lsst_full.json", "gz2_sdss_decals.json", "gz2_sdss_wide_deep.json"]
)
def test_full_size_configs_are_a100_ready(config_name: str) -> None:
    config: Config = Config(str(REPOSITORY_ROOT / "configs" / config_name))

    assert config.backbone_depth == 50
    assert config.crop_size == 256
    assert config.batch_size == 64
    assert config.use_mixed_precision is True
    assert config.number_of_seeds >= 1


def test_submit_script_reads_seed_count_from_config(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"number_of_seeds": 3}))
    fake_binary_directory: Path = tmp_path / "bin"
    fake_binary_directory.mkdir()
    sbatch_log: Path = tmp_path / "sbatch.log"
    fake_sbatch: Path = fake_binary_directory / "sbatch"
    fake_sbatch.write_text(f'#!/bin/bash\necho "$@" >> {sbatch_log}\necho 4242\n')
    fake_sbatch.chmod(0o755)

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        ["bash", str(SLURM_DIRECTORY / "submit_experiment.sh"), str(config_path), "--set", "number_of_seeds=4"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPOSITORY_ROOT), "PATH": f"{fake_binary_directory}:{Path(sys.executable).parent}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    submitted_lines: list[str] = sbatch_log.read_text().splitlines()
    assert "--array=0-7" in submitted_lines[0]
    assert "--dependency=afterok:4242" in submitted_lines[1]


def test_mixed_precision_is_disabled_with_warning_on_cpu(make_tiny_config: ConfigFactory, caplog: pytest.LogCaptureFixture) -> None:
    trainer: Trainer = Trainer(make_tiny_config(use_mixed_precision=True, device="cpu", maximum_epochs=1))
    history: list[dict[str, float]] = trainer.train()

    assert trainer.mixed_precision_enabled is False
    assert trainer.gradient_scaler.is_enabled() is False
    assert "only supported on cuda" in caplog.text
    assert len(history) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mixed_precision_training_on_cuda_has_finite_losses(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(make_tiny_config(use_mixed_precision=True, device="cuda", maximum_epochs=2))
    history: list[dict[str, float]] = trainer.train()

    assert trainer.mixed_precision_enabled is True
    assert trainer.gradient_scaler.is_enabled() is True
    assert len(history) == 2
    for epoch_record in history:
        for loss_name in ("cross_entropy_loss", "adaptive_clustering_loss", "entropy_separation_loss", "total_loss"):
            assert torch.isfinite(torch.tensor(epoch_record[loss_name]))
