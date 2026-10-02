import json
import os
from pathlib import Path

import pytest

from astrouda.cli import main, parse_override, train_command


def test_parse_override_reads_json_values() -> None:
    assert parse_override("batch_size=4") == ("batch_size", 4)
    assert parse_override("use_nesterov=false") == ("use_nesterov", False)
    assert parse_override('output_directory="/tmp/x"') == ("output_directory", "/tmp/x")
    assert parse_override("entropy_step_values=[0.1, -0.1]") == ("entropy_step_values", [0.1, -0.1])
    assert parse_override("output_directory=/tmp/x") == ("output_directory", "/tmp/x")
    assert parse_override("entropy_boundary_initial=null") == ("entropy_boundary_initial", None)


def test_set_overrides_are_applied(tiny_config_path: Path, tmp_path: Path) -> None:
    other_output: Path = tmp_path / "overridden"
    trainer = train_command(
        str(tiny_config_path), ["maximum_epochs=1", f"output_directory={json.dumps(str(other_output))}", "batch_size=4"]
    )

    assert trainer.config.maximum_epochs == 1
    assert trainer.config.batch_size == 4
    assert len(trainer.history) == 1
    assert os.path.exists(other_output / "checkpoint.pt")


def test_cli_exits_non_zero_on_bad_override(tiny_config_path: Path) -> None:
    assert main(["train", "--config", str(tiny_config_path), "--set", "not_a_key=1"]) == 1
    assert main(["train", "--config", str(tiny_config_path), "--set", "batch_size=\"four\""]) == 1


def test_cli_exits_non_zero_on_missing_config(tmp_path: Path) -> None:
    assert main(["train", "--config", str(tmp_path / "missing.json")]) == 1


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        main([])
