import json
import math
import os
from pathlib import Path

from astrouda.cli import main
from tests.conftest import SMOKE_CONFIG_PATH


def test_smoke_config_trains_end_to_end_on_cpu(tmp_path: Path) -> None:
    output_directory: Path = tmp_path / "smoke_output"

    exit_code: int = main(
        [
            "train",
            "--config",
            str(SMOKE_CONFIG_PATH),
            "--set",
            f"output_directory={json.dumps(str(output_directory))}",
            "--set",
            'log_level="WARNING"',
        ]
    )

    assert exit_code == 0
    for file_name in ("config.json", "history.json", "checkpoint.pt", "best_model.pt"):
        assert os.path.exists(output_directory / file_name), file_name

    with open(output_directory / "history.json") as history_file:
        history: list[dict[str, float]] = json.load(history_file)
    with open(SMOKE_CONFIG_PATH) as smoke_file:
        expected_epochs: int = json.load(smoke_file)["maximum_epochs"]

    assert len(history) == expected_epochs
    for record in history:
        assert all(math.isfinite(value) for value in record.values())
        assert 0.0 <= record["target_validation_accuracy"] <= 1.0
