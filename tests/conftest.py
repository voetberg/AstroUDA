import json
from pathlib import Path
from typing import Any, Callable

import pytest

from astrouda.config import Config

REPOSITORY_ROOT: Path = Path(__file__).resolve().parent.parent
SMOKE_CONFIG_PATH: Path = REPOSITORY_ROOT / "configs" / "smoke.json"

TINY_OVERRIDES: dict[str, Any] = {
    "synthetic_samples_per_domain": 40,
    "native_image_size": 16,
    "crop_size": 16,
    "augmentation_zoom_size": 20,
    "batch_size": 8,
    "log_level": "WARNING",
}


@pytest.fixture
def tiny_config_path(tmp_path: Path) -> Path:
    "configs/smoke.json shrunk to a few seconds of CPU work, with its output under tmp_path."
    with open(SMOKE_CONFIG_PATH, "r") as smoke_file:
        tiny_values: dict[str, Any] = json.load(smoke_file)

    tiny_values.update(TINY_OVERRIDES)
    tiny_values["output_directory"] = str(tmp_path / "output")
    tiny_config_path: Path = tmp_path / "tiny.json"
    with open(tiny_config_path, "w") as tiny_file:
        json.dump(tiny_values, tiny_file)

    return tiny_config_path


@pytest.fixture
def make_tiny_config(tiny_config_path: Path, tmp_path: Path) -> Callable[..., Config]:
    def build(**overrides: Any) -> Config:
        return Config(str(tiny_config_path), **overrides)

    return build
