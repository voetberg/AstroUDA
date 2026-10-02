import json
from pathlib import Path

import pytest

from astrouda.config import Config


def test_defaults_are_independent_per_instance() -> None:
    first_config: Config = Config()
    second_config: Config = Config()
    first_config.entropy_step_values.append(9.9)

    assert 9.9 not in second_config.entropy_step_values
    assert 9.9 not in Config.entropy_step_values


def test_json_overwrites_defaults(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"batch_size": 7, "learning_rate": 0.5, "entropy_boundary_initial": 0.25}))

    config: Config = Config(str(config_path))

    assert config.batch_size == 7
    assert config.learning_rate == 0.5
    assert config.entropy_boundary_initial == 0.25
    assert config.momentum == Config.momentum


def test_overrides_beat_json(tmp_path: Path) -> None:
    config_path: Path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"batch_size": 7}))

    assert Config(str(config_path), batch_size=3).batch_size == 3


def test_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        Config(batch_sizee=4)


def test_wrong_type_raises() -> None:
    with pytest.raises(TypeError):
        Config(batch_size="4")
    with pytest.raises(TypeError):
        Config(use_nesterov=1)
    with pytest.raises(TypeError):
        Config(entropy_step_values=[0.3, "x"])


def test_int_is_accepted_for_float() -> None:
    assert Config(learning_rate=1).learning_rate == 1


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        Config(train_fraction=0.9)


def test_json_round_trip(tmp_path: Path) -> None:
    output_path: Path = tmp_path / "out.json"
    original_config: Config = Config(batch_size=5, backbone_depth=18)
    original_config.to_json(str(output_path))

    assert Config(str(output_path)).to_dictionary() == original_config.to_dictionary()


@pytest.mark.parametrize("config_name", ["smoke", "lsst_full"])
def test_shipped_configs_load(config_name: str) -> None:
    Config(f"configs/{config_name}.json")
