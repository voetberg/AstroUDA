import json
import math
from pathlib import Path
from typing import Any

import pytest

from astrouda.optimization import SearchSpace, sample_trial_parameters

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]

FULL_DEFINITION: dict[str, Any] = {
    "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "momentum": {"type": "uniform", "low": 0.8, "high": 0.99},
    "top_k": {"type": "int", "low": 2, "high": 7},
    "bank_size": {"type": "choice", "choices": [512, 1024, 2048]},
}


def test_sampling_is_deterministic() -> None:
    search_space: SearchSpace = SearchSpace(FULL_DEFINITION)

    assert sample_trial_parameters(search_space, 3, 5) == sample_trial_parameters(search_space, 3, 5)


def test_trial_parameters_do_not_depend_on_other_trials() -> None:
    search_space: SearchSpace = SearchSpace(FULL_DEFINITION)

    direct_sample: dict[str, Any] = sample_trial_parameters(search_space, 3, 7)
    for earlier_index in range(7):
        sample_trial_parameters(search_space, 3, earlier_index)

    assert sample_trial_parameters(search_space, 3, 7) == direct_sample


def test_seed_and_index_change_the_sample() -> None:
    search_space: SearchSpace = SearchSpace(FULL_DEFINITION)
    reference: dict[str, Any] = sample_trial_parameters(search_space, 0, 0)

    assert sample_trial_parameters(search_space, 1, 0) != reference
    assert sample_trial_parameters(search_space, 0, 1) != reference


def test_definition_key_order_is_irrelevant() -> None:
    reversed_definition: dict[str, Any] = dict(reversed(list(FULL_DEFINITION.items())))

    assert sample_trial_parameters(SearchSpace(reversed_definition), 2, 4) == sample_trial_parameters(SearchSpace(FULL_DEFINITION), 2, 4)


def test_every_distribution_respects_bounds_and_types() -> None:
    search_space: SearchSpace = SearchSpace(FULL_DEFINITION)
    sampled_int_values: set[int] = set()

    for trial_index in range(300):
        sample: dict[str, Any] = sample_trial_parameters(search_space, 0, trial_index)
        assert 1e-4 <= sample["learning_rate"] <= 1e-2
        assert 0.8 <= sample["momentum"] <= 0.99
        assert isinstance(sample["top_k"], int) and 2 <= sample["top_k"] <= 7
        assert sample["bank_size"] in [512, 1024, 2048]
        sampled_int_values.add(sample["top_k"])

    assert sampled_int_values == set(range(2, 8))


def test_loguniform_is_uniform_in_log_space() -> None:
    search_space: SearchSpace = SearchSpace({"learning_rate": FULL_DEFINITION["learning_rate"]})
    log_values: list[float] = [math.log10(sample_trial_parameters(search_space, 0, index)["learning_rate"]) for index in range(400)]

    assert abs(sum(log_values) / len(log_values) - (-3.0)) < 0.15


@pytest.mark.parametrize(
    "definition, error_type, message_fragment",
    [
        ({"not_a_config_key": {"type": "uniform", "low": 0, "high": 1}}, KeyError, "not_a_config_key"),
        ({"output_directory": {"type": "choice", "choices": ["a"]}}, ValueError, "managed by the optimiser"),
        ({"momentum": {"type": "gaussian", "low": 0, "high": 1}}, ValueError, "unknown type"),
        ({"momentum": {"type": "uniform", "low": 0.9, "high": 0.5}}, ValueError, "low < high"),
        ({"momentum": {"type": "uniform", "low": 0.5}}, ValueError, "'high'"),
        ({"learning_rate": {"type": "loguniform", "low": 0.0, "high": 1.0}}, ValueError, "low > 0"),
        ({"top_k": {"type": "int", "low": 1.5, "high": 4}}, ValueError, "integer bounds"),
        ({"top_k": {"type": "uniform", "low": 1.5, "high": 4.5}}, TypeError, "rejected by Config"),
        ({"bank_size": {"type": "choice", "choices": []}}, ValueError, "non-empty"),
        ({"bank_size": {"type": "choice", "choices": [64, "big"]}}, TypeError, "rejected by Config"),
        ({"momentum": {"type": "uniform", "low": 0, "high": 1, "step": 2}}, ValueError, "unexpected fields"),
        ({}, ValueError, "non-empty"),
    ],
)
def test_invalid_search_spaces_raise_clear_errors(definition: dict[str, Any], error_type: type, message_fragment: str) -> None:
    with pytest.raises(error_type, match=message_fragment):
        SearchSpace(definition)


@pytest.mark.parametrize("file_name", ["search_space_example.json", "search_space_smoke.json"])
def test_shipped_search_spaces_load(file_name: str) -> None:
    search_space: SearchSpace = SearchSpace.from_json(str(REPOSITORY_ROOT / "configs" / file_name))

    assert "learning_rate" in search_space.parameter_names
    assert len(search_space.parameter_names) == 7


def test_from_json_round_trip(tmp_path: Path) -> None:
    search_space_path: Path = tmp_path / "space.json"
    search_space_path.write_text(json.dumps(FULL_DEFINITION))

    assert SearchSpace.from_json(str(search_space_path)).parameter_names == sorted(FULL_DEFINITION)
