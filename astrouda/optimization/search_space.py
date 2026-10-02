import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from astrouda.config import Config

logger: logging.Logger = logging.getLogger(__name__)

DISTRIBUTION_TYPES: tuple[str, ...] = ("uniform", "loguniform", "int", "choice")
RESERVED_KEYS: tuple[str, ...] = ("output_directory", "resume_from_checkpoint")


@dataclass(frozen=True)
class ParameterSpecification:
    name: str
    distribution: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[list[Any]] = None

    def probe_values(self) -> list[Any]:
        "Values that must be accepted by the Config field: both bounds, or every choice."
        if self.distribution == "choice":
            assert self.choices is not None
            return list(self.choices)

        return [self.low, self.high]


class SearchSpace:
    """
    JSON format, one entry per Config field:

        {"learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
         "top_k": {"type": "int", "low": 2, "high": 7},
         "bank_size": {"type": "choice", "choices": [512, 1024, 2048]}}

    "int" bounds are inclusive. Everything is validated at construction.
    """

    def __init__(self, definition: dict[str, Any]) -> None:
        if not isinstance(definition, dict) or len(definition) == 0:
            raise ValueError("Search space must be a non-empty JSON object mapping Config keys to distributions")

        self.parameter_specifications: list[ParameterSpecification] = [
            self._parse_entry(name, entry) for name, entry in sorted(definition.items())
        ]
        logger.debug(f"Search space ready with {len(self.parameter_specifications)} parameters: {self.parameter_names}")

    @classmethod
    def from_json(cls, search_space_path: str) -> "SearchSpace":
        with open(search_space_path, "r") as search_space_file:
            definition: dict[str, Any] = json.load(search_space_file)
        logger.debug(f"Read search space with {len(definition)} entries from {search_space_path}")

        return cls(definition)

    @property
    def parameter_names(self) -> list[str]:
        return [specification.name for specification in self.parameter_specifications]

    @classmethod
    def _parse_entry(cls, name: str, entry: Any) -> ParameterSpecification:
        valid_names: list[str] = Config._attribute_names()
        if name not in valid_names:
            raise KeyError(f"Search space key '{name}' is not a Config field. Valid keys: {valid_names}")
        if name in RESERVED_KEYS:
            raise ValueError(f"Search space key '{name}' is managed by the optimiser and cannot be searched")
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError(f"Search space entry '{name}' must be an object with a 'type' of {DISTRIBUTION_TYPES}")

        distribution: str = entry["type"]
        if distribution not in DISTRIBUTION_TYPES:
            raise ValueError(f"Search space entry '{name}' has unknown type '{distribution}', expected one of {DISTRIBUTION_TYPES}")

        allowed_fields: set[str] = {"type", "choices"} if distribution == "choice" else {"type", "low", "high"}
        unexpected_fields: set[str] = set(entry) - allowed_fields
        if unexpected_fields:
            raise ValueError(f"Search space entry '{name}' of type '{distribution}' has unexpected fields {sorted(unexpected_fields)}")

        specification: ParameterSpecification = (
            cls._parse_choice(name, entry) if distribution == "choice" else cls._parse_range(name, distribution, entry)
        )
        cls._check_against_config(specification)
        logger.debug(f"Parsed search space entry {specification}")

        return specification

    @staticmethod
    def _parse_choice(name: str, entry: dict[str, Any]) -> ParameterSpecification:
        choices: Any = entry.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise ValueError(f"Search space entry '{name}' of type 'choice' needs a non-empty 'choices' list")

        return ParameterSpecification(name=name, distribution="choice", choices=choices)

    @staticmethod
    def _parse_range(name: str, distribution: str, entry: dict[str, Any]) -> ParameterSpecification:
        for bound_name in ("low", "high"):
            bound: Any = entry.get(bound_name)
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(f"Search space entry '{name}' needs a numeric '{bound_name}', got {bound!r}")

        low: float = entry["low"]
        high: float = entry["high"]
        if distribution == "int" and not (isinstance(low, int) and isinstance(high, int)):
            raise ValueError(f"Search space entry '{name}' of type 'int' needs integer bounds, got low={low!r} high={high!r}")
        if low >= high and not (distribution == "int" and low == high):
            raise ValueError(f"Search space entry '{name}' needs low < high, got low={low} high={high}")
        if distribution == "loguniform" and low <= 0:
            raise ValueError(f"Search space entry '{name}' of type 'loguniform' needs low > 0, got {low}")

        return ParameterSpecification(name=name, distribution=distribution, low=low, high=high)

    @staticmethod
    def _check_against_config(specification: ParameterSpecification) -> None:
        for probe_value in specification.probe_values():
            try:
                Config(**{specification.name: probe_value})
            except (TypeError, ValueError) as error:
                raise type(error)(
                    f"Search space entry '{specification.name}' produces value {probe_value!r} rejected by Config: {error}"
                ) from error
