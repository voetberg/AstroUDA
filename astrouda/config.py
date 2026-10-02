import copy
import json
import logging
import types
import typing
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

logger: logging.Logger = logging.getLogger(__name__)


class Config:
    """
    Every tunable of the project lives here as a typed class attribute with its default.

    Values are overwritten from a JSON file and/or keyword overrides.
    Unknown keys and wrongly typed values raise, so a typo in a JSON never silently falls back to a default.
    Settings that the paper (arXiv:2302.02005) does not specify are marked "design choice".
    """

    # ---- Run ----
    experiment_name: str = "astrouda"
    output_directory: str = "./outputs"
    random_seed: int = 0
    log_level: str = "DEBUG"
    device: str = "auto"  # "auto", "cpu", "cuda" or "mps"
    use_mixed_precision: bool = False
    number_of_dataloader_workers: int = 0
    pin_memory: bool = False

    # ---- Data ----
    dataset_name: str = "lsst"  # "lsst", "gz2_sdss_decals", "gz2_sdss_wide_deep" or "synthetic"
    source_domain_name: str = "LSST Y1"
    target_domain_name: str = "LSST Y10"
    data_directory: str = "./data/lsst"
    number_of_classes: int = 3
    number_of_image_channels: int = 3
    native_image_size: int = 100
    crop_size: int = 256  # paper: 256x256 crops
    train_fraction: float = 0.6  # paper: 60/20/20 split
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    split_seed: int = 0
    batch_size: int = 64
    synthetic_samples_per_domain: int = 256  # only used by dataset_name == "synthetic"

    # ---- Augmentation ----
    use_augmented_views: bool = True
    augmentation_rotation_degrees: int = 90  # paper: 90 degree rotation
    augmentation_zoom_size: int = 300  # paper: zoom to 300 then crop back to crop_size

    # ---- Model ----
    backbone_depth: int = 50  # paper: ResNet50, random initialisation. 18 for smoke tests
    use_small_input_stem: bool = False  # 3x3 stride-1 stem with no max-pool, for tiny images
    use_domain_specific_batch_norm: bool = True  # design choice: one BatchNorm per domain, own statistics and affine
    number_of_domains: int = 2  # 0 = source, 1 = target

    # ---- Losses ----
    enable_domain_adaptation: bool = True  # False gives the source-only baseline (CE only)
    domain_adaptation_weight: float = 0.005  # paper: lambda
    top_k: int = 3  # paper: 3 for 3 classes, 7 for 10 classes
    bank_size: int = 2048  # design choice: FIFO bank of detached probabilities, in samples, source and target
    probability_epsilon: float = 1e-8  # numerical floor for log and BCE

    # ---- Entropy separation tuner (Algorithm 1) ----
    entropy_boundary_initial: Optional[float] = None  # None -> log(K) / 2
    confidence_margin_initial: Optional[float] = None  # None -> 0.4 if K <= 5 else 1.3
    entropy_step_values: list[float] = [0.3, -0.3, 0.5, -0.5]
    entropy_boundary_patience_epochs: int = 5
    confidence_margin_patience_epochs: int = 2
    entropy_boundary_floor: float = 0.0  # design choice: rho is clamped to >= this
    confidence_margin_floor: float = 0.05  # design choice: m is clamped to >= this, so the margin stays positive
    tuner_reference_loss: str = "train_total_loss"  # design choice: L_min is the minimum training total loss

    # ---- Optimisation ----
    learning_rate: float = 0.001
    momentum: float = 0.9
    use_nesterov: bool = True
    weight_decay: float = 1e-4
    learning_rate_decay_factor: float = 0.1
    learning_rate_decay_every_epochs: int = 10  # paper: 10 for 10-class runs, 7 for the 3-class run
    maximum_epochs: int = 100
    early_stopping_patience_epochs: int = 12
    early_stopping_metric: str = "target_validation_accuracy"
    checkpoint_every_epochs: int = 1
    resume_from_checkpoint: Optional[str] = None

    # ---- Evaluation ----
    number_of_seeds: int = 5

    def __init__(self, config_path: Optional[str] = None, **overrides: Any) -> None:
        for attribute_name in self._attribute_names():
            default_value: Any = getattr(type(self), attribute_name)
            setattr(self, attribute_name, copy.deepcopy(default_value))

        if config_path is not None:
            with open(config_path, "r") as config_file:
                json_values: dict[str, Any] = json.load(config_file)
            logger.debug(f"Read {len(json_values)} settings from {config_path}")
            self._apply(json_values)

        self._apply(overrides)
        self._validate_fractions()
        logger.debug(f"Config ready: {self.to_dictionary()}")

    @classmethod
    def _attribute_names(cls) -> list[str]:
        return list(get_type_hints(cls).keys())

    def _apply(self, values: dict[str, Any]) -> None:
        valid_names: list[str] = self._attribute_names()
        type_hints: dict[str, Any] = get_type_hints(type(self))

        for name, value in values.items():
            if name not in valid_names:
                raise KeyError(f"Unknown config key '{name}'. Valid keys: {valid_names}")
            if not self._matches_type(value, type_hints[name]):
                raise TypeError(f"Config key '{name}' expects {type_hints[name]}, got {type(value).__name__}: {value!r}")
            setattr(self, name, value)

    @classmethod
    def _matches_type(cls, value: Any, expected_type: Any) -> bool:
        origin: Any = get_origin(expected_type)

        if origin is Union or origin is types.UnionType:
            return any(cls._matches_type(value, option) for option in get_args(expected_type))
        if expected_type is type(None):
            return value is None
        if origin is list:
            (element_type,) = get_args(expected_type)
            return isinstance(value, list) and all(cls._matches_type(element, element_type) for element in value)
        if expected_type is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type is int:
            return isinstance(value, int) and not isinstance(value, bool)
        return isinstance(value, expected_type)

    def _validate_fractions(self) -> None:
        fraction_total: float = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(fraction_total - 1.0) > 1e-6:
            raise ValueError(f"train/validation/test fractions must sum to 1, got {fraction_total}")

    def to_dictionary(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._attribute_names()}

    def to_json(self, output_path: str) -> None:
        with open(output_path, "w") as output_file:
            json.dump(self.to_dictionary(), output_file, indent=2)

    def __repr__(self) -> str:
        return f"Config({self.to_dictionary()})"
