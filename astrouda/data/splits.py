from dataclasses import dataclass

import numpy as np
import torch

from astrouda.config import Config
from astrouda.logging_utils import get_logger


@dataclass
class SplitIndices:
    "Indices into one domain for each of the three splits. Disjoint and together cover the whole domain."

    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor


def stratified_split(labels: torch.Tensor, domain_identifier: int, config: Config) -> SplitIndices:
    """
    Seeded stratified train/validation/test split of one domain.

    Each class is shuffled with a generator seeded by (Config.split_seed, domain_identifier) and cut with
    Config.train_fraction and Config.validation_fraction. The remainder of each class goes to test.
    Source (domain_identifier 0) and target (1) therefore get independent splits.
    """
    logger = get_logger(__name__, config)
    random_generator: np.random.Generator = np.random.default_rng([config.split_seed, domain_identifier])
    label_array: np.ndarray = labels.cpu().numpy()

    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_index in np.unique(label_array):
        class_positions: np.ndarray = np.flatnonzero(label_array == class_index)
        shuffled_positions: np.ndarray = random_generator.permutation(class_positions)

        train_count: int = int(round(config.train_fraction * len(shuffled_positions)))
        validation_count: int = min(
            int(round(config.validation_fraction * len(shuffled_positions))), len(shuffled_positions) - train_count
        )
        logger.debug(
            f"Domain {domain_identifier} class {class_index}: {len(shuffled_positions)} samples -> "
            f"{train_count} train, {validation_count} validation, "
            f"{len(shuffled_positions) - train_count - validation_count} test"
        )

        train_parts.append(shuffled_positions[:train_count])
        validation_parts.append(shuffled_positions[train_count : train_count + validation_count])
        test_parts.append(shuffled_positions[train_count + validation_count :])

    split_indices: SplitIndices = SplitIndices(
        train=torch.from_numpy(np.sort(np.concatenate(train_parts))).long(),
        validation=torch.from_numpy(np.sort(np.concatenate(validation_parts))).long(),
        test=torch.from_numpy(np.sort(np.concatenate(test_parts))).long(),
    )
    logger.debug(
        f"Domain {domain_identifier} split sizes: {len(split_indices.train)} / "
        f"{len(split_indices.validation)} / {len(split_indices.test)}"
    )

    return split_indices
