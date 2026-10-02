from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn.functional as functional

from astrouda.config import Config
from astrouda.data.augmentation import TwoViewAugmentation
from astrouda.data.batch import DomainAdaptationBatch, DomainData
from astrouda.data.lsst import LSSTLoader
from astrouda.data.splits import SplitIndices, stratified_split
from astrouda.data.synthetic import SyntheticGenerator
from astrouda.logging_utils import get_logger


class DomainAdaptationLoader:
    """
    Iterable of DomainAdaptationBatch over one split of a source domain and a target domain.

    Training mode (shuffle=True): every epoch the source and target indices are shuffled by two separate
    generators, seeded by (Config.random_seed, epoch, domain), so a source batch and a target batch never share
    a pairing. An epoch has max(source, target samples) // batch_size batches (at least one); the smaller domain
    wraps around into a fresh permutation. Target labels are left out of the batch. The epoch counter advances
    each time the loader is iterated, so a run is reproducible. Call set_epoch to jump to a given epoch.

    Evaluation mode (shuffle=False): both domains are walked in order, Config.batch_size at a time, covering
    each sample exactly once; target labels are included and no views are made.

    Images are resized from native_image_size to crop_size when the batch is built.
    """

    def __init__(
        self,
        config: Config,
        source_data: DomainData,
        target_data: DomainData,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        shuffle: bool,
        augmentation: Optional[TwoViewAugmentation],
    ) -> None:
        self.config: Config = config
        self.logger = get_logger(__name__, config)
        self.source_data: DomainData = source_data
        self.target_data: DomainData = target_data
        self.source_indices: torch.Tensor = source_indices
        self.target_indices: torch.Tensor = target_indices
        self.shuffle: bool = shuffle
        self.augmentation: Optional[TwoViewAugmentation] = augmentation
        self.epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        larger_domain_size: int = max(len(self.source_indices), len(self.target_indices))
        if self.shuffle:
            return max(1, larger_domain_size // self.config.batch_size)

        return -(-larger_domain_size // self.config.batch_size)

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] == self.config.crop_size and images.shape[-2] == self.config.crop_size:
            return images
        if images.shape[0] == 0:
            return images.new_zeros((0, images.shape[1], self.config.crop_size, self.config.crop_size))

        resized_images: torch.Tensor = functional.interpolate(
            images, size=(self.config.crop_size, self.config.crop_size), mode="bilinear", align_corners=False, antialias=True
        )

        return resized_images.clamp(0.0, 1.0)

    def _shuffled_positions(self, domain_size: int, domain_identifier: int, batch_count: int) -> torch.Tensor:
        random_generator: torch.Generator = torch.Generator().manual_seed(
            self.config.random_seed * 1_000_003 + self.epoch * 1009 + domain_identifier
        )
        needed_count: int = batch_count * self.config.batch_size
        permutations: list[torch.Tensor] = []
        while sum(len(permutation) for permutation in permutations) < needed_count:
            permutations.append(torch.randperm(domain_size, generator=random_generator))

        return torch.cat(permutations)[:needed_count]

    def _build_batch(self, source_positions: torch.Tensor, target_positions: torch.Tensor) -> DomainAdaptationBatch:
        source_rows: torch.Tensor = self.source_indices[source_positions]
        target_rows: torch.Tensor = self.target_indices[target_positions]

        source_images: torch.Tensor = self._prepare_images(self.source_data.images[source_rows])
        target_images: torch.Tensor = self._prepare_images(self.target_data.images[target_rows])

        source_view_one: Optional[torch.Tensor] = None
        source_view_two: Optional[torch.Tensor] = None
        target_view_one: Optional[torch.Tensor] = None
        target_view_two: Optional[torch.Tensor] = None
        if self.shuffle and self.augmentation is not None:
            source_view_one, source_view_two = self.augmentation(source_images)
            target_view_one, target_view_two = self.augmentation(target_images)

        return DomainAdaptationBatch(
            source_images=source_images,
            source_labels=self.source_data.labels[source_rows],
            target_images=target_images,
            target_labels=None if self.shuffle else self.target_data.labels[target_rows],
            source_view_one=source_view_one,
            source_view_two=source_view_two,
            target_view_one=target_view_one,
            target_view_two=target_view_two,
            source_indices=source_positions,
            target_indices=target_positions,
        )

    def __iter__(self) -> Iterator[DomainAdaptationBatch]:
        batch_count: int = len(self)
        self.logger.debug(f"Iterating {batch_count} batches, shuffle={self.shuffle}, epoch={self.epoch}")

        if self.shuffle:
            source_order: torch.Tensor = self._shuffled_positions(len(self.source_indices), 0, batch_count)
            target_order: torch.Tensor = self._shuffled_positions(len(self.target_indices), 1, batch_count)
        else:
            source_order = torch.arange(len(self.source_indices))
            target_order = torch.arange(len(self.target_indices))

        for batch_index in range(batch_count):
            batch_slice: slice = slice(batch_index * self.config.batch_size, (batch_index + 1) * self.config.batch_size)
            yield self._build_batch(source_order[batch_slice], target_order[batch_slice])

        if self.shuffle:
            self.epoch += 1


@dataclass
class DataLoaders:
    """
    train, validation, test: DomainAdaptationLoader for each split.
    source_train_class_counts: int64 tensor (number_of_classes,), labelled source training samples per class,
        for computing class weights.
    number_of_classes, source_train_size, target_train_size: bookkeeping.
    """

    train: DomainAdaptationLoader
    validation: DomainAdaptationLoader
    test: DomainAdaptationLoader
    source_train_class_counts: torch.Tensor
    number_of_classes: int
    source_train_size: int
    target_train_size: int


def load_domains(config: Config) -> tuple[DomainData, DomainData]:
    logger = get_logger(__name__, config)
    logger.debug(f"Loading dataset '{config.dataset_name}'")

    if config.dataset_name == "synthetic":
        generator: SyntheticGenerator = SyntheticGenerator(config)

        return generator.generate(0), generator.generate(1)
    if config.dataset_name == "lsst":
        lsst_loader: LSSTLoader = LSSTLoader(config)

        return lsst_loader.load_domain(config.source_domain_name), lsst_loader.load_domain(config.target_domain_name)

    raise NotImplementedError(f"dataset_name '{config.dataset_name}' has no loader yet; use 'lsst' or 'synthetic'")


def build_data_loaders(config: Config) -> DataLoaders:
    """
    Build train, validation and test loaders from Config.
    Splits are stratified, seeded by Config.split_seed and made separately for the source and the target domain.
    """
    logger = get_logger(__name__, config)
    source_data, target_data = load_domains(config)

    source_split: SplitIndices = stratified_split(source_data.labels, 0, config)
    target_split: SplitIndices = stratified_split(target_data.labels, 1, config)
    augmentation: Optional[TwoViewAugmentation] = TwoViewAugmentation(config) if config.use_augmented_views else None

    def make_loader(source_indices: torch.Tensor, target_indices: torch.Tensor, shuffle: bool) -> DomainAdaptationLoader:
        return DomainAdaptationLoader(
            config, source_data, target_data, source_indices, target_indices, shuffle, augmentation if shuffle else None
        )

    source_train_class_counts: torch.Tensor = torch.bincount(
        source_data.labels[source_split.train], minlength=config.number_of_classes
    )
    logger.debug(f"Source training class counts: {source_train_class_counts.tolist()}")

    return DataLoaders(
        train=make_loader(source_split.train, target_split.train, True),
        validation=make_loader(source_split.validation, target_split.validation, False),
        test=make_loader(source_split.test, target_split.test, False),
        source_train_class_counts=source_train_class_counts,
        number_of_classes=config.number_of_classes,
        source_train_size=len(source_split.train),
        target_train_size=len(target_split.train),
    )
