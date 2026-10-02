from pathlib import Path

import h5py
import numpy as np
import torch

from astrouda.config import Config
from astrouda.data.batch import DomainData
from astrouda.logging_utils import get_logger

ZENODO_URL: str = "https://zenodo.org/records/7473597"


class GalaxyZoo2Loader:
    """
    Loads the Galaxy Zoo 2 HDF5 files of the Zenodo record (downloading is not implemented).

    Experiments (Config.dataset_name) and their default files inside Config.data_directory, source then target:
        gz2_sdss_decals    sdss_1.h5, decals.h5
        gz2_sdss_wide_deep sdss_2.h5, sdss_stripe82.h5
    Config.gz2_source_file and Config.gz2_target_file override the file names.
    The record lists these four files but does not say which SDSS file belongs to which experiment;
    the pairing above is an assumption.

    Layout assumption (not confirmed from the record, hence configurable): each file holds one HDF5 dataset of
    images and one of integer labels, named Config.gz2_image_key and Config.gz2_label_key. If a key contains
    "{part}" it is expanded for each name in Config.gz2_part_names and the parts are concatenated, because the
    train/validation/test split is redone here with Config fractions.
    Images are (N, C, H, W) or (N, H, W, C) and are returned channels-first; Config.gz2_channel_order picks the
    file channel for each output channel, which must come out as i, r, g.
    Labels are (N,) integers or one-hot (N, K); they must lie in [0, Config.number_of_classes).
    Config.number_of_classes is free (9 or 10); Config.gz2_class_names is only used to name classes in logs.

    Normalisation to [0, 1] (paper): uint8 is divided by 255, any other dtype is min-max scaled with the global
    minimum and maximum of that domain (design choice). Images stay at native resolution.
    """

    EXPERIMENT_FILES: dict[str, tuple[str, str]] = {
        "gz2_sdss_decals": ("sdss_1.h5", "decals.h5"),
        "gz2_sdss_wide_deep": ("sdss_2.h5", "sdss_stripe82.h5"),
    }

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.logger = get_logger(__name__, config)

    def domain_file_path(self, domain_identifier: int) -> Path:
        override_name: str | None = self.config.gz2_source_file if domain_identifier == 0 else self.config.gz2_target_file
        file_name: str = (
            override_name
            if override_name is not None
            else self.EXPERIMENT_FILES[self.config.dataset_name][domain_identifier]
        )

        return Path(self.config.data_directory) / file_name

    def _check_files_exist(self) -> None:
        expected_paths: list[Path] = [self.domain_file_path(0), self.domain_file_path(1)]
        missing_paths: list[Path] = [path for path in expected_paths if not path.is_file()]

        if missing_paths:
            raise FileNotFoundError(
                f"Missing Galaxy Zoo 2 files for '{self.config.dataset_name}': {[str(path) for path in missing_paths]}. "
                f"Expected {[str(path) for path in expected_paths]}. "
                f"Download them from {ZENODO_URL} into {Path(self.config.data_directory).resolve()}"
            )

    def _expanded_keys(self, key_template: str) -> list[str]:
        if "{part}" not in key_template:
            return [key_template]

        return [key_template.format(part=part_name) for part_name in self.config.gz2_part_names]

    def _read_arrays(self, file_path: Path, key_template: str) -> np.ndarray:
        arrays: list[np.ndarray] = []

        with h5py.File(file_path, "r") as hdf5_file:
            for key in self._expanded_keys(key_template):
                if key not in hdf5_file:
                    raise KeyError(
                        f"{file_path} has no dataset '{key}'; it contains {list(hdf5_file.keys())}. "
                        "Set gz2_image_key, gz2_label_key and gz2_part_names in the config to match."
                    )
                self.logger.debug(f"Reading '{key}' from {file_path.name}: shape {hdf5_file[key].shape}")
                arrays.append(np.asarray(hdf5_file[key]))

        return np.concatenate(arrays)

    def _to_channels_first(self, images: np.ndarray, file_path: Path) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError(f"{file_path} images must be a 4D array, got shape {images.shape}")

        if images.shape[1] != self.config.number_of_image_channels and images.shape[-1] == self.config.number_of_image_channels:
            self.logger.debug(f"{file_path.name} looks channels-last {images.shape}, transposing")
            images = np.transpose(images, (0, 3, 1, 2))
        if images.shape[1] != self.config.number_of_image_channels:
            raise ValueError(f"{file_path} has shape {images.shape}, expected {self.config.number_of_image_channels} channels")
        if sorted(self.config.gz2_channel_order) != list(range(self.config.number_of_image_channels)):
            raise ValueError(f"gz2_channel_order {self.config.gz2_channel_order} is not a permutation of the image channels")

        return images[:, self.config.gz2_channel_order]

    def _normalise(self, images: np.ndarray) -> np.ndarray:
        if images.dtype == np.uint8:
            return images.astype(np.float32) / 255.0

        images = images.astype(np.float32)
        minimum_value: float = float(images.min())
        maximum_value: float = float(images.max())
        self.logger.debug(f"Min-max normalising from [{minimum_value}, {maximum_value}]")

        return (images - minimum_value) / max(maximum_value - minimum_value, 1e-12)

    def load_domain(self, domain_identifier: int) -> DomainData:
        """domain_identifier 0 is the source domain, 1 the target domain."""
        self._check_files_exist()
        file_path: Path = self.domain_file_path(domain_identifier)

        images: np.ndarray = self._to_channels_first(self._read_arrays(file_path, self.config.gz2_image_key), file_path)
        labels: np.ndarray = self._read_arrays(file_path, self.config.gz2_label_key)
        if labels.ndim == 2:
            labels = labels.argmax(axis=1)
        if labels.ndim != 1 or len(labels) != len(images):
            raise ValueError(f"{file_path} has labels of shape {labels.shape} for {len(images)} images")

        labels = labels.astype(np.int64)
        if labels.min() < 0 or labels.max() >= self.config.number_of_classes:
            raise ValueError(
                f"{file_path} labels span [{labels.min()}, {labels.max()}] but number_of_classes is {self.config.number_of_classes}"
            )

        normalised_images: np.ndarray = self._normalise(images)
        class_counts: list[int] = np.bincount(labels, minlength=self.config.number_of_classes).tolist()
        named_counts: dict[str, int] = {
            (self.config.gz2_class_names[index] if index < len(self.config.gz2_class_names) else str(index)): count
            for index, count in enumerate(class_counts)
        }
        self.logger.debug(f"{file_path.name}: images {normalised_images.shape}, class counts {named_counts}")

        return DomainData(images=torch.from_numpy(np.ascontiguousarray(normalised_images)), labels=torch.from_numpy(labels))
