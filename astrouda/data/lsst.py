import re
from pathlib import Path

import numpy as np
import torch

from astrouda.config import Config
from astrouda.data.batch import DomainData
from astrouda.logging_utils import get_logger

PART_NAMES: tuple[str, ...] = ("train", "validation", "val", "valid", "test")


class LSSTLoader:
    """
    Loads the Zenodo LSST Y1/Y10 arrays from Config.data_directory (downloading is not implemented).

    Expected files, looked for in the directory itself and then in its immediate subdirectories:
        images_<tag>_<part>.npy   shape (N, 3, 100, 100), for example images_Y1_train.npy
        labels_<part>.npy         shape (N,), integer classes (a one-hot (N, K) array is accepted)
    where <tag> is the "Y<number>" found in the domain name ("LSST Y1" -> "Y1") and <part> is one of
    train, validation/val/valid, test. At least the train part must exist; every part that exists is concatenated,
    because the split into train/validation/test is redone here with Config fractions.
    Channels-last arrays (N, H, W, C) are transposed to channels-first.

    Normalisation to [0, 1] (paper): uint8 arrays are divided by 255; any other dtype is min-max scaled with
    the global minimum and maximum of that domain (design choice, the paper does not give the recipe).
    Images stay at native resolution here, resizing to crop_size happens when batches are built.
    """

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.logger = get_logger(__name__, config)

    @staticmethod
    def domain_tag(domain_name: str) -> str:
        tag_match = re.search(r"Y\d+", domain_name)
        if tag_match is None:
            raise ValueError(f"Cannot find a 'Y<number>' tag such as 'Y1' in the domain name '{domain_name}'")

        return tag_match.group(0)

    def _search_directories(self) -> list[Path]:
        root_directory: Path = Path(self.config.data_directory)
        if not root_directory.is_dir():
            raise FileNotFoundError(f"LSST data directory does not exist: {root_directory.resolve()}")

        subdirectories: list[Path] = sorted(path for path in root_directory.iterdir() if path.is_dir())

        return [root_directory, *subdirectories]

    def _find_file(self, file_name: str) -> Path | None:
        for directory in self._search_directories():
            candidate_path: Path = directory / file_name
            if candidate_path.is_file():
                self.logger.debug(f"Found {file_name} at {candidate_path}")
                return candidate_path

        return None

    def _find_parts(self, domain_name: str) -> list[tuple[Path, Path]]:
        tag: str = self.domain_tag(domain_name)
        found_pairs: list[tuple[Path, Path]] = []
        missing_files: list[str] = []

        for part_name in PART_NAMES:
            image_path: Path | None = self._find_file(f"images_{tag}_{part_name}.npy")
            label_path: Path | None = self._find_file(f"labels_{part_name}.npy")

            if image_path is not None and label_path is not None:
                found_pairs.append((image_path, label_path))
            elif part_name == "train":
                if image_path is None:
                    missing_files.append(f"images_{tag}_train.npy")
                if label_path is None:
                    missing_files.append("labels_train.npy")
            elif (image_path is None) != (label_path is None):
                missing_files.append(f"labels_{part_name}.npy" if label_path is None else f"images_{tag}_{part_name}.npy")

        if missing_files:
            raise FileNotFoundError(
                f"Missing LSST files for domain '{domain_name}' under {Path(self.config.data_directory).resolve()}: "
                f"{missing_files}. Download the Zenodo arrays into that directory."
            )

        return found_pairs

    def _to_channels_first(self, images: np.ndarray, source_path: Path) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError(f"{source_path} must be a 4D array, got shape {images.shape}")
        if images.shape[1] != self.config.number_of_image_channels and images.shape[-1] == self.config.number_of_image_channels:
            self.logger.debug(f"{source_path} looks channels-last {images.shape}, transposing")
            images = np.transpose(images, (0, 3, 1, 2))
        if images.shape[1] != self.config.number_of_image_channels:
            raise ValueError(
                f"{source_path} has shape {images.shape}, expected {self.config.number_of_image_channels} channels"
            )

        return images

    def _normalise(self, images: np.ndarray) -> np.ndarray:
        if images.dtype == np.uint8:
            return images.astype(np.float32) / 255.0

        images = images.astype(np.float32)
        minimum_value: float = float(images.min())
        maximum_value: float = float(images.max())
        self.logger.debug(f"Min-max normalising from [{minimum_value}, {maximum_value}]")

        return (images - minimum_value) / max(maximum_value - minimum_value, 1e-12)

    def load_domain(self, domain_name: str) -> DomainData:
        image_arrays: list[np.ndarray] = []
        label_arrays: list[np.ndarray] = []

        for image_path, label_path in self._find_parts(domain_name):
            images: np.ndarray = self._to_channels_first(np.load(image_path), image_path)
            labels: np.ndarray = np.load(label_path)
            if labels.ndim == 2:
                labels = labels.argmax(axis=1)
            if labels.ndim != 1 or len(labels) != len(images):
                raise ValueError(f"{label_path} has {labels.shape} labels for {len(images)} images in {image_path}")

            self.logger.debug(f"Loaded {image_path.name}: {images.shape}, dtype {images.dtype}")
            image_arrays.append(images)
            label_arrays.append(labels.astype(np.int64))

        all_labels: np.ndarray = np.concatenate(label_arrays)
        if all_labels.min() < 0 or all_labels.max() >= self.config.number_of_classes:
            raise ValueError(
                f"Labels span [{all_labels.min()}, {all_labels.max()}] but number_of_classes is {self.config.number_of_classes}"
            )

        normalised_images: np.ndarray = self._normalise(np.concatenate(image_arrays))
        self.logger.debug(f"Domain '{domain_name}': {normalised_images.shape}, class counts {np.bincount(all_labels).tolist()}")

        return DomainData(images=torch.from_numpy(normalised_images), labels=torch.from_numpy(all_labels))
