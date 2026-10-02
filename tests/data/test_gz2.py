from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest
import torch

from astrouda.config import Config
from astrouda.data import DomainAdaptationBatch, DomainData, GalaxyZoo2Loader, build_data_loaders

EXPERIMENT_FILE_NAMES: dict[str, tuple[str, str]] = {
    "gz2_sdss_decals": ("sdss_1.h5", "decals.h5"),
    "gz2_sdss_wide_deep": ("sdss_2.h5", "sdss_stripe82.h5"),
}


def write_domain_file(
    file_path: Path,
    sample_count: int = 40,
    image_size: int = 12,
    class_count: int = 10,
    channels_last: bool = False,
    seed: int = 0,
    label_key: str = "labels",
    image_key: str = "images",
) -> np.ndarray:
    random_generator: np.random.Generator = np.random.default_rng(seed)
    images: np.ndarray = random_generator.integers(0, 256, size=(sample_count, 3, image_size, image_size), dtype=np.uint8)
    labels: np.ndarray = np.arange(sample_count) % class_count

    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file[image_key] = np.transpose(images, (0, 2, 3, 1)) if channels_last else images
        hdf5_file[label_key] = labels

    return images


def write_experiment(directory: Path, dataset_name: str, **file_options: object) -> list[np.ndarray]:
    return [
        write_domain_file(directory / file_name, seed=seed, **file_options)
        for seed, file_name in enumerate(EXPERIMENT_FILE_NAMES[dataset_name])
    ]


def gz2_config(directory: Path, dataset_name: str, **overrides: object) -> Config:
    default_overrides: dict[str, object] = dict(
        dataset_name=dataset_name,
        data_directory=str(directory),
        number_of_classes=10,
        native_image_size=12,
        crop_size=12,
        augmentation_zoom_size=14,
        batch_size=4,
    )
    default_overrides.update(overrides)

    return Config(**default_overrides)


@pytest.mark.parametrize("dataset_name", list(EXPERIMENT_FILE_NAMES))
def test_both_experiments_load(tmp_path: Path, dataset_name: str) -> None:
    written_images: list[np.ndarray] = write_experiment(tmp_path, dataset_name)
    loader: GalaxyZoo2Loader = GalaxyZoo2Loader(gz2_config(tmp_path, dataset_name))

    for domain_identifier in (0, 1):
        domain: DomainData = loader.load_domain(domain_identifier)

        assert domain.images.shape == (40, 3, 12, 12)
        assert domain.images.dtype == torch.float32
        assert domain.images.min() >= 0.0 and domain.images.max() <= 1.0
        assert domain.labels.dtype == torch.int64
        assert domain.labels.min() == 0 and domain.labels.max() == 9
        assert torch.allclose(domain.images, torch.from_numpy(written_images[domain_identifier]).float() / 255.0)


def test_channels_last_is_transposed(tmp_path: Path) -> None:
    written_images: list[np.ndarray] = write_experiment(tmp_path, "gz2_sdss_decals", channels_last=True)
    domain: DomainData = GalaxyZoo2Loader(gz2_config(tmp_path, "gz2_sdss_decals")).load_domain(0)

    assert domain.images.shape == (40, 3, 12, 12)
    assert torch.allclose(domain.images, torch.from_numpy(written_images[0]).float() / 255.0)


def test_channel_order_is_permuted(tmp_path: Path) -> None:
    written_images: list[np.ndarray] = write_experiment(tmp_path, "gz2_sdss_decals")
    config: Config = gz2_config(tmp_path, "gz2_sdss_decals", gz2_channel_order=[2, 1, 0])
    domain: DomainData = GalaxyZoo2Loader(config).load_domain(0)

    assert torch.allclose(domain.images[:, 0], torch.from_numpy(written_images[0][:, 2]).float() / 255.0)
    assert torch.allclose(domain.images[:, 2], torch.from_numpy(written_images[0][:, 0]).float() / 255.0)


def test_invalid_channel_order_raises(tmp_path: Path) -> None:
    write_experiment(tmp_path, "gz2_sdss_decals")

    with pytest.raises(ValueError):
        GalaxyZoo2Loader(gz2_config(tmp_path, "gz2_sdss_decals", gz2_channel_order=[0, 0, 1])).load_domain(0)


def test_file_name_overrides_and_custom_keys(tmp_path: Path) -> None:
    write_domain_file(tmp_path / "mine_a.h5", image_key="x", label_key="y")
    write_domain_file(tmp_path / "mine_b.h5", image_key="x", label_key="y", seed=1)
    config: Config = gz2_config(
        tmp_path, "gz2_sdss_decals", gz2_source_file="mine_a.h5", gz2_target_file="mine_b.h5", gz2_image_key="x", gz2_label_key="y"
    )

    assert len(GalaxyZoo2Loader(config).load_domain(1)) == 40


def test_part_keys_are_concatenated(tmp_path: Path) -> None:
    for file_name in EXPERIMENT_FILE_NAMES["gz2_sdss_decals"]:
        with h5py.File(tmp_path / file_name, "w") as hdf5_file:
            for part_name, part_size in (("train", 6), ("val", 2), ("test", 2)):
                hdf5_file[f"images_{part_name}"] = np.zeros((part_size, 3, 12, 12), dtype=np.uint8)
                hdf5_file[f"labels_{part_name}"] = np.arange(part_size) % 10
    config: Config = gz2_config(tmp_path, "gz2_sdss_decals", gz2_image_key="images_{part}", gz2_label_key="labels_{part}")

    assert len(GalaxyZoo2Loader(config).load_domain(0)) == 10


def test_missing_dataset_key_lists_available_keys(tmp_path: Path) -> None:
    write_experiment(tmp_path, "gz2_sdss_decals", image_key="pixels")

    with pytest.raises(KeyError, match="pixels"):
        GalaxyZoo2Loader(gz2_config(tmp_path, "gz2_sdss_decals")).load_domain(0)


def test_nine_class_handling(tmp_path: Path) -> None:
    write_experiment(tmp_path, "gz2_sdss_decals", class_count=9)
    domain: DomainData = GalaxyZoo2Loader(gz2_config(tmp_path, "gz2_sdss_decals", number_of_classes=9)).load_domain(0)

    assert domain.labels.max() == 8


def test_label_out_of_range_raises(tmp_path: Path) -> None:
    write_experiment(tmp_path, "gz2_sdss_decals", class_count=10)

    with pytest.raises(ValueError):
        GalaxyZoo2Loader(gz2_config(tmp_path, "gz2_sdss_decals", number_of_classes=9)).load_domain(0)


def test_one_hot_labels_are_accepted(tmp_path: Path) -> None:
    write_experiment(tmp_path, "gz2_sdss_decals")
    with h5py.File(tmp_path / "sdss_1.h5", "r+") as hdf5_file:
        one_hot_labels: np.ndarray = np.eye(10, dtype=np.int64)[np.asarray(hdf5_file["labels"])]
        del hdf5_file["labels"]
        hdf5_file["labels"] = one_hot_labels
    domain: DomainData = GalaxyZoo2Loader(gz2_config(tmp_path, "gz2_sdss_decals")).load_domain(0)

    assert domain.labels.shape == (40,)
    assert domain.labels.max() == 9


def test_float_images_are_min_max_scaled(tmp_path: Path) -> None:
    write_experiment(tmp_path, "gz2_sdss_decals")
    with h5py.File(tmp_path / "sdss_1.h5", "r+") as hdf5_file:
        float_images: np.ndarray = np.asarray(hdf5_file["images"]).astype(np.float32) * 3.0 - 100.0
        del hdf5_file["images"]
        hdf5_file["images"] = float_images
    domain: DomainData = GalaxyZoo2Loader(gz2_config(tmp_path, "gz2_sdss_decals")).load_domain(0)

    assert domain.images.min() == 0.0 and domain.images.max() == 1.0


@pytest.mark.parametrize("dataset_name", list(EXPERIMENT_FILE_NAMES))
def test_missing_files_error_names_files_and_zenodo(tmp_path: Path, dataset_name: str) -> None:
    with pytest.raises(FileNotFoundError) as error_info:
        GalaxyZoo2Loader(gz2_config(tmp_path, dataset_name)).load_domain(0)

    for file_name in EXPERIMENT_FILE_NAMES[dataset_name]:
        assert file_name in str(error_info.value)
    assert "zenodo.org/records/7473597" in str(error_info.value)


def test_one_missing_file_is_reported(tmp_path: Path) -> None:
    write_domain_file(tmp_path / "sdss_2.h5")

    with pytest.raises(FileNotFoundError, match="sdss_stripe82.h5"):
        GalaxyZoo2Loader(gz2_config(tmp_path, "gz2_sdss_wide_deep")).load_domain(0)


@pytest.mark.parametrize("dataset_name", list(EXPERIMENT_FILE_NAMES))
def test_build_data_loaders_dispatches_gz2(tmp_path: Path, dataset_name: str) -> None:
    write_experiment(tmp_path, dataset_name)
    batch: DomainAdaptationBatch = next(iter(build_data_loaders(gz2_config(tmp_path, dataset_name)).train))

    assert batch.source_images.shape == (4, 3, 12, 12)
    assert batch.target_images.shape == (4, 3, 12, 12)


def test_smoke_10class_config_yields_ten_class_batches() -> None:
    config: Config = Config("configs/smoke_10class.json")
    data_loaders = build_data_loaders(config)
    seen_labels: Optional[torch.Tensor] = None

    for batch in data_loaders.train:
        seen_labels = batch.source_labels if seen_labels is None else torch.cat([seen_labels, batch.source_labels])

    assert config.top_k == 7 and config.number_of_classes == 10
    assert data_loaders.number_of_classes == 10
    assert len(data_loaders.source_train_class_counts) == 10
    assert int(data_loaders.source_train_class_counts.min()) > 0
    assert seen_labels is not None and int(seen_labels.max()) <= 9 and seen_labels.shape[0] > 0


@pytest.mark.parametrize("config_name", ["gz2_sdss_decals", "gz2_sdss_wide_deep", "smoke_10class"])
def test_new_configs_load(config_name: str) -> None:
    config: Config = Config(f"configs/{config_name}.json")

    assert config.number_of_classes == 10
    assert config.top_k == 7
    assert config.learning_rate_decay_every_epochs == 10


def test_full_config_domain_names() -> None:
    decals_config: Config = Config("configs/gz2_sdss_decals.json")
    deep_config: Config = Config("configs/gz2_sdss_wide_deep.json")

    assert (decals_config.source_domain_name, decals_config.target_domain_name) == ("SDSS", "DECaLS")
    assert (deep_config.source_domain_name, deep_config.target_domain_name) == ("SDSS Wide", "SDSS Deep (Stripe 82)")
