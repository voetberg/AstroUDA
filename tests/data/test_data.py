from pathlib import Path

import numpy as np
import pytest
import torch

from astrouda.config import Config
from astrouda.data import (
    DataLoaders,
    DomainAdaptationBatch,
    LSSTLoader,
    SyntheticGenerator,
    TwoViewAugmentation,
    build_data_loaders,
    stratified_split,
)


def make_config(**overrides: object) -> Config:
    default_overrides: dict[str, object] = dict(
        dataset_name="synthetic",
        synthetic_samples_per_domain=100,
        number_of_classes=3,
        native_image_size=16,
        crop_size=16,
        augmentation_zoom_size=20,
        batch_size=8,
    )
    default_overrides.update(overrides)

    return Config(**default_overrides)


def collect_batches(loader: object) -> list[DomainAdaptationBatch]:
    return list(loader)


def test_split_sizes_and_disjointness() -> None:
    config: Config = make_config()
    labels: torch.Tensor = SyntheticGenerator(config).generate(0).labels
    split = stratified_split(labels, 0, config)

    combined: torch.Tensor = torch.cat([split.train, split.validation, split.test])
    assert len(combined) == len(labels)
    assert len(torch.unique(combined)) == len(labels)
    assert abs(len(split.train) / len(labels) - 0.6) < 0.05
    assert abs(len(split.validation) / len(labels) - 0.2) < 0.05
    assert abs(len(split.test) / len(labels) - 0.2) < 0.05


def test_split_is_deterministic_and_seed_dependent() -> None:
    labels: torch.Tensor = SyntheticGenerator(make_config()).generate(0).labels
    first_split = stratified_split(labels, 0, make_config(split_seed=3))
    second_split = stratified_split(labels, 0, make_config(split_seed=3))
    other_split = stratified_split(labels, 0, make_config(split_seed=4))

    assert torch.equal(first_split.train, second_split.train)
    assert not torch.equal(first_split.train, other_split.train)


def test_split_is_stratified() -> None:
    config: Config = make_config()
    labels: torch.Tensor = SyntheticGenerator(config).generate(0).labels
    split = stratified_split(labels, 0, config)

    for class_index in range(config.number_of_classes):
        class_total: int = int((labels == class_index).sum())
        class_train: int = int((labels[split.train] == class_index).sum())
        assert abs(class_train - 0.6 * class_total) <= 1


def test_source_and_target_splits_are_independent() -> None:
    config: Config = make_config()
    labels: torch.Tensor = SyntheticGenerator(config).generate(0).labels

    assert not torch.equal(stratified_split(labels, 0, config).train, stratified_split(labels, 1, config).train)


def test_shapes_and_value_range() -> None:
    config: Config = make_config(native_image_size=12, crop_size=16, augmentation_zoom_size=20)
    data_loaders: DataLoaders = build_data_loaders(config)

    for loader in (data_loaders.train, data_loaders.validation, data_loaders.test):
        for batch in loader:
            for images in (batch.source_images, batch.target_images):
                assert images.shape[1:] == (3, 16, 16)
                assert images.dtype == torch.float32
                assert images.min() >= 0.0 and images.max() <= 1.0


def test_validation_and_test_cover_each_sample_once() -> None:
    data_loaders: DataLoaders = build_data_loaders(make_config())

    for loader in (data_loaders.validation, data_loaders.test):
        batches = collect_batches(loader)
        source_seen: torch.Tensor = torch.cat([batch.source_indices for batch in batches])
        target_seen: torch.Tensor = torch.cat([batch.target_indices for batch in batches])
        assert len(torch.unique(source_seen)) == len(source_seen) == len(loader.source_indices)
        assert len(torch.unique(target_seen)) == len(target_seen) == len(loader.target_indices)


def test_source_target_not_index_coupled() -> None:
    data_loaders: DataLoaders = build_data_loaders(make_config(random_seed=5))
    batches = collect_batches(data_loaders.train)

    aligned_batches: int = sum(bool(torch.equal(batch.source_indices, batch.target_indices)) for batch in batches)
    assert aligned_batches < len(batches)

    source_order: torch.Tensor = torch.cat([batch.source_indices for batch in batches])
    target_order: torch.Tensor = torch.cat([batch.target_indices for batch in batches])
    assert not torch.equal(source_order, target_order)

    repeated_batches = collect_batches(build_data_loaders(make_config(random_seed=5)).train)
    assert torch.equal(source_order, torch.cat([batch.source_indices for batch in repeated_batches]))


def test_epochs_reshuffle() -> None:
    data_loaders: DataLoaders = build_data_loaders(make_config())
    first_epoch = torch.cat([batch.source_indices for batch in data_loaders.train])
    second_epoch = torch.cat([batch.source_indices for batch in data_loaders.train])

    assert not torch.equal(first_epoch, second_epoch)


def test_target_labels_only_outside_training() -> None:
    data_loaders: DataLoaders = build_data_loaders(make_config())

    assert all(batch.target_labels is None for batch in data_loaders.train)
    assert all(batch.source_labels is not None for batch in data_loaders.train)
    for loader in (data_loaders.validation, data_loaders.test):
        for batch in loader:
            assert batch.target_labels is not None
            assert len(batch.target_labels) == len(batch.target_images)


def test_views_present_only_when_enabled_and_for_training() -> None:
    enabled: DataLoaders = build_data_loaders(make_config(use_augmented_views=True))
    train_batch: DomainAdaptationBatch = next(iter(enabled.train))
    for views in (train_batch.source_view_one, train_batch.source_view_two, train_batch.target_view_one, train_batch.target_view_two):
        assert views is not None and views.shape == train_batch.source_images.shape
    assert next(iter(enabled.validation)).source_view_one is None

    disabled: DataLoaders = build_data_loaders(make_config(use_augmented_views=False))
    assert next(iter(disabled.train)).source_view_one is None


def test_class_counts_match_train_split() -> None:
    data_loaders: DataLoaders = build_data_loaders(make_config())

    assert data_loaders.source_train_class_counts.shape == (3,)
    assert int(data_loaders.source_train_class_counts.sum()) == data_loaders.source_train_size


def test_rotation_is_exactly_ninety_degrees() -> None:
    augmentation: TwoViewAugmentation = TwoViewAugmentation(make_config())
    images: torch.Tensor = torch.rand(4, 3, 16, 16)
    rotated_view, _ = augmentation(images)

    assert torch.equal(rotated_view, torch.rot90(images, k=1, dims=(-2, -1)))
    assert torch.equal(torch.rot90(rotated_view, k=3, dims=(-2, -1)), images)


def test_non_right_angle_rotation_rejected() -> None:
    with pytest.raises(ValueError):
        TwoViewAugmentation(make_config(augmentation_rotation_degrees=45))


def test_zoom_and_crop_shape_range_and_content() -> None:
    augmentation: TwoViewAugmentation = TwoViewAugmentation(make_config(crop_size=32, augmentation_zoom_size=36, native_image_size=32))
    images: torch.Tensor = torch.rand(2, 3, 32, 32)
    _, zoomed_view = augmentation(images)

    assert zoomed_view.shape == images.shape
    assert zoomed_view.min() >= 0.0 and zoomed_view.max() <= 1.0
    assert not torch.equal(zoomed_view, images)

    constant_images: torch.Tensor = torch.full((1, 3, 32, 32), 0.5)
    assert torch.allclose(augmentation(constant_images)[1], constant_images, atol=1e-6)


def test_zoom_smaller_than_crop_rejected() -> None:
    with pytest.raises(ValueError):
        TwoViewAugmentation(make_config(crop_size=16, augmentation_zoom_size=8))


def test_synthetic_generator_deterministic_and_class_conditional() -> None:
    config: Config = make_config()
    first = SyntheticGenerator(config).generate(0)
    second = SyntheticGenerator(config).generate(0)
    other_seed = SyntheticGenerator(make_config(random_seed=1)).generate(0)

    assert torch.equal(first.images, second.images) and torch.equal(first.labels, second.labels)
    assert not torch.equal(first.images, other_seed.images)
    assert first.images.shape == (100, 3, 16, 16)

    class_means: torch.Tensor = torch.stack([first.images[first.labels == index].mean(dim=0) for index in range(3)])
    assert (class_means[0] - class_means[1]).abs().mean() > 0.05


def test_synthetic_domain_shift() -> None:
    generator: SyntheticGenerator = SyntheticGenerator(make_config())

    assert generator.generate(1).images.mean() < generator.generate(0).images.mean() - 0.05


def write_lsst_files(directory: Path, sample_count: int = 30, parts: tuple[str, ...] = ("train", "test")) -> None:
    random_generator: np.random.Generator = np.random.default_rng(0)
    for part_name in parts:
        np.save(directory / f"labels_{part_name}.npy", np.arange(sample_count) % 3)
        for tag in ("Y1", "Y10"):
            np.save(
                directory / f"images_{tag}_{part_name}.npy",
                random_generator.uniform(0, 500, size=(sample_count, 3, 10, 10)).astype(np.float32),
            )


def lsst_config(directory: Path) -> Config:
    return make_config(
        dataset_name="lsst", data_directory=str(directory), native_image_size=10, crop_size=16, augmentation_zoom_size=20
    )


def test_lsst_loader_end_to_end(tmp_path: Path) -> None:
    write_lsst_files(tmp_path)
    data_loaders: DataLoaders = build_data_loaders(lsst_config(tmp_path))
    batch: DomainAdaptationBatch = next(iter(data_loaders.train))

    assert data_loaders.source_train_size == 36
    assert batch.source_images.shape == (8, 3, 16, 16)
    assert 0.0 <= batch.source_images.min() and batch.source_images.max() <= 1.0


def test_lsst_normalisation_range_and_uint8(tmp_path: Path) -> None:
    write_lsst_files(tmp_path)
    domain = LSSTLoader(lsst_config(tmp_path)).load_domain("LSST Y1")
    assert domain.images.min() == 0.0 and domain.images.max() == 1.0
    assert domain.images.shape == (60, 3, 10, 10)

    np.save(tmp_path / "images_Y1_train.npy", np.full((30, 3, 10, 10), 255, dtype=np.uint8))
    np.save(tmp_path / "images_Y1_test.npy", np.zeros((30, 3, 10, 10), dtype=np.uint8))
    uint8_domain = LSSTLoader(lsst_config(tmp_path)).load_domain("LSST Y1")
    assert set(uint8_domain.images.unique().tolist()) == {0.0, 1.0}


def test_lsst_finds_files_in_subdirectory_and_channels_last(tmp_path: Path) -> None:
    subdirectory: Path = tmp_path / "zenodo"
    subdirectory.mkdir()
    write_lsst_files(subdirectory, parts=("train",))
    np.save(subdirectory / "images_Y1_train.npy", np.random.rand(30, 10, 10, 3).astype(np.float32))

    domain = LSSTLoader(lsst_config(tmp_path)).load_domain("LSST Y1")
    assert domain.images.shape == (30, 3, 10, 10)


def test_lsst_missing_files_raise_clear_error(tmp_path: Path) -> None:
    write_lsst_files(tmp_path, parts=("train",))
    (tmp_path / "images_Y10_train.npy").unlink()

    with pytest.raises(FileNotFoundError, match="images_Y10_train.npy"):
        LSSTLoader(lsst_config(tmp_path)).load_domain("LSST Y10")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LSSTLoader(lsst_config(tmp_path / "nowhere")).load_domain("LSST Y1")


def test_lsst_label_mismatch_raises(tmp_path: Path) -> None:
    write_lsst_files(tmp_path, parts=("train",))
    np.save(tmp_path / "labels_train.npy", np.zeros(5, dtype=np.int64))

    with pytest.raises(ValueError):
        LSSTLoader(lsst_config(tmp_path)).load_domain("LSST Y1")


def test_unknown_dataset_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_data_loaders(make_config(dataset_name="gz2_sdss_decals"))


def test_smoke_config_builds() -> None:
    config: Config = Config("configs/smoke.json")
    batch: DomainAdaptationBatch = next(iter(build_data_loaders(config).train))

    assert batch.source_images.shape == (8, 3, 32, 32)
    assert batch.target_view_two is not None
