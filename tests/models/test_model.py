import pytest
import torch
from torch import Tensor, nn

from astrouda.config import Config
from astrouda.models import (
    DomainAdaptationNetwork,
    DomainSpecificBatchNorm2d,
    ModelOutput,
    SharedBatchNorm2d,
    build_model,
)

BATCH_SIZE: int = 4
IMAGE_SIZE: int = 32


def make_config(**overrides: object) -> Config:
    default_overrides: dict[str, object] = {"backbone_depth": 18, "use_small_input_stem": True, "log_level": "WARNING"}
    default_overrides.update(overrides)

    return Config(**default_overrides)


def make_images(config: Config) -> Tensor:
    return torch.randn(BATCH_SIZE, config.number_of_image_channels, IMAGE_SIZE, IMAGE_SIZE)


def domain_batch_norm_layers(model: nn.Module) -> list[DomainSpecificBatchNorm2d]:
    return [module for module in model.modules() if isinstance(module, DomainSpecificBatchNorm2d)]


def snapshot_statistics(layers: list[DomainSpecificBatchNorm2d], domain_index: int) -> list[tuple[Tensor, Tensor, Tensor]]:
    return [
        (
            layer.domain_batch_norms[domain_index].running_mean.clone(),
            layer.domain_batch_norms[domain_index].running_var.clone(),
            layer.domain_batch_norms[domain_index].num_batches_tracked.clone(),
        )
        for layer in layers
    ]


@pytest.mark.parametrize("backbone_depth, expected_feature_dimension", [(18, 512), (50, 2048)])
@pytest.mark.parametrize("use_small_input_stem", [True, False])
def test_output_shapes(backbone_depth: int, expected_feature_dimension: int, use_small_input_stem: bool) -> None:
    config: Config = make_config(backbone_depth=backbone_depth, use_small_input_stem=use_small_input_stem)
    model: DomainAdaptationNetwork = build_model(config)

    output: ModelOutput = model(make_images(config), domain_index=1)

    assert model.feature_dimension == expected_feature_dimension
    assert output.features.shape == (BATCH_SIZE, expected_feature_dimension)
    assert output.logits.shape == (BATCH_SIZE, config.number_of_classes)


def test_classify_features_matches_logits() -> None:
    config: Config = make_config()
    model: DomainAdaptationNetwork = build_model(config).eval()

    output: ModelOutput = model(make_images(config), domain_index=0)

    assert torch.allclose(model.classify_features(output.features), output.logits)


def test_stem_variants() -> None:
    small_model: DomainAdaptationNetwork = build_model(make_config(use_small_input_stem=True))
    standard_model: DomainAdaptationNetwork = build_model(make_config(use_small_input_stem=False))

    assert small_model.feature_extractor.stem_convolution.kernel_size == (3, 3)
    assert small_model.feature_extractor.stem_convolution.stride == (1, 1)
    assert isinstance(small_model.feature_extractor.stem_pool, nn.Identity)
    assert standard_model.feature_extractor.stem_convolution.kernel_size == (7, 7)
    assert standard_model.feature_extractor.stem_convolution.stride == (2, 2)
    assert isinstance(standard_model.feature_extractor.stem_pool, nn.MaxPool2d)


def test_unsupported_depth_raises() -> None:
    with pytest.raises(ValueError):
        build_model(make_config(backbone_depth=34))


def test_no_pretrained_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid_download(*arguments: object, **keyword_arguments: object) -> None:
        raise AssertionError("weights must never be downloaded")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbid_download)
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbid_download)

    first_model: DomainAdaptationNetwork = build_model(make_config())
    second_model: DomainAdaptationNetwork = build_model(make_config())

    first_weight: Tensor = first_model.feature_extractor.stem_convolution.weight
    second_weight: Tensor = second_model.feature_extractor.stem_convolution.weight
    assert not torch.equal(first_weight, second_weight)


def test_domain_specific_statistics_update_only_matching_domain() -> None:
    config: Config = make_config()
    model: DomainAdaptationNetwork = build_model(config).train()
    layers: list[DomainSpecificBatchNorm2d] = domain_batch_norm_layers(model)

    source_before = snapshot_statistics(layers, domain_index=0)
    target_before = snapshot_statistics(layers, domain_index=1)

    model(make_images(config), domain_index=0)
    source_after_source_pass = snapshot_statistics(layers, domain_index=0)
    target_after_source_pass = snapshot_statistics(layers, domain_index=1)

    assert len(layers) > 0
    for before, after in zip(target_before, target_after_source_pass):
        for before_tensor, after_tensor in zip(before, after):
            assert torch.equal(before_tensor, after_tensor)
    assert any(not torch.equal(before[0], after[0]) for before, after in zip(source_before, source_after_source_pass))

    model(make_images(config), domain_index=1)
    source_after_target_pass = snapshot_statistics(layers, domain_index=0)
    target_after_target_pass = snapshot_statistics(layers, domain_index=1)

    for before, after in zip(source_after_source_pass, source_after_target_pass):
        for before_tensor, after_tensor in zip(before, after):
            assert torch.equal(before_tensor, after_tensor)
    assert any(not torch.equal(before[0], after[0]) for before, after in zip(target_after_source_pass, target_after_target_pass))


def test_no_plain_batch_norm_outside_domain_modules() -> None:
    model: DomainAdaptationNetwork = build_model(make_config())
    wrapped_batch_norm_ids: set[int] = {
        id(inner_norm) for layer in domain_batch_norm_layers(model) for inner_norm in layer.domain_batch_norms
    }

    unwrapped: list[nn.BatchNorm2d] = [
        module for module in model.modules() if isinstance(module, nn.BatchNorm2d) and id(module) not in wrapped_batch_norm_ids
    ]
    assert unwrapped == []


def test_convolution_weights_shared_and_affine_separate() -> None:
    model: DomainAdaptationNetwork = build_model(make_config())

    convolution_parameter_ids: list[int] = [
        id(parameter) for module in model.modules() if isinstance(module, nn.Conv2d) for parameter in module.parameters()
    ]
    assert len(convolution_parameter_ids) == len(set(convolution_parameter_ids))

    for layer in domain_batch_norm_layers(model):
        source_norm, target_norm = layer.domain_batch_norms[0], layer.domain_batch_norms[1]
        assert source_norm.weight is not target_norm.weight
        assert source_norm.bias is not target_norm.bias
        assert source_norm.running_mean is not target_norm.running_mean

    parameter_count_domain_specific: int = sum(parameter.numel() for parameter in model.parameters())
    parameter_count_shared: int = sum(
        parameter.numel() for parameter in build_model(make_config(use_domain_specific_batch_norm=False)).parameters()
    )
    assert parameter_count_domain_specific > parameter_count_shared


def test_affine_parameters_receive_gradient_only_for_used_domain() -> None:
    config: Config = make_config()
    model: DomainAdaptationNetwork = build_model(config).train()

    model(make_images(config), domain_index=0).logits.sum().backward()

    stem_layer: DomainSpecificBatchNorm2d = model.feature_extractor.stem_norm
    assert stem_layer.domain_batch_norms[0].weight.grad is not None
    assert stem_layer.domain_batch_norms[1].weight.grad is None


def test_shared_batch_norm_ignores_domain_index() -> None:
    config: Config = make_config(use_domain_specific_batch_norm=False)
    model: DomainAdaptationNetwork = build_model(config).eval()
    images: Tensor = make_images(config)

    assert not domain_batch_norm_layers(model)
    assert any(isinstance(module, SharedBatchNorm2d) for module in model.modules())
    with torch.no_grad():
        assert torch.equal(model(images, domain_index=0).logits, model(images, domain_index=1).logits)

    model.train()
    model(images, domain_index=1)
    stem_norm: SharedBatchNorm2d = model.feature_extractor.stem_norm
    assert stem_norm.num_batches_tracked.item() == 1


@pytest.mark.parametrize("invalid_domain_index", [-1, 2, 5])
def test_invalid_domain_index_raises(invalid_domain_index: int) -> None:
    config: Config = make_config()
    model: DomainAdaptationNetwork = build_model(config)

    with pytest.raises(ValueError):
        model(make_images(config), domain_index=invalid_domain_index)


def test_gradients_reach_extractor_and_head() -> None:
    config: Config = make_config()
    model: DomainAdaptationNetwork = build_model(config).train()

    output: ModelOutput = model(make_images(config), domain_index=0)
    (output.logits.sum() + output.features.sum()).backward()

    assert model.classifier_head.weight.grad is not None
    assert model.feature_extractor.stem_convolution.weight.grad is not None
    assert model.feature_extractor.stem_convolution.weight.grad.abs().sum() > 0


def test_eval_mode_is_deterministic() -> None:
    config: Config = make_config()
    model: DomainAdaptationNetwork = build_model(config).eval()
    images: Tensor = make_images(config)

    with torch.no_grad():
        first_output: ModelOutput = model(images, domain_index=1)
        second_output: ModelOutput = model(images, domain_index=1)

    assert torch.equal(first_output.logits, second_output.logits)
    assert torch.equal(first_output.features, second_output.features)
