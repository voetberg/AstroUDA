from typing import Optional, Type, Union

from torch import Tensor, nn

from astrouda.config import Config
from astrouda.logging_utils import get_logger
from astrouda.models.normalization import build_normalization_layer

BLOCKS_PER_STAGE: dict[int, list[int]] = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}
STAGE_WIDTHS: list[int] = [64, 128, 256, 512]


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, input_channels: int, width: int, stride: int, config: Config) -> None:
        super().__init__()
        output_channels: int = width * self.expansion

        self.first_convolution: nn.Conv2d = nn.Conv2d(input_channels, width, 3, stride, 1, bias=False)
        self.first_norm: nn.Module = build_normalization_layer(width, config)
        self.second_convolution: nn.Conv2d = nn.Conv2d(width, output_channels, 3, 1, 1, bias=False)
        self.second_norm: nn.Module = build_normalization_layer(output_channels, config)
        self.activation: nn.ReLU = nn.ReLU(inplace=True)

        self.shortcut_convolution: Optional[nn.Conv2d] = None
        self.shortcut_norm: Optional[nn.Module] = None
        if stride != 1 or input_channels != output_channels:
            self.shortcut_convolution = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
            self.shortcut_norm = build_normalization_layer(output_channels, config)

    def forward(self, images: Tensor, domain_index: int) -> Tensor:
        residual: Tensor = images

        hidden: Tensor = self.activation(self.first_norm(self.first_convolution(images), domain_index))
        hidden = self.second_norm(self.second_convolution(hidden), domain_index)

        if self.shortcut_convolution is not None and self.shortcut_norm is not None:
            residual = self.shortcut_norm(self.shortcut_convolution(images), domain_index)

        return self.activation(hidden + residual)


class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(self, input_channels: int, width: int, stride: int, config: Config) -> None:
        super().__init__()
        output_channels: int = width * self.expansion

        self.reduce_convolution: nn.Conv2d = nn.Conv2d(input_channels, width, 1, bias=False)
        self.reduce_norm: nn.Module = build_normalization_layer(width, config)
        self.spatial_convolution: nn.Conv2d = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.spatial_norm: nn.Module = build_normalization_layer(width, config)
        self.expand_convolution: nn.Conv2d = nn.Conv2d(width, output_channels, 1, bias=False)
        self.expand_norm: nn.Module = build_normalization_layer(output_channels, config)
        self.activation: nn.ReLU = nn.ReLU(inplace=True)

        self.shortcut_convolution: Optional[nn.Conv2d] = None
        self.shortcut_norm: Optional[nn.Module] = None
        if stride != 1 or input_channels != output_channels:
            self.shortcut_convolution = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
            self.shortcut_norm = build_normalization_layer(output_channels, config)

    def forward(self, images: Tensor, domain_index: int) -> Tensor:
        residual: Tensor = images

        hidden: Tensor = self.activation(self.reduce_norm(self.reduce_convolution(images), domain_index))
        hidden = self.activation(self.spatial_norm(self.spatial_convolution(hidden), domain_index))
        hidden = self.expand_norm(self.expand_convolution(hidden), domain_index)

        if self.shortcut_convolution is not None and self.shortcut_norm is not None:
            residual = self.shortcut_norm(self.shortcut_convolution(images), domain_index)

        return self.activation(hidden + residual)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-18 or ResNet-50 up to and including global average pooling, randomly initialised (no pretrained weights exist in this code path).
    Architecture follows the standard ResNet v1.5 layout (stride on the 3x3 convolution of bottlenecks).
    `use_small_input_stem` replaces the 7x7 stride-2 stem and max-pool by one 3x3 stride-1 convolution (design choice for tiny images).
    Every normalisation layer takes the domain index explicitly.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.logger = get_logger(__name__, config)

        if config.backbone_depth not in BLOCKS_PER_STAGE:
            raise ValueError(f"backbone_depth must be one of {sorted(BLOCKS_PER_STAGE)}, got {config.backbone_depth}")

        block_class: Type[Union[BasicBlock, BottleneckBlock]] = (
            BasicBlock if config.backbone_depth == 18 else BottleneckBlock
        )

        if config.use_small_input_stem:
            self.stem_convolution: nn.Conv2d = nn.Conv2d(config.number_of_image_channels, 64, 3, 1, 1, bias=False)
            self.stem_pool: nn.Module = nn.Identity()
        else:
            self.stem_convolution = nn.Conv2d(config.number_of_image_channels, 64, 7, 2, 3, bias=False)
            self.stem_pool = nn.MaxPool2d(3, 2, 1)
        self.stem_norm: nn.Module = build_normalization_layer(64, config)
        self.activation: nn.ReLU = nn.ReLU(inplace=True)

        self.blocks: nn.ModuleList = nn.ModuleList()
        current_channels: int = 64
        for stage_index, (stage_width, block_count) in enumerate(
            zip(STAGE_WIDTHS, BLOCKS_PER_STAGE[config.backbone_depth])
        ):
            for block_index in range(block_count):
                stride: int = 2 if (stage_index > 0 and block_index == 0) else 1
                self.blocks.append(block_class(current_channels, stage_width, stride, config))
                current_channels = stage_width * block_class.expansion

        self.feature_dimension: int = current_channels
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)

        self._initialise_weights()
        self.logger.debug(
            f"Built ResNet{config.backbone_depth} extractor, feature_dimension={self.feature_dimension}, "
            f"small_stem={config.use_small_input_stem}, domain_specific_bn={config.use_domain_specific_batch_norm}"
        )

    def _initialise_weights(self) -> None:
        "Kaiming normal (fan_out, relu) for convolutions, the standard from-scratch ResNet recipe."
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, images: Tensor, domain_index: int) -> Tensor:
        hidden: Tensor = self.activation(self.stem_norm(self.stem_convolution(images), domain_index))
        hidden = self.stem_pool(hidden)

        for block in self.blocks:
            hidden = block(hidden, domain_index)

        return self.global_pool(hidden).flatten(start_dim=1)
