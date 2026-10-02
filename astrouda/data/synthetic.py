import math

import torch
import torch.nn.functional as functional

from astrouda.config import Config
from astrouda.data.batch import DomainData
from astrouda.logging_utils import get_logger


class SyntheticGenerator:
    """
    Class-conditional synthetic images for smoke tests and unit tests.

    Class c is an oriented sinusoidal grating (orientation pi * c / number_of_classes) with a Gaussian blob on top whose
    position moves with the class; each channel gets its own phase. Source images are crisp and bright with low noise.
    The target domain is darker, blurred and noisier, which is the domain shift.
    Output shape is (N, number_of_image_channels, native_image_size, native_image_size), values in [0, 1].
    Fully determined by Config.random_seed, so repeated calls give identical tensors.
    """

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.logger = get_logger(__name__, config)

    def _class_templates(self) -> torch.Tensor:
        coordinate_axis: torch.Tensor = torch.linspace(-1.0, 1.0, self.config.native_image_size)
        vertical_grid, horizontal_grid = torch.meshgrid(coordinate_axis, coordinate_axis, indexing="ij")

        templates: list[torch.Tensor] = []
        for class_index in range(self.config.number_of_classes):
            orientation: float = math.pi * class_index / self.config.number_of_classes
            blob_angle: float = 2.0 * math.pi * class_index / self.config.number_of_classes
            blob_vertical: float = 0.5 * math.sin(blob_angle)
            blob_horizontal: float = 0.5 * math.cos(blob_angle)

            projected_coordinate: torch.Tensor = horizontal_grid * math.cos(orientation) + vertical_grid * math.sin(orientation)
            blob: torch.Tensor = torch.exp(
                -((vertical_grid - blob_vertical) ** 2 + (horizontal_grid - blob_horizontal) ** 2) / 0.08
            )

            channel_templates: list[torch.Tensor] = []
            for channel_index in range(self.config.number_of_image_channels):
                channel_phase: float = 0.7 * channel_index
                grating: torch.Tensor = 0.5 + 0.5 * torch.sin(6.0 * math.pi * projected_coordinate + channel_phase)
                channel_templates.append(0.5 * grating + 0.5 * blob)
            templates.append(torch.stack(channel_templates))

        return torch.stack(templates)

    def _apply_target_shift(self, images: torch.Tensor, random_generator: torch.Generator) -> torch.Tensor:
        blurred_images: torch.Tensor = functional.avg_pool2d(images, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        darkened_images: torch.Tensor = 0.6 * blurred_images + 0.05
        extra_noise: torch.Tensor = 0.1 * torch.randn(images.shape, generator=random_generator)

        return darkened_images + extra_noise

    def generate(self, domain_identifier: int) -> DomainData:
        """domain_identifier 0 is the source domain, 1 the target domain."""
        random_generator: torch.Generator = torch.Generator().manual_seed(self.config.random_seed * 7919 + domain_identifier)
        sample_count: int = self.config.synthetic_samples_per_domain

        labels: torch.Tensor = torch.randperm(sample_count, generator=random_generator) % self.config.number_of_classes
        images: torch.Tensor = self._class_templates()[labels]
        images = images + 0.05 * torch.randn(images.shape, generator=random_generator)

        if domain_identifier == 1:
            images = self._apply_target_shift(images, random_generator)

        images = images.clamp(0.0, 1.0).float().contiguous()
        self.logger.debug(
            f"Synthetic domain {domain_identifier}: images {tuple(images.shape)}, "
            f"mean {images.mean().item():.3f}, class counts {torch.bincount(labels).tolist()}"
        )

        return DomainData(images=images, labels=labels.long())
