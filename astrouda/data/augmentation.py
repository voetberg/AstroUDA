import torch
import torch.nn.functional as functional

from astrouda.config import Config
from astrouda.logging_utils import get_logger


class TwoViewAugmentation:
    """
    Two deterministic views of a batch of images (paper: a 90 degree rotation, and zooming to 300 px then cropping back to 256).

    view_one: rotation by Config.augmentation_rotation_degrees, which must be a multiple of 90, done with
        torch.rot90 so it is exact. Direction is counter-clockwise (torch.rot90 on the height and width axes).
    view_two: bilinear zoom to Config.augmentation_zoom_size, then a centre crop back to Config.crop_size.

    The paper does not say whether the rotation and the zoom are combined in one view or split in two.
    They are kept as one transform per view so that the two views differ as much as possible.
    Inputs are CPU tensors of shape (B, C, crop_size, crop_size) in [0, 1]; outputs have the same shape and range.
    """

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.logger = get_logger(__name__, config)

        if self.config.augmentation_rotation_degrees % 90 != 0:
            raise ValueError(
                f"augmentation_rotation_degrees must be a multiple of 90, got {self.config.augmentation_rotation_degrees}"
            )
        if self.config.augmentation_zoom_size < self.config.crop_size:
            raise ValueError(
                f"augmentation_zoom_size ({self.config.augmentation_zoom_size}) must be at least "
                f"crop_size ({self.config.crop_size})"
            )
        self.logger.debug(
            f"Augmentation: rotate {self.config.augmentation_rotation_degrees} degrees, "
            f"zoom to {self.config.augmentation_zoom_size}, crop to {self.config.crop_size}"
        )

    def rotate(self, images: torch.Tensor) -> torch.Tensor:
        quarter_turns: int = (self.config.augmentation_rotation_degrees // 90) % 4

        return torch.rot90(images, k=quarter_turns, dims=(-2, -1))

    def zoom_and_crop(self, images: torch.Tensor) -> torch.Tensor:
        zoomed_images: torch.Tensor = functional.interpolate(
            images,
            size=(self.config.augmentation_zoom_size, self.config.augmentation_zoom_size),
            mode="bilinear",
            align_corners=False,
        )
        crop_start: int = (self.config.augmentation_zoom_size - self.config.crop_size) // 2
        crop_end: int = crop_start + self.config.crop_size

        return zoomed_images[:, :, crop_start:crop_end, crop_start:crop_end].clamp(0.0, 1.0).contiguous()

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.logger.debug(f"Building two views for a batch of shape {tuple(images.shape)}")

        return self.rotate(images), self.zoom_and_crop(images)
