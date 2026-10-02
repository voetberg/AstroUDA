from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DomainData:
    """
    One whole domain held in memory.

    images: float32, shape (N, channels, native_image_size, native_image_size), values in [0, 1].
    labels: int64, shape (N,), values in [0, number_of_classes).
    """

    images: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.images.shape[0])


@dataclass
class DomainAdaptationBatch:
    """
    One step of data for the trainer or evaluator. Source and target are sampled independently:
    source_images[i] and target_images[i] have no relationship to each other.

    All image tensors are float32 on CPU, shape (B, channels, crop_size, crop_size), values in [0, 1].

    Training batches (batch_size samples per domain):
        source_images, source_labels   labelled source samples
        target_images                  unlabelled target samples
        target_labels                  always None, the target labels never reach training
        source_view_one / source_view_two / target_view_one / target_view_two
                                       the two augmented views of the images above (view_one is the 90 degree
                                       rotation, view_two the zoom then centre crop). None when
                                       Config.use_augmented_views is False.

    Validation and test batches (in order, no shuffling):
        target_labels is filled so target accuracy can be computed. The view tensors are None.
        When one domain has fewer samples than the other, its tensors have zero rows in the trailing batches.

    source_indices and target_indices are the positions of the samples inside their own split, for
    bookkeeping and tests. They are not meant to be used as model input.
    """

    source_images: torch.Tensor
    source_labels: torch.Tensor
    target_images: torch.Tensor
    target_labels: Optional[torch.Tensor]
    source_view_one: Optional[torch.Tensor]
    source_view_two: Optional[torch.Tensor]
    target_view_one: Optional[torch.Tensor]
    target_view_two: Optional[torch.Tensor]
    source_indices: torch.Tensor
    target_indices: torch.Tensor
