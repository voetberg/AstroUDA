from dataclasses import dataclass

from torch import Tensor, nn

from astrouda.config import Config
from astrouda.logging_utils import get_logger
from astrouda.models.resnet import ResNetFeatureExtractor


@dataclass
class ModelOutput:
    features: Tensor  # (batch, feature_dimension), after global pooling
    logits: Tensor  # (batch, number_of_classes)


class DomainAdaptationNetwork(nn.Module):
    "One feature extractor and one linear classifier head, shared by every loss."

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.logger = get_logger(__name__, config)

        self.feature_extractor: ResNetFeatureExtractor = ResNetFeatureExtractor(config)
        self.feature_dimension: int = self.feature_extractor.feature_dimension
        self.classifier_head: nn.Linear = nn.Linear(self.feature_dimension, config.number_of_classes)

        self.logger.debug(f"Network ready: {self.feature_dimension} features -> {config.number_of_classes} classes")

    def classify_features(self, features: Tensor) -> Tensor:
        return self.classifier_head(features)

    def forward(self, images: Tensor, domain_index: int) -> ModelOutput:
        features: Tensor = self.feature_extractor(images, domain_index)
        logits: Tensor = self.classify_features(features)
        self.logger.debug(f"Forward domain={domain_index} images={tuple(images.shape)} features={tuple(features.shape)}")

        return ModelOutput(features=features, logits=logits)


def build_model(config: Config) -> DomainAdaptationNetwork:
    "Randomly initialised network; no pretrained weights are ever loaded."
    return DomainAdaptationNetwork(config)
