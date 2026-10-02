from torch import Tensor, nn

from astrouda.config import Config


class SharedBatchNorm2d(nn.BatchNorm2d):
    """
    Ordinary BatchNorm2d with the domain-aware call signature.
    The domain index is accepted and ignored, so source and target share statistics and affine parameters.
    """

    def forward(self, images: Tensor, domain_index: int) -> Tensor:  # type: ignore[override]
        return super().forward(images)


class DomainSpecificBatchNorm2d(nn.Module):
    """
    Domain-specific batch normalisation (design choice, the paper gives no detail).
    Holds one full BatchNorm2d per domain: each has its own running mean, running variance, weight and bias.
    The domain is chosen by an explicit integer argument at forward time; nothing is stored between calls.
    """

    def __init__(self, number_of_features: int, number_of_domains: int) -> None:
        super().__init__()
        self.number_of_domains: int = number_of_domains
        self.domain_batch_norms: nn.ModuleList = nn.ModuleList(
            [nn.BatchNorm2d(number_of_features) for _ in range(number_of_domains)]
        )

    def forward(self, images: Tensor, domain_index: int) -> Tensor:
        if not 0 <= domain_index < self.number_of_domains:
            raise ValueError(f"domain_index must be in [0, {self.number_of_domains}), got {domain_index}")

        return self.domain_batch_norms[domain_index](images)


def build_normalization_layer(number_of_features: int, config: Config) -> nn.Module:
    "Factory used by every norm site of the backbone, so BN type is decided in exactly one place."
    if config.use_domain_specific_batch_norm:
        return DomainSpecificBatchNorm2d(number_of_features, config.number_of_domains)

    return SharedBatchNorm2d(number_of_features)
