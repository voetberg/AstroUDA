import logging

import torch

logger: logging.Logger = logging.getLogger(__name__)


def resolve_device(device_name: str) -> torch.device:
    "'auto' picks cuda, then mps, then cpu. Any other name is passed to torch and must be available."
    if device_name == "auto":
        if torch.cuda.is_available():
            resolved_name: str = "cuda"
        elif torch.backends.mps.is_available():
            resolved_name = "mps"
        else:
            resolved_name = "cpu"
    else:
        resolved_name = device_name

    resolved_device: torch.device = torch.device(resolved_name)
    logger.debug(f"Device '{device_name}' resolved to {resolved_device}")

    return resolved_device
