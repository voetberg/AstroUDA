import logging
import os
import random
from typing import Any

import numpy as np
import torch

logger: logging.Logger = logging.getLogger(__name__)


def capture_random_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_states(random_states: dict[str, Any]) -> None:
    random.setstate(random_states["python"])
    np.random.set_state(random_states["numpy"])
    torch.set_rng_state(random_states["torch"].cpu())
    if random_states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(random_states["cuda"])
    logger.debug("Random states restored")


def save_checkpoint(checkpoint: dict[str, Any], checkpoint_path: str) -> None:
    "Written to a temporary file first so an interrupted save never corrupts the previous checkpoint."
    temporary_path: str = f"{checkpoint_path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    logger.debug(f"Checkpoint written to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    logger.debug(f"Loading checkpoint {checkpoint_path}")

    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
