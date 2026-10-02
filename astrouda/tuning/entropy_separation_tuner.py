"""
Hyperparameter tuner for the entropy separation parameters (Algorithm 1, arXiv:2302.02005).

The tuner is a finite-state machine fed one training loss per epoch. It has no torch dependency.

State: phase, step_index j, non_improving_epochs counter, minimum_loss (L_min), rho, m.
An epoch is "improving" when loss < L_min (strict, so an equal loss is non-improving).

Improving epoch:    L_min = loss, counter = 0, phase and j unchanged.
Non-improving epoch: counter += 1, then the transition table below is evaluated.

    phase              fires when                                  action                       next phase
    -----------------  ------------------------------------------  ---------------------------  -----------------
    ADJUST_BOUNDARY    counter > entropy_boundary_patience_epochs  rho += step[j], counter = 0  ADJUST_MARGIN
    ADJUST_MARGIN      counter > confidence_margin_patience_epochs m += step[j],   counter = 0  DOUBLE_MARGIN
    DOUBLE_MARGIN      counter > confidence_margin_patience_epochs m *= 2,         counter = 0  ADJUST_BOUNDARY
                                                                   and j = (j + 1) mod len(step)

Design choices where the paper is silent:
    - L_min is the minimum training total loss seen so far (config.tuner_reference_loss).
    - The step index wraps around after the last step.
    - The doubling phase reuses confidence_margin_patience_epochs.
    - An improvement resets only the counter, the cycle position is kept.
    - rho is clamped to >= entropy_boundary_floor and m to >= confidence_margin_floor after every change.
    - Initial rho = log(K) / 2 and initial m = 0.4 (K <= 5) or 1.3 (K > 5) unless the config overrides them.
"""

import logging
import math
from enum import Enum
from typing import Any, Optional

from astrouda.config import Config
from astrouda.logging_utils import get_logger


class TunerPhase(str, Enum):
    ADJUST_BOUNDARY = "adjust_boundary"
    ADJUST_MARGIN = "adjust_margin"
    DOUBLE_MARGIN = "double_margin"


class EntropySeparationTuner:
    """Tracks rho (entropy_boundary) and m (confidence_margin) across epochs. Call update once per epoch."""

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.logger: logging.Logger = get_logger(__name__, config)

        self._entropy_boundary: float = self._clamp_boundary(self._initial_boundary())
        self._confidence_margin: float = self._clamp_margin(self._initial_margin())
        self.phase: TunerPhase = TunerPhase.ADJUST_BOUNDARY
        self.step_index: int = 0
        self.non_improving_epochs: int = 0
        self.minimum_loss: float = math.inf
        self.epoch_count: int = 0
        self.history: list[tuple[int, float, float, float, str]] = []

        self.logger.debug(
            f"Tuner initialised: rho={self._entropy_boundary}, m={self._confidence_margin}, "
            f"steps={self.config.entropy_step_values}"
        )

    @property
    def entropy_boundary(self) -> float:
        return float(self._entropy_boundary)

    @property
    def confidence_margin(self) -> float:
        return float(self._confidence_margin)

    def _initial_boundary(self) -> float:
        if self.config.entropy_boundary_initial is not None:
            return float(self.config.entropy_boundary_initial)

        return math.log(self.config.number_of_classes) / 2.0

    def _initial_margin(self) -> float:
        if self.config.confidence_margin_initial is not None:
            return float(self.config.confidence_margin_initial)

        return 0.4 if self.config.number_of_classes <= 5 else 1.3

    def _clamp_boundary(self, value: float) -> float:
        return max(float(value), float(self.config.entropy_boundary_floor))

    def _clamp_margin(self, value: float) -> float:
        return max(float(value), float(self.config.confidence_margin_floor))

    def _current_step(self) -> float:
        return float(self.config.entropy_step_values[self.step_index])

    def _patience_for_phase(self) -> int:
        if self.phase == TunerPhase.ADJUST_BOUNDARY:
            return self.config.entropy_boundary_patience_epochs

        return self.config.confidence_margin_patience_epochs

    def update(self, epoch_loss: float) -> None:
        self.epoch_count += 1
        epoch_loss = float(epoch_loss)

        if epoch_loss < self.minimum_loss:
            self.minimum_loss = epoch_loss
            self.non_improving_epochs = 0
            self.logger.debug(f"Epoch {self.epoch_count}: loss {epoch_loss} improved, counter reset")
        else:
            self.non_improving_epochs += 1
            self.logger.debug(
                f"Epoch {self.epoch_count}: loss {epoch_loss} >= L_min {self.minimum_loss}, "
                f"non-improving epochs {self.non_improving_epochs} in phase {self.phase.value}"
            )
            if self.non_improving_epochs > self._patience_for_phase():
                self._fire_transition()

        self.history.append(
            (self.epoch_count, epoch_loss, self.entropy_boundary, self.confidence_margin, self.phase.value)
        )

    def _fire_transition(self) -> None:
        if self.phase == TunerPhase.ADJUST_BOUNDARY:
            self._entropy_boundary = self._clamp_boundary(self._entropy_boundary + self._current_step())
            self.phase = TunerPhase.ADJUST_MARGIN
        elif self.phase == TunerPhase.ADJUST_MARGIN:
            self._confidence_margin = self._clamp_margin(self._confidence_margin + self._current_step())
            self.phase = TunerPhase.DOUBLE_MARGIN
        else:
            self._confidence_margin = self._clamp_margin(self._confidence_margin * 2.0)
            self.phase = TunerPhase.ADJUST_BOUNDARY
            self.step_index = (self.step_index + 1) % len(self.config.entropy_step_values)

        self.non_improving_epochs = 0
        self.logger.debug(
            f"Transition to {self.phase.value}: rho={self._entropy_boundary}, m={self._confidence_margin}, "
            f"step_index={self.step_index}"
        )

    def state_dict(self) -> dict[str, Any]:
        reference_loss: Optional[float] = None if math.isinf(self.minimum_loss) else self.minimum_loss

        return {
            "entropy_boundary": self._entropy_boundary,
            "confidence_margin": self._confidence_margin,
            "phase": self.phase.value,
            "step_index": self.step_index,
            "non_improving_epochs": self.non_improving_epochs,
            "minimum_loss": reference_loss,
            "epoch_count": self.epoch_count,
            "history": [list(record) for record in self.history],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._entropy_boundary = float(state["entropy_boundary"])
        self._confidence_margin = float(state["confidence_margin"])
        self.phase = TunerPhase(state["phase"])
        self.step_index = int(state["step_index"])
        self.non_improving_epochs = int(state["non_improving_epochs"])
        self.minimum_loss = math.inf if state["minimum_loss"] is None else float(state["minimum_loss"])
        self.epoch_count = int(state["epoch_count"])
        self.history = [
            (int(epoch), float(loss), float(rho), float(margin), str(phase))
            for epoch, loss, rho, margin, phase in state["history"]
        ]
        self.logger.debug(f"Tuner state loaded at epoch {self.epoch_count}, phase {self.phase.value}")
