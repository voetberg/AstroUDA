import logging
import math
import os
import time
import traceback
from typing import Any, Optional

from astrouda.config import Config
from astrouda.logging_utils import get_logger
from astrouda.optimization.sampler import sample_trial_parameters
from astrouda.optimization.search_space import SearchSpace
from astrouda.optimization.trial_log import TrialLog, TrialRecord
from astrouda.training import Trainer


class OptimizationRunner:
    """
    Random search over a SearchSpace. Output layout under <output_directory>/optimization/:
    trials.jsonl, best_config.json and one trial_NNNN directory per trial.
    """

    def __init__(
        self,
        config_path: str,
        search_space: SearchSpace,
        number_of_trials: int,
        config_overrides: Optional[dict[str, Any]] = None,
        objective_metric: str = "target_validation_accuracy",
        search_seed: Optional[int] = None,
    ) -> None:
        self.config_path: str = config_path
        self.search_space: SearchSpace = search_space
        self.number_of_trials: int = number_of_trials
        self.config_overrides: dict[str, Any] = dict(config_overrides or {})
        self.objective_metric: str = objective_metric

        clashing_keys: set[str] = set(self.config_overrides) & set(self.search_space.parameter_names)
        if clashing_keys:
            raise ValueError(f"Keys {sorted(clashing_keys)} are both searched and fixed by --set")

        self.base_config: Config = Config(self.config_path, **self.config_overrides)
        self.search_seed: int = self.base_config.random_seed if search_seed is None else search_seed
        self.logger: logging.Logger = get_logger(__name__, self.base_config)

        self.optimization_directory: str = os.path.join(self.base_config.output_directory, "optimization")
        os.makedirs(self.optimization_directory, exist_ok=True)
        self.trial_log: TrialLog = TrialLog(os.path.join(self.optimization_directory, "trials.jsonl"))
        self.best_config_path: str = os.path.join(self.optimization_directory, "best_config.json")

    def trial_directory(self, trial_index: int) -> str:
        return os.path.join(self.optimization_directory, f"trial_{trial_index:04d}")

    def build_trial_config(self, trial_index: int, sampled_parameters: dict[str, Any]) -> Config:
        trial_overrides: dict[str, Any] = {
            **self.config_overrides,
            **sampled_parameters,
            "output_directory": self.trial_directory(trial_index),
        }

        return Config(self.config_path, **trial_overrides)

    def run_trial(self, trial_index: int) -> TrialRecord:
        start_time: float = time.time()
        sampled_parameters: dict[str, Any] = sample_trial_parameters(self.search_space, self.search_seed, trial_index)
        self.logger.info(f"Trial {trial_index}/{self.number_of_trials} parameters: {sampled_parameters}")

        objective: Optional[float] = None
        epochs_run: int = 0
        error: Optional[str] = None
        try:
            trial_config: Config = self.build_trial_config(trial_index, sampled_parameters)
            history: list[dict[str, float]] = Trainer(trial_config).train()
            epochs_run = len(history)
            objective = self._objective_from_history(history)
        except Exception as exception:
            self.logger.debug(traceback.format_exc())
            error = f"{type(exception).__name__}: {exception}"
            objective = None
            self.logger.warning(f"Trial {trial_index} failed: {error}")

        record: TrialRecord = TrialRecord(
            trial_index=trial_index,
            parameters=sampled_parameters,
            objective=objective,
            objective_metric=self.objective_metric,
            epochs_run=epochs_run,
            status="ok" if error is None else "failed",
            error=error,
            wall_time_seconds=time.time() - start_time,
            output_directory=self.trial_directory(trial_index),
        )
        self.trial_log.append(record)
        self.write_best_config()
        self.logger.info(f"Trial {trial_index} {record.status}, objective {record.objective}, {record.wall_time_seconds:.1f}s")

        return record

    def _objective_from_history(self, history: list[dict[str, float]]) -> float:
        if not history or self.objective_metric not in history[0]:
            raise ValueError(f"Objective metric '{self.objective_metric}' missing from training history")

        best_value: float = max(epoch_record[self.objective_metric] for epoch_record in history)
        if math.isnan(best_value):
            raise ValueError(f"Objective metric '{self.objective_metric}' is NaN")

        return float(best_value)

    def run(self, trial_index: Optional[int] = None) -> list[TrialRecord]:
        "One trial when trial_index is given (reruns it unless it completed), otherwise all pending trials in order."
        if trial_index is not None and not 0 <= trial_index < self.number_of_trials:
            raise ValueError(f"--trial-index {trial_index} is outside [0, {self.number_of_trials})")

        requested_indices: list[int] = [trial_index] if trial_index is not None else list(range(self.number_of_trials))
        completed_indices: set[int] = self.trial_log.completed_indices()
        pending_indices: list[int] = [index for index in requested_indices if index not in completed_indices]
        self.logger.debug(f"Requested {requested_indices}, already completed {sorted(completed_indices)}, pending {pending_indices}")

        records: list[TrialRecord] = [self.run_trial(index) for index in pending_indices]
        self.write_best_config()

        return records

    def write_best_config(self) -> Optional[str]:
        best_record: Optional[TrialRecord] = self.trial_log.best_record()
        if best_record is None:
            self.logger.debug("No completed trial yet, best_config.json not written")

            return None

        best_config: Config = self.build_trial_config(best_record.trial_index, best_record.parameters)
        best_config.to_json(self.best_config_path)
        self.logger.debug(
            f"Best trial {best_record.trial_index} objective {best_record.objective} written to {self.best_config_path}"
        )

        return self.best_config_path
