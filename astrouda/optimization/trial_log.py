import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class TrialRecord:
    trial_index: int
    parameters: dict[str, Any]
    objective: Optional[float]
    objective_metric: str
    epochs_run: int
    status: str  # "ok" or "failed"
    error: Optional[str]
    wall_time_seconds: float
    output_directory: str


class TrialLog:
    "Append-only JSON lines file. The last record of a trial index wins, so a failed trial can be retried."

    def __init__(self, log_path: str) -> None:
        self.log_path: str = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def append(self, record: TrialRecord) -> None:
        with open(self.log_path, "a") as log_file:
            log_file.write(json.dumps(asdict(record)) + "\n")
        logger.debug(f"Appended trial {record.trial_index} ({record.status}) to {self.log_path}")

    def latest_records(self) -> dict[int, TrialRecord]:
        latest_by_index: dict[int, TrialRecord] = {}
        if not os.path.exists(self.log_path):
            return latest_by_index

        with open(self.log_path, "r") as log_file:
            for line in log_file:
                if line.strip():
                    record: TrialRecord = TrialRecord(**json.loads(line))
                    latest_by_index[record.trial_index] = record
        logger.debug(f"Read {len(latest_by_index)} trials from {self.log_path}")

        return latest_by_index

    def completed_indices(self) -> set[int]:
        return {index for index, record in self.latest_records().items() if record.status == "ok"}

    def best_record(self) -> Optional[TrialRecord]:
        "Highest objective among completed trials, lowest index on ties."
        completed_records: list[TrialRecord] = [record for record in self.latest_records().values() if record.status == "ok"]
        if not completed_records:
            return None

        return min(completed_records, key=lambda record: (-record.objective, record.trial_index))
