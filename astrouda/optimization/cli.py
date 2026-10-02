import argparse
import logging
from typing import Any, Optional

from astrouda.optimization.runner import OptimizationRunner
from astrouda.optimization.search_space import SearchSpace
from astrouda.optimization.trial_log import TrialRecord

logger: logging.Logger = logging.getLogger(__name__)


def register_optimize_parser(subparsers: Any) -> None:
    optimize_parser: argparse.ArgumentParser = subparsers.add_parser("optimize", help="Random hyperparameter search")
    optimize_parser.add_argument("--config", required=True, help="Path to the base Config JSON")
    optimize_parser.add_argument("--search-space", required=True, help="Path to the search space JSON")
    optimize_parser.add_argument("--trials", required=True, type=int, help="Total number of trials in the sweep")
    optimize_parser.add_argument("--trial-index", type=int, default=None, help="Run only this trial (Slurm array task id)")
    optimize_parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Config override, VALUE parsed as JSON"
    )
    optimize_parser.add_argument("--objective-metric", default="target_validation_accuracy", help="History metric to maximise")
    optimize_parser.add_argument("--search-seed", type=int, default=None, help="Sampler seed, defaults to the config random_seed")


def optimize_command(arguments: argparse.Namespace) -> int:
    "Exit code 0 when every trial run by this invocation succeeded, 1 otherwise."
    from astrouda.cli import parse_override

    config_overrides: dict[str, Any] = dict(parse_override(override_text) for override_text in arguments.overrides)
    runner: OptimizationRunner = OptimizationRunner(
        config_path=arguments.config,
        search_space=SearchSpace.from_json(arguments.search_space),
        number_of_trials=arguments.trials,
        config_overrides=config_overrides,
        objective_metric=arguments.objective_metric,
        search_seed=arguments.search_seed,
    )
    records: list[TrialRecord] = runner.run(arguments.trial_index)

    best_record: Optional[TrialRecord] = runner.trial_log.best_record()
    if best_record is not None:
        logger.info(f"Best trial {best_record.trial_index}: {best_record.objective:.4f} with {best_record.parameters}")
    failed_count: int = sum(record.status == "failed" for record in records)
    logger.debug(f"Ran {len(records)} trials, {failed_count} failed")

    return 1 if failed_count else 0
