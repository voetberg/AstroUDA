import argparse
from typing import Any

from astrouda.experiment.aggregation import aggregate_command
from astrouda.experiment.runs import experiment_command


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a Config JSON")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Config override, VALUE parsed as JSON"
    )


def register_experiment_subcommands(subparsers: Any) -> None:
    experiment_parser: argparse.ArgumentParser = subparsers.add_parser(
        "experiment", help="Run adapted and source-only training over several seeds, then aggregate"
    )
    _add_shared_arguments(experiment_parser)
    experiment_parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Run only this index (seed_index * 2 + 0 adapted / 1 source-only) and skip aggregation",
    )
    experiment_parser.set_defaults(
        handler=lambda arguments: experiment_command(arguments.config, arguments.overrides, arguments.run_index)
    )

    aggregate_parser: argparse.ArgumentParser = subparsers.add_parser(
        "aggregate", help="Aggregate saved per-run reports into aggregate.json, comparison_table.md and plots"
    )
    _add_shared_arguments(aggregate_parser)
    aggregate_parser.set_defaults(handler=lambda arguments: aggregate_command(arguments.config, arguments.overrides))
