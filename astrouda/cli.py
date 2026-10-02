import argparse
import json
import logging
import sys
from typing import Any, Optional

from astrouda.config import Config
from astrouda.exit_codes import EX_TEMPFAIL
from astrouda.optimization.cli import optimize_command, register_optimize_parser
from astrouda.training import Trainer

logger: logging.Logger = logging.getLogger(__name__)


def parse_override(override_text: str) -> tuple[str, Any]:
    "'key=value' where value is JSON. A value that is not valid JSON (for example a bare path) is kept as a string."
    if "=" not in override_text:
        raise ValueError(f"--set expects key=value, got '{override_text}'")

    key, value_text = override_text.split("=", 1)
    try:
        value: Any = json.loads(value_text)
    except json.JSONDecodeError:
        logger.debug(f"Override '{key}' is not valid JSON, using the raw string {value_text!r}")
        value = value_text

    return key, value


def build_argument_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="astrouda")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser: argparse.ArgumentParser = subparsers.add_parser("train", help="Train from a JSON config")
    train_parser.add_argument("--config", required=True, help="Path to a Config JSON")
    train_parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Config override, VALUE parsed as JSON"
    )

    from astrouda.experiment.cli_registration import register_experiment_subcommands

    register_experiment_subcommands(subparsers)
    register_optimize_parser(subparsers)

    return parser


def train_command(config_path: str, override_texts: list[str]) -> Trainer:
    "Returns the Trainer; check trainer.stopped_for_time_limit to see whether training ended early."
    overrides: dict[str, Any] = dict(parse_override(override_text) for override_text in override_texts)
    config: Config = Config(config_path, **overrides)
    trainer: Trainer = Trainer(config)
    trainer.train()

    return trainer


def main(argument_list: Optional[list[str]] = None) -> int:
    "Returns the process exit code: 0 on success, 1 on any failure, EX_TEMPFAIL (75) when stopped for the time limit."
    arguments: argparse.Namespace = build_argument_parser().parse_args(argument_list)

    try:
        if arguments.command == "train":
            trainer: Trainer = train_command(arguments.config, arguments.overrides)
            if trainer.stopped_for_time_limit:
                return EX_TEMPFAIL
        elif arguments.command == "experiment":
            experiment_exit_code: int = arguments.handler(arguments)
            if experiment_exit_code != 0:
                return experiment_exit_code
        elif arguments.command == "aggregate":
            arguments.handler(arguments)
        elif arguments.command == "optimize":
            return optimize_command(arguments)
    except Exception:
        logger.exception("Training failed")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
