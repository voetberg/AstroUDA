import argparse
import json
import logging
import sys
from typing import Any, Optional

from astrouda.config import Config
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

    return parser


def train_command(config_path: str, override_texts: list[str]) -> Trainer:
    overrides: dict[str, Any] = dict(parse_override(override_text) for override_text in override_texts)
    config: Config = Config(config_path, **overrides)
    trainer: Trainer = Trainer(config)
    trainer.train()

    return trainer


def main(argument_list: Optional[list[str]] = None) -> int:
    "Returns the process exit code: 0 on success, 1 on any failure."
    arguments: argparse.Namespace = build_argument_parser().parse_args(argument_list)

    try:
        if arguments.command == "train":
            train_command(arguments.config, arguments.overrides)
    except Exception:
        logger.exception("Training failed")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
