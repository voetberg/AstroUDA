import logging

from astrouda.config import Config


def get_logger(name: str, config: Config) -> logging.Logger:
    "Package logger. All modules log through this so one config setting controls verbosity."
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(config.log_level)

    return logger
