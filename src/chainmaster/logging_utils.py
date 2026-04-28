from __future__ import annotations

import logging
import os
import sys


LOGGER_NAME = "chainmaster"
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def resolve_log_level(default: str = DEFAULT_LOG_LEVEL) -> str:
    return (
        os.getenv("LANGCHAINDEMO_LOG_LEVEL")
        or os.getenv("LOG_LEVEL")
        or default
    ).upper()


def configure_logging(level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        if level is not None:
            logger.setLevel(level.upper())
        return logger

    logger.setLevel((level or resolve_log_level()).upper())
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
