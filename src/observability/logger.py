from __future__ import annotations

import json
import logging
import sys
from typing import Any


def _setup() -> logging.Logger:
    logger = logging.getLogger("llm-shield")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


_logger = _setup()


def log_request(data: dict[str, Any]) -> None:
    _logger.info(json.dumps(data, default=str))
