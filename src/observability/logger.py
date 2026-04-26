from __future__ import annotations

import json
import logging
import sys
import time
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
    _logger.info(json.dumps({**data, "ts": time.time()}, default=str))


def log_error(event: str, error: Exception, extra: dict[str, Any] | None = None) -> None:
    _logger.error(json.dumps({
        "event": event,
        "error": type(error).__name__,
        "detail": str(error),
        "ts": time.time(),
        **(extra or {}),
    }, default=str))
