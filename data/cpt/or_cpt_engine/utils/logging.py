from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FILE_ONLY_LOGGERS = ("or_cpt_engine.model_call",)
_MODEL_FAILURE_LOGGER = "or_cpt_engine.model_failure"


def _reset_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(
    level: str = "INFO",
    *,
    log_file: str | None = None,
    model_failure_log_file: str | None = None,
) -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    root_logger = logging.getLogger()
    _reset_logger_handlers(root_logger)
    root_logger.setLevel(resolved_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    for logger_name in _FILE_ONLY_LOGGERS:
        file_only_logger = logging.getLogger(logger_name)
        _reset_logger_handlers(file_only_logger)
        file_only_logger.setLevel(resolved_level)
        file_only_logger.propagate = False
    model_failure_logger = logging.getLogger(_MODEL_FAILURE_LOGGER)
    _reset_logger_handlers(model_failure_logger)
    model_failure_logger.setLevel(resolved_level)
    model_failure_logger.propagate = False

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        for logger_name in _FILE_ONLY_LOGGERS:
            file_only_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_only_handler.setLevel(resolved_level)
            file_only_handler.setFormatter(formatter)
            logging.getLogger(logger_name).addHandler(file_only_handler)

    if model_failure_log_file:
        failure_path = Path(model_failure_log_file)
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_handler = logging.FileHandler(failure_path, encoding="utf-8")
        failure_handler.setLevel(resolved_level)
        failure_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger(_MODEL_FAILURE_LOGGER).addHandler(failure_handler)

    noisy_level = logging.DEBUG if resolved_level <= logging.DEBUG else logging.WARNING
    logging.getLogger("httpx").setLevel(noisy_level)
    logging.getLogger("httpcore").setLevel(noisy_level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_model_failure_case(**payload: Any) -> None:
    serialized = dict(payload)
    serialized.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
    logging.getLogger(_MODEL_FAILURE_LOGGER).info(
        json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
    )
