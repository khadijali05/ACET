"""Safe local error logging for the Smoke-Detection application."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "smoke_detection"
_CREDENTIAL_URL = re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^\s/@]+@", re.IGNORECASE)
_SENSITIVE_VALUE = re.compile(
    r"\b(password|passwd|token|access_token|api_key|auth)=([^\s&]+)", re.IGNORECASE
)


class _SensitiveDataFilter(logging.Filter):
    """Remove common camera credentials from log messages before writing them."""

    @staticmethod
    def _redact(value: str) -> str:
        value = _CREDENTIAL_URL.sub(r"\1[redacted]@", value)
        return _SENSITIVE_VALUE.sub(r"\1=[redacted]", value)

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            record.args = tuple(
                self._redact(item) if isinstance(item, str) else item for item in record.args
            )
        return True


def configure_error_logging(project_root: str | Path) -> logging.Logger:
    """Create one rotating local error log and return the project logger.

    The function is safe to call repeatedly during Streamlit reruns. Logs are
    kept under ``logs/smoke_detection.log`` and rotated at 5 MB per file.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_smoke_detection_configured", False):
        return logger

    log_directory = Path(project_root).resolve() / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = RotatingFileHandler(
        log_directory / "smoke_detection.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.addFilter(_SensitiveDataFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    logger._smoke_detection_configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(component: str | None = None) -> logging.Logger:
    """Return a named project logger after ``configure_error_logging`` runs."""
    return logging.getLogger(f"{LOGGER_NAME}.{component}" if component else LOGGER_NAME)
