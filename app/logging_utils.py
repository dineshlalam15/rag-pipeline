"""Structured local JSON logging to stdout and optional rotating log file."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Formats log records as single-line JSON for local inspection."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    enable_file_logs: bool = False,
    log_file_path: str = "./logs/rag_pipeline.log",
) -> logging.Logger:
    """Configure root logger with JSON stdout and optional local rotating log file."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = JsonLogFormatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if enable_file_logs:
        target = Path(log_file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(target, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return logging.getLogger("rag_pipeline")
