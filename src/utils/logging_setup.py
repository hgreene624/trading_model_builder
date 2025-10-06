"""Utility helpers for configuring structured logging outputs."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOG_INITIALIZED = False


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure console + rotating file logging for backtests."""

    global _LOG_INITIALIZED

    log_level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    log_dir = os.path.join("storage", "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # Directory creation issues should not stop execution; logging remains console-only.
        pass

    file_path = os.path.join(log_dir, "engine.log")
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == file_path for h in root_logger.handlers):
        try:
            file_handler = RotatingFileHandler(file_path, maxBytes=5 * 1024 * 1024, backupCount=3)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception:
            # File handler is best-effort; fall back to console only if it fails.
            pass

    _LOG_INITIALIZED = True
    return root_logger


__all__ = ["setup_logging"]
