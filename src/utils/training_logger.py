# src/utils/training_logger.py
from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict

class TrainingLogger:
    """Append-only JSONL logger for EA/WFO runs. No Streamlit calls here."""

    def __init__(self, log_file: str | Path, *, level: int = logging.INFO) -> None:
        self.path = Path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"ea.{self.path.name}")
        if not self._logger.handlers:
            self._logger.setLevel(level)
            h = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            h.setFormatter(fmt)
            self._logger.addHandler(h)

    def _write(self, record: Dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log(self, event: str, payload: Dict[str, Any] | None = None) -> None:
        rec = {"event": event, "payload": payload or {}}
        self._write(rec)

    def log_error(self, context: Dict[str, Any], err: Exception) -> None:
        rec = {
            "event": "error",
            "payload": {
                "context": context,
                "error_type": type(err).__name__,
                "error_msg": str(err),
            },
        }
        self._logger.exception("EA error: %s", rec["payload"])
        self._write(rec)