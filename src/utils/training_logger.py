# src/utils/training_logger.py
"""
TrainingLogger: event-driven JSONL logger for evolutionary/walk-forward runs.

- Each event is a dict: {timestamp, event, payload}
- Writes JSONL for later troubleshooting & replay
- Captures lifecycle, progress, evaluation results, anomalies (under_min_trades, degenerate_fitness), and errors
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    def __init__(self, log_path: str = "training.log", also_buffer: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = [] if also_buffer else None
        self.log("session_start", {"note": "Training session started"})

    def log(self, event: str, payload: Dict[str, Any]):
        """Write a structured event to file (and memory buffer if enabled)."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "payload": payload,
        }
        line = json.dumps(entry, default=str)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.buffer is not None:
            self.buffer.append(entry)

    def log_error(self, context: Dict[str, Any], err: Exception):
        """Log errors with traceback plus context (gen, idx, params, etc.)."""
        self.log("error", {
            **context,
            "error": str(err),
            "traceback": traceback.format_exc(),
        })

    def log_anomaly(self, event_name: str, context: Dict[str, Any]):
        """
        Log non-fatal, unintended behaviors (e.g., under_min_trades, degenerate_fitness).
        """
        self.log(event_name, context)

    def close(self):
        """Mark end of session."""
        self.log("session_end", {"note": "Training session ended"})