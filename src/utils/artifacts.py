# src/utils/artifacts.py
from __future__ import annotations
import json, csv, os, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional

ISO = "%Y-%m-%dT%H:%M:%S%z"

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)

def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("_") or "artifact"

def jsonl_append(path: str | Path, records: Iterable[Dict[str, Any]]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    return p

def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p

def write_csv(path: str | Path, rows: List[Dict[str, Any]], field_order: Optional[List[str]] = None) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    fields = field_order or sorted({k for r in rows for k in r.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p