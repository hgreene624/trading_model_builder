# src/data/universe.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

DEFAULT_INDEX_PATHS = [
    Path("data/indexes.json"),
    Path("storage/indexes.json"),
]

def load_indexes(path: str | Path | None = None) -> Dict[str, List[str]]:
    if path:
        p = Path(path)
        return json.loads(p.read_text())
    for cand in DEFAULT_INDEX_PATHS:
        if cand.exists():
            return json.loads(cand.read_text())
    return {}  # caller handles empty

def available_universes() -> List[str]:
    idx = load_indexes()
    return sorted(idx.keys())

def get_universe(name: str) -> List[str]:
    idx = load_indexes()
    return idx.get(name, [])