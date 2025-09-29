# src/data/universe.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_INDEX_PATHS = [
    Path("data/indexes.json"),
    Path("storage/indexes.json"),
]


def _normalize_symbol(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.upper().replace(".", "-")
    return s


def _normalize_member_sequence(seq: Any) -> List[str]:
    symbols: List[str] = []
    if isinstance(seq, list):
        iterable = seq
    elif isinstance(seq, dict):
        iterable = list(seq.values())
    else:
        return symbols

    seen = set()
    for item in iterable:
        symbol: str | None = None
        if isinstance(item, dict):
            for key in ("symbol", "ticker", "code", "secid"):
                if key in item:
                    symbol = _normalize_symbol(item[key])
                    break
            if symbol is None and len(item) == 1:
                symbol = _normalize_symbol(next(iter(item.values())))
        else:
            symbol = _normalize_symbol(item)
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _extract_index_map(payload: Any) -> Dict[str, List[str]]:
    if not isinstance(payload, dict):
        return {}

    raw_indexes: Dict[str, Any] | None = None
    if isinstance(payload.get("indexes"), dict):
        raw_indexes = payload["indexes"]
    else:
        # treat top-level mapping if values look list/dict-like
        candidates = {
            k: v for k, v in payload.items() if isinstance(v, (list, dict))
        }
        if candidates:
            raw_indexes = candidates

    if not raw_indexes:
        return {}

    normalized: Dict[str, List[str]] = {}
    for name, raw in raw_indexes.items():
        if raw is None:
            continue
        symbols: List[str] = []
        if isinstance(raw, dict):
            for key in ("symbols", "tickers", "members", "rows", "data"):
                if key in raw and isinstance(raw[key], (list, dict)):
                    symbols = _normalize_member_sequence(raw[key])
                    if symbols:
                        break
            if not symbols and all(isinstance(v, (str, dict)) for v in raw.values()):
                # maybe {"AAPL": {...}, "MSFT": {...}}
                rows = []
                for sym_key, val in raw.items():
                    if isinstance(val, dict):
                        row = dict(val)
                        row.setdefault("symbol", sym_key)
                        rows.append(row)
                    else:
                        rows.append({"symbol": val})
                symbols = _normalize_member_sequence(rows)
        elif isinstance(raw, list):
            symbols = _normalize_member_sequence(raw)

        if symbols:
            normalized[str(name)] = symbols
    return normalized


def load_indexes(path: str | Path | None = None) -> Dict[str, List[str]]:
    if path:
        p = Path(path)
        if not p.exists():
            return {}
        payload = json.loads(p.read_text())
        return _extract_index_map(payload)

    for cand in DEFAULT_INDEX_PATHS:
        if cand.exists():
            payload = json.loads(cand.read_text())
            data = _extract_index_map(payload)
            if data:
                return data
    return {}


def available_universes() -> List[str]:
    idx = load_indexes()
    return sorted(idx.keys())


def get_universe(name: str) -> List[str]:
    idx = load_indexes()
    return idx.get(name, [])
