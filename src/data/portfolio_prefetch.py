# src/data/portfolio_prefetch.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.data.loader import get_ohlcv
from src.storage import get_ohlcv_root


@dataclass(frozen=True)
class SymbolRange:
    symbol: str
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]

    def as_iso(self) -> Tuple[Optional[str], Optional[str]]:
        s = self.start.tz_convert("UTC").date().isoformat() if isinstance(self.start, pd.Timestamp) else None
        e = self.end.tz_convert("UTC").date().isoformat() if isinstance(self.end, pd.Timestamp) else None
        return s, e


# ---------------------------
# Cache filesystem utilities
# ---------------------------

_CACHE_FILE_RE = re.compile(r"(?P<s>\d{4}-\d{2}-\d{2})__(?P<e>\d{4}-\d{2}-\d{2})\.parquet$", re.IGNORECASE)

def _cache_dirs_for(symbol: str, timeframe: str = "1D") -> List[Path]:
    """Return provider-specific cache directories for a symbol/timeframe (alpaca & yahoo)."""
    root = get_ohlcv_root()
    sym = symbol.upper().replace("/", "_")
    tf = (timeframe or "1D").upper()
    return [
        root / "alpaca" / sym / tf,
        root / "yahoo" / sym / tf,
    ]


def _parse_cache_span(path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    m = _CACHE_FILE_RE.search(path.name)
    if not m:
        return None
    s = pd.Timestamp(m.group("s"), tz="UTC")
    e = pd.Timestamp(m.group("e"), tz="UTC")
    return s, e


def get_cached_symbol_range(symbol: str, timeframe: str = "1D") -> SymbolRange:
    """Scan cache files for a symbol/timeframe and return (min_start, max_end)."""
    mins, maxs = [], []
    for d in _cache_dirs_for(symbol, timeframe):
        if not d.exists():
            continue
        for f in d.glob("*.parquet"):
            span = _parse_cache_span(f)
            if span:
                s, e = span
                mins.append(s)
                maxs.append(e)
    if not mins or not maxs:
        return SymbolRange(symbol=symbol.upper(), start=None, end=None)
    return SymbolRange(symbol=symbol.upper(), start=min(mins), end=max(maxs))


def get_cached_ranges(symbols: Iterable[str], timeframe: str = "1D") -> Dict[str, SymbolRange]:
    out: Dict[str, SymbolRange] = {}
    for s in symbols or []:
        s2 = str(s).strip().upper()
        if not s2:
            continue
        out[s2] = get_cached_symbol_range(s2, timeframe=timeframe)
    return out


# ---------------------------
# Prefetch + ranges (main)
# ---------------------------

def prefetch_and_ranges(
    symbols: List[str],
    start: datetime,
    end: datetime,
    timeframe: str = "1D",
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    1) Prefetches OHLCV to disk cache via get_ohlcv() (idempotent).
    2) Reads cache to compute per-symbol min/max date coverage.
    Returns a dict suitable for persisting into portfolio meta:
        {
          "AAPL": {"start": "2012-01-01", "end": "2025-09-24"},
          "MSFT": {"start": "2010-06-15", "end": "2025-09-24"}
        }
    """
    syms = [str(x).strip().upper() for x in (symbols or []) if str(x).strip()]
    if not syms:
        return {}

    # Warm cache (safe to re-run)
    for s in syms:
        try:
            _ = get_ohlcv(s, start, end, timeframe=timeframe)
        except Exception as e:
            print(f"[portfolio_prefetch] {s} prefetch error (continuing): {e!r}")

    # Compute ranges from cache
    ranges = get_cached_ranges(syms, timeframe=timeframe)
    result: Dict[str, Dict[str, Optional[str]]] = {}
    for sym, rng in ranges.items():
        s_iso, e_iso = rng.as_iso()
        result[sym] = {"start": s_iso, "end": e_iso}
    return result


# ---------------------------
# Convenience helpers
# ---------------------------

def intersection_range(meta_ranges: Dict[str, Dict[str, Optional[str]]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Given the dict returned by prefetch_and_ranges(...), compute the intersection window across symbols.
    Returns ISO dates (start_iso, end_iso) or (None, None) if insufficient data.
    """
    starts, ends = [], []
    for _, rng in (meta_ranges or {}).items():
        s, e = rng.get("start"), rng.get("end")
        if s and e:
            starts.append(pd.Timestamp(s, tz="UTC"))
            ends.append(pd.Timestamp(e, tz="UTC"))
    if not starts or not ends:
        return None, None
    s_star = max(starts).date().isoformat()
    e_star = min(ends).date().isoformat()
    if s_star > e_star:
        return None, None
    return s_star, e_star


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()