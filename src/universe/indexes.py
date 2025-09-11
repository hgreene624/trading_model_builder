"""
Fetch and cache index constituents (symbols + name + sector if available).

We try Wikipedia via pandas.read_html (requires lxml or html5lib). If that fails,
we fall back to any existing cached copy under storage/universe/{key}.json.

Supported:
- sp500
- nasdaq100
- dow30
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import time
import json

import pandas as pd


CACHE_DIR = Path("storage/universe")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class IndexSpec:
    key: str
    title: str
    wiki_url: str
    symbol_col_candidates: List[str]
    name_col_candidates: List[str]
    sector_col_candidates: List[str]


SUPPORTED_INDEXES: Dict[str, IndexSpec] = {
    "sp500": IndexSpec(
        key="sp500",
        title="S&P 500",
        wiki_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        symbol_col_candidates=["Symbol", "Ticker"],
        name_col_candidates=["Security", "Company", "Name"],
        sector_col_candidates=["GICS Sector", "Sector"],
    ),
    "nasdaq100": IndexSpec(
        key="nasdaq100",
        title="Nasdaq-100",
        wiki_url="https://en.wikipedia.org/wiki/Nasdaq-100",
        symbol_col_candidates=["Ticker", "Symbol"],
        name_col_candidates=["Company", "Name", "Security"],
        sector_col_candidates=["GICS Sector", "Sector"],
    ),
    "dow30": IndexSpec(
        key="dow30",
        title="Dow 30",
        wiki_url="https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        symbol_col_candidates=["Symbol", "Ticker"],
        name_col_candidates=["Company", "Name", "Security"],
        sector_col_candidates=["Industry", "Sector"],
    ),
}


def _read_wiki_table(url: str) -> Optional[pd.DataFrame]:
    try:
        tables = pd.read_html(url)
        # Heuristic: choose the biggest table with a 'Symbol'/'Ticker' column
        best = None
        best_rows = 0
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("Symbol" in c or "Ticker" in c for c in cols):
                if len(t) > best_rows:
                    best = t
                    best_rows = len(t)
        return best
    except Exception:
        return None


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # also try loose match ignoring case and spaces
    low = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = c.strip().lower()
        if key in low:
            return low[key]
    return None


def _normalize_symbols(s: pd.Series) -> pd.Series:
    # Strip whitespace, drop periods on some symbol formats (e.g., BRK.B -> BRK-B? We'll leave as is)
    return s.astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True)


def fetch_index(spec_key: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame with columns: symbol, name, sector, index_key, fetched_at.
    Caches to storage/universe/{key}.json
    """
    if spec_key not in SUPPORTED_INDEXES:
        raise ValueError(f"Unsupported index key: {spec_key}")

    spec = SUPPORTED_INDEXES[spec_key]
    cache_file = CACHE_DIR / f"{spec.key}.json"

    # Load from cache if not forced
    if cache_file.exists() and not force_refresh:
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return pd.DataFrame(data)
        except Exception:
            pass  # fallthrough to refetch

    df = _read_wiki_table(spec.wiki_url)
    if df is None or df.empty:
        # fallback to cache if exists
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return pd.DataFrame(data)
        raise RuntimeError(f"Failed to read index members from {spec.title} wiki page")

    sym_col = _pick_col(df, spec.symbol_col_candidates)
    name_col = _pick_col(df, spec.name_col_candidates)
    sector_col = _pick_col(df, spec.sector_col_candidates)

    if sym_col is None:
        raise RuntimeError(f"Could not find symbol column for {spec.title}")

    out = pd.DataFrame()
    out["symbol"] = _normalize_symbols(df[sym_col])
    out["name"] = df[name_col].astype(str) if name_col else ""
    out["sector"] = df[sector_col].astype(str) if sector_col else ""
    out["index_key"] = spec.key
    out["index_title"] = spec.title
    out["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    # Dedup (some tables include class shares)
    out = out.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    # Cache
    try:
        cache_file.write_text(out.to_json(orient="records"), encoding="utf-8")
    except Exception:
        pass

    return out


def fetch_indexes(keys: List[str], force_refresh: bool = False) -> pd.DataFrame:
    frames = []
    for k in keys:
        try:
            frames.append(fetch_index(k, force_refresh=force_refresh))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["symbol", "name", "sector", "index_key", "index_title", "fetched_at"])
    return pd.concat(frames, ignore_index=True)


def supported_indexes() -> pd.DataFrame:
    return pd.DataFrame([
        {"key": spec.key, "title": spec.title, "wiki_url": spec.wiki_url}
        for spec in SUPPORTED_INDEXES.values()
    ])


# ---- Simulation listing helper (for Home.py) ---------------------------------
# Lightweight lister that scans JSON artifacts produced by simulations/reports.
# It normalizes a few common fields so the Home page can render a table.

from pathlib import Path as _Path
import json as _json
import time as _time
from typing import List as _List, Dict as _Dict, Tuple as _Tuple

def list_simulations(limit: int = 50,
                     roots: _Tuple[str, ...] = ("storage/simulations", "storage/reports"),
                     extensions: _Tuple[str, ...] = (".json",)) -> _List[_Dict]:
    items: _List[_Dict] = []
    for root in roots:
        p = _Path(root)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in extensions:
                continue
            try:
                data = {}
                try:
                    data = _json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    # Not a JSON we understand — we still include the file row with defaults
                    data = {}

                mtime = f.stat().st_mtime
                created = data.get("created_at") or _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(mtime))

                items.append({
                    "path": str(f),
                    "name": f.name,
                    "created_at": created,
                    "portfolio_name": data.get("portfolio_name") or data.get("portfolio") or data.get("meta", {}).get("portfolio", ""),
                    "start": data.get("start") or data.get("date_start") or data.get("start_date") or data.get("meta", {}).get("start", ""),
                    "end": data.get("end") or data.get("date_end") or data.get("end_date") or data.get("meta", {}).get("end", ""),
                    "starting_equity": data.get("starting_equity", data.get("start_equity", None)),
                    "final_equity": data.get("final_equity", data.get("equity_final", None)),
                    "modified_ts": mtime,
                })
            except Exception:
                # Skip unreadable/locked files
                continue

    # newest first
    items.sort(key=lambda r: r.get("modified_ts", 0), reverse=True)
    if limit and limit > 0:
        items = items[:limit]
    return items

# Ensure symbol is exported if __all__ is present
try:
    __all__
except NameError:
    __all__ = []
if isinstance(__all__, list) and "list_simulations" not in __all__:
    __all__.append("list_simulations")