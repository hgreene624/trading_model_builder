# src/universe/indexes.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

# We try to fetch from Wikipedia with pandas.read_html.
# To harden against environments without HTML parsers or flaky network,
# we cache results under storage/universe/*.json and fall back to a small
# built-in list so the UI never hard-stops with "No members fetched".

UNIVERSE_DIR = Path("storage/universe")
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED: Dict[str, Dict[str, str]] = {
    "sp500": {
        "name": "S&P 500",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    },
    "nasdaq100": {
        "name": "Nasdaq-100",
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
    },
    "dow30": {
        "name": "Dow 30",
        "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    },
}

# Minimal seeds to avoid empty results if web + cache both fail
FALLBACK: Dict[str, List[str]] = {
    "sp500": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM", "XOM", "UNH"],
    "nasdaq100": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "PEP", "AVGO", "COST"],
    "dow30": ["AAPL", "MSFT", "GS", "JPM", "DIS", "HD", "KO", "PG", "V", "WMT"],
}


def supported_indexes() -> List[str]:
    return list(SUPPORTED.keys())


def _cache_path(key: str) -> Path:
    return UNIVERSE_DIR / f"{key}.json"


def _norm_symbol(x: object) -> str:
    s = str(x).strip().upper()
    # Common class-share normalizations (BRK.B -> BRK-B)
    s = s.replace(".", "-")
    # Remove footnote markers or stray characters
    s = s.replace("†", "").replace("*", "")
    # Very short guards
    return s


def _extract_wiki_tableframes(html: str) -> List[pd.DataFrame]:
    # pandas.read_html requires an HTML parser (lxml or html5lib). If not
    # installed, this will raise and we’ll fall back to cache/seed.
    try:
        return pd.read_html(html)  # type: ignore[attr-defined]
    except Exception:
        return []


def _pick_members_table(tables: Sequence[pd.DataFrame]) -> Optional[pd.DataFrame]:
    # Heuristics: find a table containing a plausible ticker column
    cand_names = {"symbol", "ticker", "code"}
    best: Optional[pd.DataFrame] = None
    best_score = 0

    for t in tables:
        # Normalize col labels
        cols = [str(c).strip() for c in t.columns]
        lower = [c.lower() for c in cols]
        colmap = {c.lower(): c for c in cols}
        # candidate ticker column present?
        ticker_key = next((c for c in lower if c in cand_names), None)
        if ticker_key is None:
            continue
        tk_col = colmap[ticker_key]
        # score by number of non-null entries
        score = t[tk_col].notna().sum()
        if score > best_score:
            best = t.copy()
            best_score = score
    return best


def _coerce_members(df: pd.DataFrame, index_key: str) -> pd.DataFrame:
    # Map commonly occurring column names to canonical ones
    rename_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("symbol", "ticker", "code"):
            rename_map[c] = "symbol"
        elif lc in ("security", "company", "company name", "name"):
            rename_map[c] = "name"
        elif lc in ("gics sector", "sector"):
            rename_map[c] = "sector"
        elif lc in ("gics sub-industry", "industry", "sub-industry"):
            rename_map[c] = "industry"

    df2 = df.rename(columns=rename_map)
    if "symbol" not in df2.columns:
        # Give up on this table
        return pd.DataFrame(columns=["symbol", "name", "sector", "industry", "index", "source"])  # empty

    out = pd.DataFrame()
    out["symbol"] = df2["symbol"].map(_norm_symbol)
    out["name"] = df2.get("name")
    out["sector"] = df2.get("sector", "Unknown")
    out["industry"] = df2.get("industry")
    out["index"] = index_key
    out["source"] = "web:wikipedia"

    # Drop NA and duplicates
    out = out.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return out


def _fetch_from_web(key: str, url: str) -> pd.DataFrame:
    # Use requests with a real UA to reduce blocking
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    tables = _extract_wiki_tableframes(r.text)
    if not tables:
        return pd.DataFrame(columns=["symbol", "name", "sector", "industry", "index", "source"])  # empty

    t = _pick_members_table(tables)
    if t is None:
        return pd.DataFrame(columns=["symbol", "name", "sector", "industry", "index", "source"])  # empty

    return _coerce_members(t, key)


def _read_cache(key: str) -> Optional[pd.DataFrame]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(payload.get("rows", []))
        return df
    except Exception:
        return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    p = _cache_path(key)
    payload = {
        "index": key,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": df.to_dict(orient="records"),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_index(key: str, force_refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame with columns: symbol, name, sector, industry, index, source.

    Order of attempts: cache → web → built-in fallback seeds.
    """
    if key not in SUPPORTED:
        raise KeyError(f"Unsupported index key: {key}")

    if not force_refresh:
        cached = _read_cache(key)
        if cached is not None and len(cached) > 0:
            return cached

    # Try web fetch
    try:
        df = _fetch_from_web(key, SUPPORTED[key]["url"])
        if len(df) > 0:
            _write_cache(key, df)
            return df
    except Exception:
        pass  # fall through to cache/fallback

    # Try cache again (maybe created earlier by another run)
    cached = _read_cache(key)
    if cached is not None and len(cached) > 0:
        return cached

    # Final fallback: minimal seed list
    seeds = FALLBACK.get(key, [])
    df = pd.DataFrame({
        "symbol": [
            _norm_symbol(s) for s in seeds
        ],
        "name": None,
        "sector": "Unknown",
        "industry": None,
        "index": key,
        "source": "fallback:seed",
    })
    if len(df) > 0:
        _write_cache(key, df)
    return df


def fetch_indexes(keys: Sequence[str], force_refresh: bool = False) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for k in keys:
        try:
            frames.append(fetch_index(k, force_refresh=force_refresh))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["symbol", "name", "sector", "industry", "index", "source"])  # empty
    df = pd.concat(frames, ignore_index=True)
    # de-duplicate symbols (keep the first occurrence)
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return df


__all__ = [
    "supported_indexes",
    "fetch_index",
    "fetch_indexes",
]