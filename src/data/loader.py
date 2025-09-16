# src/data/loader.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
import importlib
import pandas as pd


# -------------------- internal: column normalization -------------------- #
def _normalize_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """
    Ensure columns: open, high, low, close, volume; DatetimeIndex in UTC; sorted ascending.

    Robustly handles MultiIndex columns from providers that return shapes like:
      ('Open','AAPL') or ('AAPL','Open') or ('AAPL','open') ...
    If `symbol` is provided, we try to slice the MultiIndex on ANY level that matches it.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # If MultiIndex columns, try to slice to the requested symbol on any level
    if isinstance(df.columns, pd.MultiIndex):
        if symbol:
            sym_lower = str(symbol).lower()
            # Try each level to find the ticker
            for lvl in range(df.columns.nlevels):
                level_vals = [str(x).lower() for x in df.columns.get_level_values(lvl)]
                if sym_lower in level_vals:
                    try:
                        sliced = df.xs(symbol, axis=1, level=lvl)
                        df = sliced
                        break
                    except Exception:
                        # Sometimes levels hold upper/lower symbols inconsistently - try case-insensitive
                        # Build a mapper to match case-insensitive
                        idx_map = {}
                        for tup in df.columns:
                            parts = [str(x) for x in (tup if isinstance(tup, tuple) else (tup,))]
                            if len(parts) > lvl and parts[lvl].lower() == sym_lower:
                                idx_map[tup] = tuple([p for i, p in enumerate(parts) if i != lvl])
                        if idx_map:
                            df = df.rename(columns=idx_map)
                            # Drop the now-redundant level if still MultiIndex
                            if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 1:
                                # Try to collapse duplicated levels if any
                                try:
                                    df.columns = pd.MultiIndex.from_tuples(
                                        [tuple(c for c in col if c is not None) for col in df.columns]
                                    )
                                except Exception:
                                    pass
                            break

        # If still MultiIndex after slicing, flatten by picking the token that looks like a field name
        if isinstance(df.columns, pd.MultiIndex):
            def pick_field(col) -> str:
                # Examine every element for OHLCV keywords
                parts = [str(x).strip().lower() for x in (col if isinstance(col, tuple) else (col,))]
                wanted = {"open", "high", "low", "close", "volume", "adj close", "adj_close", "adjclose"}
                for p in parts:
                    if p in wanted:
                        return p
                # aliases
                alias = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
                for p in parts:
                    if p in alias:
                        return alias[p]
                # fallback: last token
                return parts[-1] if parts else "close"

            df.columns = [pick_field(col) for col in df.columns]

    # Lower-case all columns for easier matching
    df.columns = [str(c).lower() for c in df.columns]

    # Map common variants to canonical names
    rename_map = {
        "adj close": "adj_close",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "open_price": "open",
        "close_price": "close",
        "high_price": "high",
        "low_price": "low",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    df.rename(columns=rename_map, inplace=True)

    # If we still don’t have all OHLC, just return what we have (caller can decide to error)
    have = set(df.columns)
    need = {"open", "high", "low", "close"}
    if not need.issubset(have):
        return df

    # Build a datetime index if needed
    if not isinstance(df.index, pd.DatetimeIndex):
        lower = {c.lower(): c for c in df.columns}
        for t in ("timestamp", "time", "date", "datetime"):
            if t in lower:
                df.index = pd.to_datetime(df[lower[t]], utc=True, errors="coerce")
                if t != "timestamp":
                    df.drop(columns=[lower[t]], inplace=True, errors="ignore")
                break

    # Ensure UTC tz-aware index
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    df.sort_index(inplace=True)

    # Keep only the core columns if present
    keep = [c for c in ("open", "high", "low", "close", "volume", "adj_close") if c in df.columns]
    return df[keep] if keep else df


# -------------------- public: unified loader -------------------- #
def get_ohlcv(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified OHLCV loader used by training/backtests.
    Tries Alpaca first; on any error or empty result, falls back to Yahoo Finance.
    Returns a DataFrame with columns: open, high, low, close, [volume[, adj_close]], UTC index.
    """
    # Try Alpaca (modern alpaca-py backend)
    try:
        alpaca = importlib.import_module("src.data.alpaca_data")
        df_a = alpaca.load_ohlcv(symbol, start, end, timeframe or "1Day")
        df_a = _normalize_ohlcv(df_a, symbol)
        if df_a is not None and not df_a.empty and {"open", "high", "low", "close"}.issubset({c.lower() for c in df_a.columns}):
            return df_a
    except Exception:
        # swallow and fallback to YF
        pass

    # Fallback: Yahoo Finance
    yf = importlib.import_module("src.data.yf")
    df_y = yf.load_ohlcv(symbol, start, end, timeframe or "1d")
    df_y = _normalize_ohlcv(df_y, symbol)
    return df_y