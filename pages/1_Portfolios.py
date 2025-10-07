# pages/1_Portfolios.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Storage + data helpers from your project
from src.storage import (
    get_ohlcv_root,
    list_index_cache,
    load_index_members,
    save_portfolio,
)
from src.data.loader import get_ohlcv
from src.data.portfolio_prefetch import list_cached_shards, now_utc_iso

# -----------------------
# Page config / Title
# -----------------------
st.set_page_config(page_title="Portfolios", layout="wide")
st.title("📊 Portfolios")

ss = st.session_state


# -----------------------
# Small utils
# -----------------------

def _add_range_from_bars(ranges_dict: dict, symbol: str, bars: pd.DataFrame) -> None:
    """Update per-symbol min/max date coverage using a bars DataFrame."""
    if bars is None or bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
        return
    idx = bars.index
    # Normalize index to UTC date boundaries
    if idx.tz is None:
        s = pd.Timestamp(idx.min(), tz="UTC")
        e = pd.Timestamp(idx.max(), tz="UTC")
    else:
        s = idx.min().tz_convert("UTC")
        e = idx.max().tz_convert("UTC")
    s_iso = s.date().isoformat()
    e_iso = e.date().isoformat()
    prev = ranges_dict.get(symbol)
    if not prev:
        ranges_dict[symbol] = {"start": s_iso, "end": e_iso}
    else:
        # expand coverage if we fetched more
        ranges_dict[symbol] = {
            "start": min(prev.get("start") or s_iso, s_iso),
            "end": max(prev.get("end") or e_iso, e_iso),
        }

def _upsert_liquidity_cols(meta_df: pd.DataFrame, liq_rows: list[dict]) -> pd.DataFrame:
    """
    Update/overwrite liquidity columns on meta_df using liq_rows
    (each row has: symbol, median_close_priors, median_dollar_vol_priors)
    without using DataFrame.merge (so no overlap errors).
    """
    out = meta_df.copy()

    # Ensure columns exist exactly once
    for col in ("median_close_priors", "median_dollar_vol_priors"):
        if col not in out.columns:
            out[col] = None
    # drop any accidental duplicates by name (just in case)
    out = out.loc[:, ~out.columns.duplicated()]

    # Build liq_df with a stable schema
    liq_df = pd.DataFrame(
        liq_rows,
        columns=["symbol", "median_close_priors", "median_dollar_vol_priors"],
    )
    if liq_df.empty:
        return out

    liq_df = liq_df.drop_duplicates(subset=["symbol"]).set_index("symbol")
    close_map = liq_df["median_close_priors"].to_dict()
    dvol_map = liq_df["median_dollar_vol_priors"].to_dict()

    out["median_close_priors"] = out["symbol"].map(close_map)
    out["median_dollar_vol_priors"] = out["symbol"].map(dvol_map)
    return out

def _to_dt(d: date | datetime) -> datetime:
    """
    Return a UTC-naive datetime (tzinfo=None). If an aware datetime is passed,
    convert to UTC and drop tzinfo. This avoids provider-side tz errors.
    """
    if isinstance(d, datetime):
        dt = d
    else:
        dt = datetime(d.year, d.month, d.day)
    if dt.tzinfo is None:
        return dt  # assume already UTC-naive
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _normalize_symbols(seq) -> list[str]:
    seen, out = set(), []
    for x in seq or []:
        s = str(x).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _ensure_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a 'symbol' column exists, building it from common alternatives if needed.
    """
    if "symbol" in df.columns:
        return df
    candidates = [c for c in ["Ticker", "ticker", "Symbol", "SYMBOL", "secid"] if c in df.columns]
    if candidates:
        df = df.rename(columns={candidates[0]: "symbol"})
    else:
        # If truly absent, create an empty symbol column to avoid hard failures.
        df = df.copy()
        df["symbol"] = []
    return df


@st.cache_data(show_spinner=False)
def _load_index_df(index_key: str | None) -> pd.DataFrame:
    """
    Try to load index metadata via storage helpers. Fallback to a tiny
    built-in sample so the page stays functional if nothing is cached yet.
    Expected columns: symbol, name, sector (extra cols are fine).
    """
    # 1) load from storage if an index_key was chosen
    df = pd.DataFrame()
    if index_key:
        payload = load_index_members(index_key)
        if isinstance(payload, dict):
            ss["_pf_index_payload"] = payload
        else:
            ss.pop("_pf_index_payload", None)
        # Common shapes:
        # { "members": [{"symbol":...,"name":...,"sector":...}, ...] }
        # { "symbols": ["AAPL","MSFT", ...] }
        if isinstance(payload, dict):
            if "members" in payload and isinstance(payload["members"], list):
                df = pd.DataFrame(payload["members"])
            elif "symbols" in payload and isinstance(payload["symbols"], list):
                df = pd.DataFrame({"symbol": payload["symbols"]})
            else:
                # Fallback: try to coerce dict into a frame (best effort)
                try:
                    df = pd.DataFrame(payload)
                except Exception:
                    df = pd.DataFrame()
        elif isinstance(payload, list):
            # Maybe already a list of dicts or strings
            try:
                df = pd.DataFrame(payload)
            except Exception:
                df = pd.DataFrame()
    else:
        ss.pop("_pf_index_payload", None)

    # 2) fallback if nothing in storage or malformed
    if df.empty:
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA"],
                "name": ["Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla"],
                "sector": ["Tech", "Tech", "Consumer", "Tech", "Tech", "Auto"],
            }
        )

    # Normalize 'symbol' column and optional cols
    df = _ensure_symbol_column(df)
    df["symbol"] = _normalize_symbols(df["symbol"].tolist())
    for col in ("name", "sector"):
        if col not in df.columns:
            df[col] = None
    return df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)


# -----------------------
# Sidebar / Inputs
# -----------------------
st.sidebar.header("Universe")

# Discover available indexes from storage
available_indexes = list_index_cache()
index_key = st.sidebar.selectbox(
    "Index universe (from storage)",
    options=(available_indexes or ["<built-in sample>"]),
    index=0,
)

index_meta_payload = ss.get("_pf_index_payload") if index_key != "<built-in sample>" else None
if isinstance(index_meta_payload, dict):
    source_meta = index_meta_payload.get("meta", {})
    total = len(index_meta_payload.get("symbols") or [])
    source_bits = []
    if total:
        source_bits.append(f"{total} tickers")
    if source_meta.get("source_type"):
        source_bits.append(str(source_meta["source_type"]))
    if source_meta.get("source_path"):
        source_bits.append(Path(source_meta["source_path"]).name)
    if source_bits:
        st.sidebar.caption(" • ".join(source_bits))

# Core page controls
col_top1, col_top2, col_top3 = st.columns([1, 1, 1])
with col_top1:
    max_portfolio_size = st.number_input(
        "Max tickers in portfolio",
        min_value=1,
        max_value=2000,
        value=50,
        step=1,
        help="Caps the number of tickers that proceed through fetch and filtering.",
    )
with col_top2:
    priors_years = st.number_input(
        "Years of priors for liquidity",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Window used to compute median close & median $ volume.",
    )
with col_top3:
    st.markdown(" ")

# Load universe df and apply simple pre-filters
df_universe = _load_index_df(None if index_key == "<built-in sample>" else index_key)

# Optional search / sector filters
col_f1, col_f2 = st.columns([1, 1])
with col_f1:
    q = st.text_input("Search (symbol or name)", value="").strip().upper()
with col_f2:
    sectors = sorted([s for s in df_universe["sector"].dropna().unique().tolist()])
    chosen_sectors = st.multiselect("Sectors", sectors, default=sectors)

mask = pd.Series(True, index=df_universe.index)
if q:
    mask = mask & (
        df_universe["symbol"].str.contains(q, case=False, na=False)
        | df_universe["name"].fillna("").str.contains(q, case=False, na=False)
    )
if chosen_sectors:
    mask = mask & df_universe["sector"].isin(chosen_sectors)

meta_filt = df_universe.loc[mask].reset_index(drop=True)

st.markdown(f"**Filtered universe:** {len(meta_filt)} / {len(df_universe)} tickers")

# ---- cap the portfolio here (NEW behavior) ----
meta_filt_capped = meta_filt.head(int(max_portfolio_size)).copy()
st.caption(f"Applied portfolio cap → working universe: **{len(meta_filt_capped)}** tickers")

# Prepare priors window
priors_end = datetime.now(timezone.utc)
priors_start = priors_end - timedelta(days=365 * int(priors_years))

# Add liquidity columns (will be filled post-fetch)
for col in ("median_close_priors", "median_dollar_vol_priors"):
    if col not in meta_filt_capped.columns:
        meta_filt_capped[col] = None

st.divider()

# -----------------------
# Fetch + Liquidity
# -----------------------
st.subheader("Fetch bars & compute liquidity")

col_btn, col_help = st.columns([1, 3])
with col_help:
    st.write(
        "This will prefetch OHLCV to disk cache for the **capped** symbols and compute "
        "median close & median dollar volume over the priors window. Subsequent runs will read from cache/RAM."
    )

if st.button("📉 Fetch OHLCV & compute liquidity", type="primary"):
    tickers = meta_filt_capped["symbol"].tolist()
    if not tickers:
        st.warning("No tickers in the working universe after filters/cap.")
        st.stop()

    # Prefetch + compute liquidity + build per-symbol ranges inline
    per_ranges: dict[str, dict[str, str]] = {}
    liq_rows: list[dict] = []
    prog = st.progress(0.0)
    for i, sym in enumerate(tickers, start=1):
        try:
            # Warm cache (and RAM on repeat) and return bars
            bars = get_ohlcv(sym, _to_dt(priors_start), _to_dt(priors_end), "1D")

            # Liquidity
            med_close = float(bars["close"].median())
            med_dvol = float((bars["close"] * bars["volume"]).median())
            liq_rows.append(
                {"symbol": sym, "median_close_priors": med_close, "median_dollar_vol_priors": med_dvol}
            )

            # Coverage range for this symbol
            _add_range_from_bars(per_ranges, sym, bars)
        except Exception as e:
            st.write(f"[liquidity] {sym}: {e!r}")
        finally:
            prog.progress(i / len(tickers))
    prog.empty()

    # Persist ranges for saving into portfolio meta
    ss["pf_per_ranges"] = per_ranges

    # Capture cached shard metadata (provider, path, start/end) for persistence
    ss["pf_data_shards"] = list_cached_shards(tickers, timeframe="1D")
    ss["pf_data_shards_root"] = str(get_ohlcv_root())

    # Upsert liquidity columns without merging (no overlap)
    meta_filt_capped = _upsert_liquidity_cols(meta_filt_capped, liq_rows)

    # Store in session so filters below operate on enriched data
    ss["universe_enriched"] = meta_filt_capped

# Use enriched if available, otherwise fall back to capped
enriched = ss.get("universe_enriched", meta_filt_capped.copy())
enriched = enriched.loc[:, ~enriched.columns.duplicated()]

# -----------------------
# Liquidity filters
# -----------------------
st.subheader("Filter by liquidity")

col_l1, col_l2 = st.columns(2)
with col_l1:
    min_price = st.number_input("Min median close (priors)", value=0.0, step=1.0)
    max_price = st.number_input(
        "Max median close (priors)",
        value=0.0,
        step=1.0,
        help="Set to 0 to disable the max filter.",
    )
with col_l2:
    min_dvol = st.number_input("Min median $ volume (priors)", value=0.0, step=1.0)
    max_dvol = st.number_input(
        "Max median $ volume (priors)",
        value=0.0,
        step=1.0,
        help="Set to 0 to disable the max filter.",
    )

if not enriched.empty:
    work = enriched.copy()
    work["median_close_priors"] = pd.to_numeric(work["median_close_priors"], errors="coerce")
    work["median_dollar_vol_priors"] = pd.to_numeric(
        work["median_dollar_vol_priors"], errors="coerce"
    )

    price_series = work["median_close_priors"]
    dvol_series = work["median_dollar_vol_priors"]

    mask_price = price_series.fillna(0) >= float(min_price)
    if float(max_price) > 0:
        mask_price &= price_series.fillna(float("inf")) <= float(max_price)

    mask_dvol = dvol_series.fillna(0) >= float(min_dvol)
    if float(max_dvol) > 0:
        mask_dvol &= dvol_series.fillna(float("inf")) <= float(max_dvol)

    mask2 = mask_price & mask_dvol
    filtered = work.loc[mask2].reset_index(drop=True)

    st.write(f"After liquidity filters: **{len(filtered)} / {len(work)}** tickers remain.")
    st.dataframe(filtered,  width="stretch")
else:
    st.info("No OHLCV/liquidity computed yet.")

st.divider()

# -----------------------
# Save portfolio
# -----------------------
st.subheader("Save selection to portfolio")

col_s1, col_s2 = st.columns([2, 1])
with col_s1:
    portfolio_name = st.text_input("Portfolio name", value="my_portfolio").strip()
with col_s2:
    include_current_filters = st.checkbox("Save only filtered tickers", value=True)

to_save_df = ss.get("universe_enriched", meta_filt_capped.copy())
if include_current_filters and not to_save_df.empty:
    save_price = pd.to_numeric(to_save_df["median_close_priors"], errors="coerce")
    save_dvol = pd.to_numeric(to_save_df["median_dollar_vol_priors"], errors="coerce")

    mask_save_price = save_price.fillna(0) >= float(min_price)
    if float(max_price) > 0:
        mask_save_price &= save_price.fillna(float("inf")) <= float(max_price)

    mask_save_dvol = save_dvol.fillna(0) >= float(min_dvol)
    if float(max_dvol) > 0:
        mask_save_dvol &= save_dvol.fillna(float("inf")) <= float(max_dvol)

    mask_save = mask_save_price & mask_save_dvol
    to_save_df = to_save_df.loc[mask_save].reset_index(drop=True)

tickers_to_save = _normalize_symbols(to_save_df["symbol"].tolist()) if not to_save_df.empty else []

meta_payload = {
    "source": "1_Portfolios",
    "index_key": None if index_key == "<built-in sample>" else index_key,
    "filters": {
        "search": q,
        "sectors": chosen_sectors,
        "min_median_close_priors": float(min_price),
        "max_median_close_priors": float(max_price),
        "min_median_dollar_vol_priors": float(min_dvol),
        "max_median_dollar_vol_priors": float(max_dvol),
    },
    "windows": {
        "priors": [_to_dt(priors_start).date().isoformat(), _to_dt(priors_end).date().isoformat()],
    },
    "count": len(tickers_to_save),
    "last_prefetch_at": now_utc_iso(),
    # NEW: persist per-symbol coverage from prefetch
    "per_symbol_ranges": ss.get("pf_per_ranges", {}),
    "data_cache_root": ss.get("pf_data_shards_root") or str(get_ohlcv_root()),
    "data_shards": {"1D": ss.get("pf_data_shards", {})},
}

disabled_save = not portfolio_name or not tickers_to_save

if st.button("💾 Save portfolio", type="primary", disabled=disabled_save):
    payload = save_portfolio(portfolio_name, tickers_to_save, meta=meta_payload)
    st.success(f"Saved portfolio **{payload['name']}** with **{len(payload['tickers'])}** tickers.")
    st.json(payload, expanded=False)