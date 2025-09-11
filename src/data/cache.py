# src/data/cache.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.data.alpaca_data import load_ohlcv

OHLCV_DIR = Path("storage/ohlcv")


def _sanitize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace("/", "-").replace(".", "-")


def _file_path(symbol: str, start: str, end: str, ext: str = "parquet") -> Path:
    sym = _sanitize_symbol(symbol)
    folder = OHLCV_DIR / sym
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"{sym}__{start}__{end}.{ext}"
    return folder / fname


def _write_df(df: pd.DataFrame, parquet_target: Path) -> Path:
    try:
        df.to_parquet(parquet_target, index=True)
        return parquet_target
    except Exception:
        csv_target = parquet_target.with_suffix(".csv")
        out = df.copy()
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "date"})
        out.to_csv(csv_target, index=False)
        return csv_target


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def _try_read_local(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    p_parq = _file_path(symbol, start, end, "parquet")
    p_csv = _file_path(symbol, start, end, "csv")
    if p_parq.exists():
        return _read_df(p_parq)
    if p_csv.exists():
        return _read_df(p_csv)
    return None


def get_ohlcv_persisted(symbol: str, start: str, end: str, *, force_refresh: bool = False) -> pd.DataFrame:
    if not force_refresh:
        cached = _try_read_local(symbol, start, end)
        if cached is not None and not cached.empty:
            return cached

    df = load_ohlcv(symbol, start, end)
    if df is None or df.empty:
        raise ValueError(f"No OHLCV for {symbol} between {start} and {end}")

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    target = _file_path(symbol, start, end, "parquet")
    _write_df(df, target)
    return df


@st.cache_data(show_spinner=False)
def get_ohlcv_cached(symbol: str, start: str, end: str, *, force_refresh: bool = False) -> pd.DataFrame:
    return get_ohlcv_persisted(symbol, start, end, force_refresh=force_refresh)


def list_local_ohlcv(symbol: str) -> list[Path]:
    sym = _sanitize_symbol(symbol)
    folder = OHLCV_DIR / sym
    if not folder.exists():
        return []
    return sorted(folder.glob(f"{sym}__*.*"))


def purge_local_ohlcv(symbol: Optional[str] = None) -> int:
    count = 0
    if symbol:
        roots = [OHLCV_DIR / _sanitize_symbol(symbol)]
    else:
        roots = [p for p in OHLCV_DIR.glob("*") if p.is_dir()]
    for root in roots:
        for p in root.glob("*.*"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
    return count

# pages/1_Portfolios.py
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.universe import indexes as UNV
from src.storage import list_portfolios, load_portfolio, save_portfolio

try:
    from src.data.cache import get_ohlcv_cached as load_ohlcv_window
except Exception:
    from src.data.alpaca_data import load_ohlcv as load_ohlcv_window


st.set_page_config(page_title="Portfolios", layout="wide")
load_dotenv()

st.title("🗂️ Portfolios — Universe → Filter → Fetch OHLCV → Save")

# =====================
# Controls: indexes & dates
# =====================
index_map = {meta["name"]: key for key, meta in UNV.SUPPORTED.items()}
choices = st.multiselect(
    "Select index collections",
    options=list(index_map.keys()),
    default=[name for name in ["S&P 500", "Nasdaq-100"] if name in index_map],
    key="idx_choices",
)
force_refresh = st.checkbox(
    "Force refresh index membership (fetch from Wikipedia)",
    value=False,
    help="If off, we use cached constituents under storage/universe/.",
    key="idx_force_refresh",
)

# Windows used later when fetching OHLCV
_today = date.today()
col_dt1, col_dt2, col_dt3, col_dt4 = st.columns(4)
with col_dt1:
    priors_start = st.date_input("Priors start", value=date(_today.year - 10, 1, 1), key="P_start")
with col_dt2:
    priors_end = st.date_input("Priors end", value=date(_today.year - 3, 12, 31), key="P_end")
with col_dt3:
    select_start = st.date_input("Selection start (OOS)", value=date(_today.year - 2, 1, 1), key="S_start")
with col_dt4:
    select_end = st.date_input("Selection end (OOS)", value=date(_today.year - 1, 12, 31), key="S_end")

st.caption("Step 1 fetches constituents only. Apply metadata filters first, then fetch OHLCV for the filtered tickers.")

# =====================
# Session state
# =====================
ss = st.session_state
if "idx_meta" not in ss:
    ss.idx_meta = pd.DataFrame()  # metadata only
if "idx_members" not in ss:
    ss.idx_members = pd.DataFrame()  # metadata + (optional) liquidity stats

# =====================
# Step 1: Fetch constituents (metadata only)
# =====================
col_btn1, col_btn2 = st.columns([1, 2])
with col_btn1:
    btn_fetch_meta = st.button("📄 Fetch constituents (metadata only)", use_container_width=True, key="btn_fetch_meta")
with col_btn2:
    st.write("")

if btn_fetch_meta:
    if not choices:
        st.warning("Select at least one index.")
    else:
        keys = [index_map[c] for c in choices]
        with st.spinner("Fetching index constituents…"):
            members = UNV.fetch_indexes(keys, force_refresh=force_refresh)
        if members.empty:
            st.error("No members fetched. Check your network or try again.")
        else:
            ss.idx_meta = members.copy()
            ss.idx_members = members.copy()  # start equal; will enrich after OHLCV
            st.success(f"Fetched {len(members)} symbols across {len(choices)} indexes.")

# =====================
# Metadata filters (no OHLCV needed)
# =====================
if not ss.idx_meta.empty:
    st.subheader("Filter by metadata (pre-bars)")
    meta = ss.idx_meta.copy()

    # sector/industry lists may be partly missing depending on wiki tables
    sectors = sorted([s for s in meta.get("sector", pd.Series(dtype=str)).dropna().unique().tolist() if s])
    industries = sorted([s for s in meta.get("industry", pd.Series(dtype=str)).dropna().unique().tolist() if s])

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sel_sectors = st.multiselect("Sectors", options=sectors, default=sectors, key="ms_sectors")
    with c2:
        sel_industries = st.multiselect("Industries", options=industries, key="ms_industries")
    with c3:
        sym_query = st.text_input("Symbol filter (substring)", value="", key="ti_symbol_filter")

    f = pd.Series(True, index=meta.index)
    if sectors:
        f &= meta["sector"].isin(sel_sectors)
    if sel_industries:
        f &= meta["industry"].isin(sel_industries)
    if sym_query.strip():
        s = sym_query.strip().upper()
        f &= meta["symbol"].str.contains(s, case=False, na=False)

    meta_filt = meta.loc[f].copy()
    st.write(f"Filtered (metadata-only): **{len(meta_filt)}** / {len(meta)} tickers")
    st.dataframe(meta_filt[[c for c in ["symbol", "name", "sector", "industry"] if c in meta_filt.columns]].reset_index(drop=True), use_container_width=True, height=300)

    # Optionally limit how many symbols to fetch bars for
    colK1, colK2 = st.columns([1, 1])
    with colK1:
        max_to_fetch = st.number_input("Max tickers to fetch bars", min_value=1, max_value=2000, value=min(200, len(meta_filt)), key="ni_max_to_fetch")
    with colK2:
        st.caption("Use this to control initial data load time. You can refine filters and fetch more later.")

    # =====================
    # Step 2: Fetch OHLCV for the filtered list (compute liquidity)
    # =====================
    if st.button("📥 Fetch OHLCV & compute liquidity for filtered", type="primary", use_container_width=True, key="btn_fetch_bars"):
        tickers = meta_filt["symbol"].head(int(max_to_fetch)).tolist()
        if not tickers:
            st.warning("No symbols to fetch.")
        else:
            prog = st.progress(0.0)
            rows = []
            misses = 0
            for i, sym in enumerate(tickers):
                ok = False
                try:
                    df_p = load_ohlcv_window(sym, priors_start.isoformat(), priors_end.isoformat())
                    df_s = load_ohlcv_window(sym, select_start.isoformat(), select_end.isoformat())
                    ok = (df_p is not None and not df_p.empty) and (df_s is not None and not df_s.empty)
                except Exception:
                    ok = False
                if ok:
                    try:
                        m_close = float(df_p["close"].median())
                        m_dvol = float((df_p["close"] * df_p["volume"]).median())
                    except Exception:
                        m_close, m_dvol = np.nan, np.nan
                    rows.append({
                        "symbol": sym,
                        "median_close_priors": m_close,
                        "median_dollar_vol_priors": m_dvol,
                    })
                else:
                    misses += 1
                if (i + 1) % 10 == 0 or (i + 1) == len(tickers):
                    prog.progress((i + 1) / max(1, len(tickers)))

            liq = pd.DataFrame(rows)
            enriched = meta.merge(liq, on="symbol", how="left")
            ss.idx_members = enriched

            # snapshot for provenance
            out_dir = Path("storage/universe")
            out_dir.mkdir(parents=True, exist_ok=True)
            keys = [index_map[c] for c in choices] if choices else ["custom"]
            snap_path = out_dir / f"members_enriched__{'_'.join(keys)}__{_today.isoformat()}.parquet"
            try:
                enriched.to_parquet(snap_path, index=False)
            except Exception:
                enriched.to_csv(snap_path.with_suffix(".csv"), index=False)

            st.success(f"Enriched {len(rows)} symbols with OHLCV. Misses: {misses}. Snapshot: {snap_path.name}")

# =====================
# Post-bars filters (need liquidity columns)
# =====================
if not ss.idx_members.empty and ("median_close_priors" in ss.idx_members.columns):
    st.subheader("Filter by liquidity (post-bars)")
    dfm = ss.idx_members.copy()

    cA, cB = st.columns(2)
    with cA:
        min_price = st.number_input("Min median close (priors)", min_value=0.0, value=5.0, step=0.5, key="ni_min_price")
    with cB:
        min_dvol = st.number_input("Min median $ volume (priors)", min_value=0.0, value=2e7, step=1e6, key="ni_min_dvol")

    f2 = pd.Series(True, index=dfm.index)
    f2 &= (dfm["median_close_priors"].fillna(0) >= float(min_price))
    f2 &= (dfm["median_dollar_vol_priors"].fillna(0) >= float(min_dvol))
    filtered = dfm.loc[f2].copy()

    st.write(f"After liquidity filters: **{len(filtered)}** / {len(dfm)} tickers remain.")
    st.dataframe(
        filtered[[
            "symbol",
            *[c for c in ["name", "sector"] if c in filtered.columns],
            "median_close_priors",
            "median_dollar_vol_priors",
        ]]
        .sort_values("median_dollar_vol_priors", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=340,
    )

    # =====================
    # Save to portfolio
    # =====================
    st.subheader("Save to portfolio")
    portfolios = list_portfolios()
    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.radio("Save mode", options=["Add to existing", "Create new"], horizontal=False, key="radio_save_mode")
    with col2:
        if mode == "Add to existing":
            target = st.selectbox("Select portfolio", options=portfolios or ["(none)"], key="sb_existing_port")
        else:
            target = st.text_input("New portfolio name", value="my_universe", key="ti_new_port_name")

    if st.button("💾 Save selection to portfolio", type="primary", use_container_width=True, key="btn_save_port"):
        syms = sorted(filtered["symbol"].unique().tolist())
        if not syms:
            st.warning("No symbols to save.")
        else:
            if mode == "Add to existing":
                if target in (None, "", "(none)"):
                    st.error("Choose an existing portfolio or switch to Create new.")
                else:
                    obj = load_portfolio(target) or {"name": target, "tickers": []}
                    old = set(obj.get("tickers", []))
                    new = sorted(old.union(syms))
                    obj["tickers"] = new
                    obj["updated"] = date.today().isoformat()
                    save_portfolio(target, obj)
                    st.success(f"Updated portfolio '{target}': {len(old)} → {len(new)} tickers.")
            else:
                if not target:
                    st.error("Enter a portfolio name.")
                else:
                    obj = {"name": target, "tickers": syms, "created": date.today().isoformat()}
                    save_portfolio(target, obj)
                    st.success(f"Created portfolio '{target}' with {len(syms)} tickers.")
else:
    if ss.idx_meta.empty:
        st.info("Use **Fetch constituents (metadata only)** to start. Then filter and fetch OHLCV for the subset you want.")
    else:
        st.info("You can fetch OHLCV for the filtered metadata set to enable liquidity filters.")