# pages/1_Portfolios.py
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import the universe module so we can access the SUPPORTED catalog (names + URLs)
from src.universe import indexes as UNV
from src.storage import list_portfolios, load_portfolio, save_portfolio

try:
    # prefer cached loader if present
    from src.data.cache import get_ohlcv_cached as load_ohlcv_window
except Exception:
    # fallback to direct alpaca loader
    from src.data.alpaca_data import load_ohlcv as load_ohlcv_window


st.set_page_config(page_title="Portfolios", layout="wide")
load_dotenv()

st.title("🗂️ Portfolios — Build from Indexes")

# ---- Controls ---------------------------------------------------------------

# Build a {display_name -> key} map from the SUPPORTED catalog
index_map = {meta["name"]: key for key, meta in UNV.SUPPORTED.items()}

choices = st.multiselect(
    "Select index collections",
    options=list(index_map.keys()),
    default=[name for name in ["S&P 500", "Nasdaq-100"] if name in index_map],
)
force_refresh = st.checkbox(
    "Force refresh index membership (fetch from Wikipedia)",
    value=False,
    help="If off, we use cached constituents under storage/universe/."
)

# Timeline defaults
today = date.today()
priors_start = st.date_input("Priors window start", value=date(today.year - 10, 1, 1))
priors_end = st.date_input("Priors window end", value=date(today.year - 3, 12, 31))
select_start = st.date_input("Selection window start (OOS)", value=date(today.year - 2, 1, 1))
select_end = st.date_input("Selection window end (OOS)", value=date(today.year - 1, 12, 31))

st.caption("Priors → learn robust bounds. Selection → rank names out-of-sample. You can tune & simulate later.")

# ---- Fetch button -----------------------------------------------------------
go = st.button("📥 Get & cache data for selected indexes", use_container_width=True)

# ---- Workspace state --------------------------------------------------------
if "idx_members" not in st.session_state:
    st.session_state.idx_members = pd.DataFrame()

if go and choices:
    keys = [index_map[c] for c in choices]
    with st.spinner("Fetching index constituents…"):
        members = UNV.fetch_indexes(keys, force_refresh=force_refresh)

    if members.empty:
        st.error("No members fetched. Check your network (Wikipedia) or try again.")
        st.stop()

    # Cache OHLCV for Priors + Selection windows
    uniq_syms = sorted(members["symbol"].unique().tolist())
    st.write(f"Found **{len(uniq_syms)}** unique tickers across {len(choices)} indexes.")
    prog = st.progress(0.0)
    rows = []
    misses = 0

    for i, sym in enumerate(uniq_syms):
        ok = False
        try:
            df_p = load_ohlcv_window(sym, priors_start.isoformat(), priors_end.isoformat())
            df_s = load_ohlcv_window(sym, select_start.isoformat(), select_end.isoformat())
            ok = (df_p is not None and not df_p.empty) and (df_s is not None and not df_s.empty)
        except Exception:
            ok = False

        if ok:
            # compute basic liquidity stats from priors window
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

        if (i + 1) % 10 == 0 or (i + 1) == len(uniq_syms):
            prog.progress((i + 1) / len(uniq_syms))

    liq = pd.DataFrame(rows)
    members = members.merge(liq, on="symbol", how="left")

    # Save a snapshot
    out_dir = Path("storage/universe")
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_path = out_dir / f"members__{'_'.join(keys)}__{today.isoformat()}.parquet"
    try:
        members.to_parquet(snap_path, index=False)
    except Exception:
        members.to_csv(snap_path.with_suffix(".csv"), index=False)

    st.session_state.idx_members = members
    st.success(f"Cached data for {len(rows)} symbols. Misses: {misses}. Snapshot saved to {snap_path.name}.")

# ---- If we have members, show filter panel ----------------------------------
if not st.session_state.idx_members.empty:
    members = st.session_state.idx_members.copy()
    st.subheader("Filter universe")

    # sector list (may be blank if fetch couldn't get sector)
    sectors = sorted([s for s in members.get("sector", pd.Series(dtype=str)).dropna().unique().tolist() if s])
    colA, colB, colC = st.columns(3)
    with colA:
        sel_sectors = st.multiselect("Sectors", options=sectors, default=sectors)
    with colB:
        min_price = st.number_input("Min median close (priors)", min_value=0.0, value=5.0, step=0.5)
    with colC:
        min_dvol = st.number_input(
            "Min median $ volume (priors)",
            min_value=0.0, value=2e7, step=1e6,
            help="Median of close*volume over the Priors window"
        )

    # apply filters
    filt = pd.Series(True, index=members.index)
    if sectors:
        filt &= members["sector"].isin(sel_sectors)
    filt &= (members["median_close_priors"].fillna(0) >= float(min_price))
    filt &= (members["median_dollar_vol_priors"].fillna(0) >= float(min_dvol))

    filtered = members.loc[filt].copy()
    st.write(f"After filters: **{len(filtered)}** / {len(members)} tickers remain.")

    st.dataframe(
        filtered[["symbol", "name", "sector", "median_close_priors", "median_dollar_vol_priors"]]
        .sort_values("median_dollar_vol_priors", ascending=False)
        .reset_index(drop=True),
        use_container_width=True, height=360
    )

    st.subheader("Save to portfolio")
    portfolios = list_portfolios()
    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.radio("Save mode", options=["Add to existing", "Create new"], horizontal=False)
    with col2:
        if mode == "Add to existing":
            target = st.selectbox("Select portfolio", options=portfolios or ["(none)"])
        else:
            target = st.text_input("New portfolio name", value="my_universe")

    if st.button("💾 Save selection to portfolio", type="primary", use_container_width=True):
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
    st.info("Select one or more indexes above and click **Get & cache data**.")