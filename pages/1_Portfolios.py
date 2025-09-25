# pages/1_Portfolios.py
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Universe: catalogs + fetch
from src.universe import indexes as UNV

# Storage API (new signatures)
from src.storage import (
    list_portfolios,
    load_portfolio,
    save_portfolio,        # save_portfolio(name, tickers, meta=None)
    append_to_portfolio,   # append_to_portfolio(name, tickers, meta_update=None)
)

# Unified, normalized OHLCV loader (auto provider selection handled inside)
from src.data.loader import get_ohlcv as load_ohlcv_window


# ────────────────────────────────────────────────────────────────────────────────
# Page setup
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolios", layout="wide")
load_dotenv()

import os

def _hydrate_env_from_secrets():
    """Populate expected Alpaca env vars from Streamlit secrets if present."""
    try:
        s = st.secrets  # type: ignore[attr-defined]
    except Exception:
        s = None
    if not s:
        return
    os.environ.setdefault("APCA_API_KEY_ID", s.get("ALPACA_API_KEY", ""))
    os.environ.setdefault("APCA_API_SECRET_KEY", s.get("ALPACA_SECRET_KEY", ""))
    if s.get("ALPACA_BASE_URL"):
        os.environ.setdefault("APCA_API_BASE_URL", s["ALPACA_BASE_URL"])
    if s.get("ALPACA_DATA_URL"):
        os.environ.setdefault("APCA_DATA_URL", s["ALPACA_DATA_URL"])
    os.environ.setdefault("APCA_FEED", s.get("ALPACA_FEED", "iex"))

_hydrate_env_from_secrets()

def _to_dt(d: date) -> datetime:
    # use UTC midnight for stable window bounds
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

st.title("🗂️ Portfolios — Universe → Filter → Fetch → Save")

st.caption(
    "Workflow: (1) Fetch index constituents (metadata only) → "
    "(2) (Optional) fetch OHLCV for a subset and compute liquidity → "
    "(3) Save the filtered tickers to a portfolio."
)

# ────────────────────────────────────────────────────────────────────────────────
# Controls: pick indexes and date windows
# ────────────────────────────────────────────────────────────────────────────────
# Build a {display_name -> key} map from the SUPPORTED catalog
index_map = {meta["name"]: key for key, meta in UNV.SUPPORTED.items()}

choices = st.multiselect(
    "Select index collections",
    options=list(index_map.keys()),
    default=[name for name in ("S&P 500", "Nasdaq-100") if name in index_map],
    key="pf_idx_choices",
)

force_refresh = st.checkbox(
    "Force refresh index membership (fetch from Wikipedia)",
    value=False,
    help="If off, cached constituents under storage/universe/ will be used if present.",
    key="pf_force_refresh",
)

# Priors + Selection windows
today = date.today()
c_dt1, c_dt2, c_dt3, c_dt4 = st.columns(4)
with c_dt1:
    priors_start = st.date_input("Priors window start", value=date(today.year - 10, 1, 1), key="pf_priors_start")
with c_dt2:
    priors_end = st.date_input("Priors window end", value=date(today.year - 3, 12, 31), key="pf_priors_end")
with c_dt3:
    select_start = st.date_input("Selection window start (OOS)", value=date(today.year - 2, 1, 1), key="pf_select_start")
with c_dt4:
    select_end = st.date_input("Selection window end (OOS)", value=date(today.year - 1, 12, 31), key="pf_select_end")

# ────────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("pf_idx_meta", pd.DataFrame())     # metadata only
ss.setdefault("pf_idx_members", pd.DataFrame())  # enriched with liquidity after bars fetch

# ────────────────────────────────────────────────────────────────────────────────
# Step 1: Fetch constituents (metadata only)
# ────────────────────────────────────────────────────────────────────────────────
c_btn1, c_btn2 = st.columns([1, 2])
with c_btn1:
    btn_fetch_meta = st.button("📄 Fetch constituents (metadata only)", use_container_width=True, key="pf_btn_fetch_meta")
with c_btn2:
    st.write("")

if btn_fetch_meta:
    if not choices:
        st.warning("Select at least one index.")
    else:
        keys = [index_map[c] for c in choices]
        with st.spinner("Fetching index constituents…"):
            members = UNV.fetch_indexes(keys, force_refresh=force_refresh)
        if members.empty:
            st.error("No members fetched. Check your network (Wikipedia) or try again.")
        else:
            ss.pf_idx_meta = members.copy()
            ss.pf_idx_members = members.copy()  # will be enriched after OHLCV
            st.success(f"Fetched {len(members)} symbols across {len(choices)} indexes.")

# ────────────────────────────────────────────────────────────────────────────────
# Metadata filters (no OHLCV needed)
# ────────────────────────────────────────────────────────────────────────────────
if not ss.pf_idx_meta.empty:
    st.subheader("Filter by metadata (pre-bars)")
    meta = ss.pf_idx_meta.copy()

    sectors = sorted([s for s in meta.get("sector", pd.Series(dtype=str)).dropna().unique().tolist() if s])
    industries = sorted([s for s in meta.get("industry", pd.Series(dtype=str)).dropna().unique().tolist() if s])

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sel_sectors = st.multiselect("Sectors", options=sectors, default=sectors, key="pf_ms_sectors")
    with c2:
        sel_industries = st.multiselect("Industries", options=industries, key="pf_ms_industries")
    with c3:
        sym_query = st.text_input("Symbol filter (substring)", value="", key="pf_ti_symbol_filter")

    f = pd.Series(True, index=meta.index)
    if sectors:
        f &= meta["sector"].isin(sel_sectors)
    if sel_industries:
        f &= meta["industry"].isin(sel_industries)
    if sym_query.strip():
        s = sym_query.strip().upper()
        f &= meta["symbol"].astype(str).str.contains(s, case=False, na=False)

    meta_filt = meta.loc[f].copy()
    st.write(f"Filtered (metadata-only): **{len(meta_filt)}** / {len(meta)} tickers")
    st.dataframe(
        meta_filt[[c for c in ["symbol", "name", "sector", "industry"] if c in meta_filt.columns]].reset_index(drop=True),
        use_container_width=True,
        height=300,
    )

    # Limit initial bars fetch for speed
    cK1, cK2 = st.columns([1, 1])
    with cK1:
        max_to_fetch = st.number_input(
            "Max tickers to fetch bars",
            min_value=1,
            max_value=max(1, len(meta_filt)),
            value=min(200, len(meta_filt)) if len(meta_filt) else 1,
            key="pf_ni_max_to_fetch",
        )
    with cK2:
        st.caption("Use this to control initial data load time. You can refine filters and fetch more later.")

    # ────────────────────────────────────────────────────────────────────────────
    # Step 2: Fetch OHLCV for filtered list and compute liquidity (priors window)
    # ────────────────────────────────────────────────────────────────────────────
    if st.button("📥 Fetch OHLCV & compute liquidity for filtered", type="primary", use_container_width=True, key="pf_btn_fetch_bars"):
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
                    df_p = load_ohlcv_window(sym, _to_dt(priors_start), _to_dt(priors_end))
                    df_s = load_ohlcv_window(sym, _to_dt(select_start), _to_dt(select_end))
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
            if "symbol" not in liq.columns:
                liq = pd.DataFrame(columns=["symbol", "median_close_priors", "median_dollar_vol_priors"])  # ensure merge key exists
            enriched = meta.merge(liq, on="symbol", how="left")
            ss.pf_idx_members = enriched

            # snapshot for provenance
            out_dir = Path("storage/universe")
            out_dir.mkdir(parents=True, exist_ok=True)
            keys = [index_map[c] for c in choices] if choices else ["custom"]
            snap_path = out_dir / f"members_enriched__{'_'.join(keys)}__{today.isoformat()}.parquet"
            try:
                enriched.to_parquet(snap_path, index=False)
            except Exception:
                enriched.to_csv(snap_path.with_suffix(".csv"), index=False)

            st.success(f"Enriched {len(rows)} symbols with OHLCV. Misses: {misses}. Snapshot: {snap_path.name}")

# ────────────────────────────────────────────────────────────────────────────────
# Step 2b: Post-bars liquidity filters (requires liquidity cols)
# ────────────────────────────────────────────────────────────────────────────────
if not ss.pf_idx_members.empty and ("median_close_priors" in ss.pf_idx_members.columns):
    st.subheader("Filter by liquidity (post-bars)")
    dfm = ss.pf_idx_members.copy()

    cA, cB = st.columns(2)
    with cA:
        min_price = st.number_input("Min median close (priors)", min_value=0.0, value=5.0, step=0.5, key="pf_min_price")
    with cB:
        min_dvol = st.number_input("Min median $ volume (priors)", min_value=0.0, value=2_000_000.0, step=250_000.0, key="pf_min_dvol")

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

    # ────────────────────────────────────────────────────────────────────────────
    # Step 3: Save to portfolio
    # ────────────────────────────────────────────────────────────────────────────
    st.subheader("Save to portfolio")

    portfolios = list_portfolios()
    cS1, cS2 = st.columns([1, 2])
    with cS1:
        save_mode = st.radio("Save mode", options=["Add to existing", "Create new"], horizontal=False, key="pf_save_mode")
    with cS2:
        if save_mode == "Add to existing":
            target_name = st.selectbox("Select portfolio", options=portfolios or ["(none)"], key="pf_existing_port")
        else:
            target_name = st.text_input("New portfolio name", value="my_universe", key="pf_new_port")

    # Build the list to save (upper-cased, de-duped)
    selected_syms = (
        filtered["symbol"]
        .astype(str)
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )

    # Provenance / meta to store alongside the portfolio
    meta_payload = {
        "source": "1_Portfolios",
        "indexes": [index_map[c] for c in choices] if choices else [],
        "filters": {
            "min_median_close_priors": float(min_price),
            "min_median_dollar_vol_priors": float(min_dvol),
        },
        "windows": {
            "priors": [priors_start.isoformat(), priors_end.isoformat()],
            "selection": [select_start.isoformat(), select_end.isoformat()],
        },
        "count": len(selected_syms),
        "saved_at": date.today().isoformat(),
    }

    if st.button("💾 Save selection to portfolio", type="primary", use_container_width=True, key="pf_btn_save"):
        if not selected_syms:
            st.warning("No symbols to save — adjust your filters.")
        else:
            if save_mode == "Add to existing":
                if target_name in (None, "", "(none)"):
                    st.error("Choose an existing portfolio or switch to Create new.")
                else:
                    append_to_portfolio(target_name, selected_syms, meta_update=meta_payload)
                    saved = load_portfolio(target_name) or {}
                    st.success(f"Updated portfolio '{target_name}' — now {len(saved.get('tickers', []))} tickers.")
                    st.write(saved)  # show what actually persisted
            else:
                if not target_name:
                    st.error("Enter a portfolio name.")
                else:
                    save_portfolio(target_name, selected_syms, meta=meta_payload)
                    saved = load_portfolio(target_name) or {}
                    st.success(f"Created portfolio '{target_name}' with {len(saved.get('tickers', []))} tickers.")
                    st.write(saved)  # show what actually persisted

else:
    if ss.pf_idx_meta.empty:
        st.info("Use **Fetch constituents (metadata only)** to start. Then filter and (optionally) fetch OHLCV to enable liquidity filters.")
    else:
        st.info("To enable liquidity filters, fetch OHLCV for the filtered metadata set.")