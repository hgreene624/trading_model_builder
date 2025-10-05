# pages/2_Strategy_Adapter.py
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.utils.holdout_chart import init_chart, on_generation_end, set_config  # package path fallback
from src.utils.training_logger import TrainingLogger
from src.data.portfolio_prefetch import intersection_range
from src.storage import append_to_portfolio, list_portfolios, load_portfolio, save_strategy_params

# --- Page chrome ---
st.set_page_config(page_title="Strategy Adapter", layout="wide")
st.title("🧪 Strategy Adapter")


# --- Helpers ---
def _utc_now():
    return datetime.now(timezone.utc)


def _safe_import(dotted: str):
    return import_module(dotted)


def _ss_get_dict(key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    if key not in st.session_state or not isinstance(st.session_state[key], dict):
        st.session_state[key] = dict(default)
    return st.session_state[key]


def _as_utc_timestamp(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        try:
            return ts.tz_localize("UTC")
        except Exception:
            return None
    try:
        return ts.tz_convert("UTC")
    except Exception:
        return None


def _extract_daily_shards(meta_entry: Any) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(meta_entry, dict):
        return {}

    # Common shapes: {"1D": {...}}, {"timeframes": {"1D": {...}}}, or {"AAPL": [...]}
    if "timeframes" in meta_entry and isinstance(meta_entry["timeframes"], dict):
        meta_entry = meta_entry["timeframes"]

    for key in ("1D", "1d", "DAILY", "daily"):
        if key in meta_entry and isinstance(meta_entry[key], dict):
            return meta_entry[key]

    # Fallback: assume already symbol → [shards]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sym, shards in meta_entry.items():
        if isinstance(shards, list):
            out[str(sym).strip().upper()] = [dict(rec) for rec in shards if isinstance(rec, dict)]
    return out


def _write_kv_table(d: Dict[str, Any], title: str = ""):
    if title:
        st.markdown(f"**{title}**")
    df = pd.DataFrame({"key": list(d.keys()), "value": [d[k] for k in d.keys()]})
    st.dataframe(df, width="stretch", height=min(360, 40 + 28 * len(df)))


def _portfolio_equity_curve(
        strategy_dotted: str,
        tickers: List[str],
        start,
        end,
        starting_equity: float,
        params: Dict[str, Any],
) -> pd.Series:
    """Simulate aggregated portfolio equity for the given params on [start, end)."""

    try:
        mod = _safe_import(strategy_dotted)
        run = getattr(mod, "run_strategy")
    except Exception:
        return pd.Series(dtype=float)

    curves: Dict[str, pd.Series] = {}
    for sym in tickers:
        try:
            result = run(sym, start, end, starting_equity, params)
        except Exception:
            continue
        eq = result.get("equity")
        if eq is None or len(eq) == 0:
            continue
        if isinstance(eq, pd.DataFrame):
            if "equity" in eq.columns:
                eq = eq["equity"]
            else:
                eq = eq.iloc[:, 0]
        elif not isinstance(eq, pd.Series):
            try:
                eq = pd.Series(eq)
            except Exception:
                continue
        eq = eq.dropna()
        if eq.empty:
            continue
        eq = eq[~eq.index.duplicated(keep="last")]
        try:
            if not isinstance(eq.index, pd.DatetimeIndex):
                eq.index = pd.to_datetime(eq.index, errors="coerce")
        except Exception:
            continue
        eq = eq[~eq.index.isna()].sort_index()
        if eq.empty:
            continue
        try:
            eq = eq.astype(float)
        except Exception:
            continue
        first_valid = None
        for val in eq.values:
            if pd.isna(val):
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            if abs(fv) > 1e-9:
                first_valid = fv
                break
        if first_valid is None:
            continue
        norm = (eq / first_valid).astype(float)
        curves[sym] = norm

    if not curves:
        return pd.Series(dtype=float)

    df = pd.DataFrame(curves).sort_index()
    df = df.ffill().dropna(how="all")
    if df.empty:
        return pd.Series(dtype=float)

    portfolio = df.mean(axis=1, skipna=True) * float(starting_equity)
    portfolio.name = "portfolio_equity"
    return portfolio


def _normalize_symbols(seq) -> list[str]:
    out: list[str] = []
    for x in (seq or []):
        s = None
        if isinstance(x, str):
            s = x
        elif isinstance(x, dict):
            s = x.get("symbol") or x.get("ticker") or x.get("Symbol") or x.get("Ticker")
        elif hasattr(x, "get"):
            try:
                s = x.get("symbol") or x.get("ticker")
            except Exception:
                s = None
        else:
            s = str(x)
        if not s:
            continue
        s = s.strip().upper()
        # Drop obvious headers/placeholders
        if s in {"SYMBOL", "SYMBOLS", "TICKER", "TICKERS", "NAME", "SECURITY", "COMPANY", "N/A", ""}:
            continue
        # Basic ticker sanity: letters/digits/.- up to 10 chars
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", s):
            continue
        out.append(s)
    # de-dup while preserving rough order
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


# --- Helper to filter params for strategy ---
def _filter_params_for_strategy(strategy_dotted: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys the current strategy accepts (e.g., ATRParams fields)."""
    try:
        mod = _safe_import(strategy_dotted)
        keys = []
        if hasattr(mod, "ATRParams"):
            try:
                keys = list(getattr(mod, "ATRParams").__annotations__.keys())
            except Exception:
                keys = []
        if not keys and hasattr(mod, "PARAMS_ALLOWED"):
            try:
                keys = list(getattr(mod, "PARAMS_ALLOWED"))
            except Exception:
                keys = []
        if not keys:
            # sensible fallback for current ATR breakout
            keys = [
                "breakout_n",
                "exit_n",
                "atr_n",
                "atr_multiple",
                "tp_multiple",
                "holding_period_limit",
                "allow_short",
                "entry_mode",
                "trend_ma",
                "dip_atr_from_high",
                "dip_lookback_high",
                "dip_rsi_max",
                "dip_confirm",
                "dip_cooldown_days",
            ]
        return {k: params[k] for k in keys if k in params}
    except Exception:
        # final fallback: safest minimal subset
        safe = [
            "breakout_n",
            "exit_n",
            "atr_n",
            "atr_multiple",
            "tp_multiple",
            "holding_period_limit",
            "entry_mode",
            "trend_ma",
            "dip_atr_from_high",
            "dip_lookback_high",
            "dip_rsi_max",
            "dip_confirm",
            "dip_cooldown_days",
        ]
        return {k: params[k] for k in safe if k in params}


# ---------- TOP-TO-BOTTOM CONFIGURATION FLOW ----------
per_symbol_ranges: Dict[str, Dict[str, Any]] = {}
data_shards_daily: Dict[str, List[Dict[str, Any]]] = {}
data_cache_root: str | None = None
coverage_start_ts: pd.Timestamp | None = None
coverage_end_ts: pd.Timestamp | None = None
coverage_total_days: int = 0
train_start_dt: datetime | None = None
train_end_dt: datetime | None = None
holdout_start_dt: datetime | None = None
holdout_end_dt: datetime | None = None
train_days: int = 0
holdout_days: int = 0
train_fraction_actual: float = 0.7
holdout_fraction_actual: float = 0.3

st.subheader("1️⃣ Portfolio & Data Setup")

# Portfolio selection (saved portfolios only for EA runs)
tickers: List[str] = []
try:
    portfolios = sorted(list_portfolios())
except Exception as e:
    st.warning(f"Could not list portfolios: {e}")
    portfolios = []

if not portfolios:
    st.error("No saved portfolios found. Create one on the Portfolios page first.")
    st.stop()

default_idx = portfolios.index("Default") if "Default" in portfolios else 0
port_name = st.selectbox(
    "Portfolio",
    options=portfolios,
    index=default_idx,
    help="EA runs over the selected portfolio's symbols.",
)

try:
    obj = load_portfolio(port_name)
    if isinstance(obj, dict):
        raw = obj.get("tickers") or obj.get("symbols") or obj.get("items") or obj.get("data") or []
    else:
        raw = obj
    tickers = _normalize_symbols(raw)
except Exception as e:
    st.error(f"Failed to load portfolio '{port_name}': {e}")
    st.stop()

if not tickers:
    st.warning("No tickers selected. Add tickers or choose a different portfolio.")
    st.stop()

st.info(
    f"Selected **{len(tickers)}** symbols: {', '.join(tickers[:12])}{'…' if len(tickers) > 12 else ''}"
)

portfolio_meta = obj.get("meta") if isinstance(obj, dict) else {}
per_symbol_ranges = portfolio_meta.get("per_symbol_ranges") or {}
data_cache_root = portfolio_meta.get("data_cache_root")
data_shards_daily = _extract_daily_shards(portfolio_meta.get("data_shards"))

coverage_start_iso, coverage_end_iso = intersection_range(per_symbol_ranges)
if (not coverage_start_iso or not coverage_end_iso) and isinstance(portfolio_meta.get("windows"), dict):
    priors = portfolio_meta.get("windows", {}).get("priors")
    if isinstance(priors, (list, tuple)) and len(priors) == 2:
        coverage_start_iso = coverage_start_iso or priors[0]
        coverage_end_iso = coverage_end_iso or priors[1]

coverage_start_ts = _as_utc_timestamp(coverage_start_iso)
coverage_end_ts = _as_utc_timestamp(coverage_end_iso)
coverage_note = None
if coverage_start_ts is None or coverage_end_ts is None or coverage_end_ts <= coverage_start_ts:
    coverage_note = "Portfolio metadata lacked overlapping coverage; defaulting to the last 365 days."
    coverage_end_ts = pd.Timestamp(_utc_now())
    coverage_start_ts = coverage_end_ts - pd.Timedelta(days=365)

coverage_total_days = max(1, int((coverage_end_ts - coverage_start_ts).days) + 1)
default_train_pct = int(st.session_state.get("adapter_train_pct") or 70)
default_train_pct = min(max(default_train_pct, 50), 95)

summary_cols = st.columns(4)
summary_cols[0].metric("Coverage start", coverage_start_ts.date().isoformat())
summary_cols[1].metric("Coverage end", coverage_end_ts.date().isoformat())
summary_cols[2].metric("Total days", coverage_total_days)
summary_cols[3].metric("Symbols", len(tickers))

st.caption(
    f"Common coverage across {len(tickers)} symbols spans {coverage_total_days} days."
)

if coverage_note:
    st.info(coverage_note)

shard_total_count = sum(len(v) for v in data_shards_daily.values())
if shard_total_count:
    shard_msg = f"Tracked cache shards: {shard_total_count} files across {len(data_shards_daily)} symbols."
    if data_cache_root:
        shard_msg += f" Root: {data_cache_root}"
    st.caption(shard_msg)

train_pct_ui = st.slider(
    "Training share (%)",
    min_value=50,
    max_value=95,
    value=int(default_train_pct),
    step=5,
    help="Percent of the available coverage to dedicate to training. The remainder is reserved for holdout/testing.",
)
st.session_state["adapter_train_pct"] = int(train_pct_ui)
train_fraction = float(train_pct_ui) / 100.0

max_train_days = max(1, coverage_total_days - 1) if coverage_total_days > 1 else 1
proposed_train_days = int(round(coverage_total_days * train_fraction))
train_days = min(max_train_days, max(1, proposed_train_days))
holdout_days = coverage_total_days - train_days
if coverage_total_days > 1 and holdout_days < 1:
    holdout_days = 1
    train_days = coverage_total_days - holdout_days

train_start_ts = coverage_start_ts
train_days = max(1, min(train_days, coverage_total_days))
train_end_candidate = train_start_ts + pd.Timedelta(days=train_days - 1)
if train_end_candidate > coverage_end_ts:
    train_end_candidate = coverage_end_ts

holdout_start_ts = min(coverage_end_ts, train_end_candidate + pd.Timedelta(days=1))
train_end_from_holdout_ts = holdout_start_ts - pd.Timedelta(days=1)
if train_end_from_holdout_ts < train_start_ts:
    train_end_from_holdout_ts = train_start_ts
train_end_ts = min(train_end_candidate, train_end_from_holdout_ts)

holdout_start_ts = min(coverage_end_ts, train_end_ts + pd.Timedelta(days=1))

train_days = max(1, int((train_end_ts - train_start_ts).days) + 1)
holdout_days = max(0, int((coverage_end_ts - holdout_start_ts).days) + 1)

train_start_dt = train_start_ts.to_pydatetime()
train_end_dt = train_end_ts.to_pydatetime()
holdout_start_dt = holdout_start_ts.to_pydatetime()
holdout_end_dt = coverage_end_ts.to_pydatetime()

train_fraction_actual = train_days / coverage_total_days if coverage_total_days else 1.0
holdout_fraction_actual = holdout_days / coverage_total_days if coverage_total_days else 0.0

timing_cols = st.columns(3)
timing_cols[0].metric("Train window", f"{train_start_dt.date().isoformat()} → {train_end_dt.date().isoformat()}")
timing_cols[1].metric("Train days", f"{train_days} ({train_fraction_actual * 100:.1f}%)")
timing_cols[2].metric("Holdout days", f"{holdout_days} ({holdout_fraction_actual * 100:.1f}%)")

st.divider()
st.subheader("2️⃣ Strategy & Search Controls")

strategy_dotted = st.selectbox(
    "Strategy",
    ["src.models.atr_breakout"],
    index=0,
    help="Select the strategy module to adapt.",
)

base = _ss_get_dict(
    "adapter_base_params",
    {
        "breakout_n": 70,
        "exit_n": 16,
        "atr_n": 8,
        "atr_multiple": 2.20,
        "tp_multiple": 1.78,
        "holding_period_limit": 20,
        "risk_per_trade": 0.005,
        "use_trend_filter": False,
        "sma_fast": 20,
        "sma_slow": 50,
        "sma_long": 200,
        "long_slope_len": 20,
        "cost_bps": 1.0,
        "execution": "close",
        "entry_mode": "breakout",
        "trend_ma": 200,
        "dip_atr_from_high": 2.0,
        "dip_lookback_high": 60,
        "dip_rsi_max": 55.0,
        "dip_confirm": False,
        "dip_cooldown_days": 5,
    },
)

for _k, _v in [
    ("entry_mode", "breakout"),
    ("trend_ma", 200),
    ("dip_atr_from_high", 2.0),
    ("dip_lookback_high", 60),
    ("dip_rsi_max", 55.0),
    ("dip_confirm", False),
    ("dip_cooldown_days", 5),
]:
    base.setdefault(_k, _v)

ea_cfg = _ss_get_dict(
    "ea_cfg",
    {
        "generations": 12,
        "pop_size": 100,
        "min_trades": 12,
        "n_jobs": max(1, min(8, (os.cpu_count() or 2) - 1)),
        "breakout_n_min": 8,
        "breakout_n_max": 80,
        "exit_n_min": 4,
        "exit_n_max": 40,
        "atr_n_min": 7,
        "atr_n_max": 35,
        "atr_multiple_min": 0.8,
        "atr_multiple_max": 4.0,
        "tp_multiple_min": 0.8,
        "tp_multiple_max": 4.0,
        "hold_min": 5,
        "hold_max": 60,
        "trend_ma_min": int(base.get("trend_ma", 200)),
        "trend_ma_max": int(base.get("trend_ma", 200)),
        "dip_atr_from_high_min": float(base.get("dip_atr_from_high", 2.0)),
        "dip_atr_from_high_max": float(base.get("dip_atr_from_high", 2.0)),
        "dip_lookback_high_min": int(base.get("dip_lookback_high", 60)),
        "dip_lookback_high_max": int(base.get("dip_lookback_high", 60)),
        "dip_rsi_max_min": float(base.get("dip_rsi_max", 55.0)),
        "dip_rsi_max_max": float(base.get("dip_rsi_max", 55.0)),
        "dip_confirm_min": int(bool(base.get("dip_confirm", False))),
        "dip_confirm_max": int(bool(base.get("dip_confirm", False))),
        "dip_cooldown_min": int(base.get("dip_cooldown_days", 5)),
        "dip_cooldown_max": int(base.get("dip_cooldown_days", 5)),
    },
)

for _k, _v in [
    ("trend_ma_min", int(base.get("trend_ma", 200))),
    ("trend_ma_max", int(base.get("trend_ma", 200))),
    ("dip_atr_from_high_min", float(base.get("dip_atr_from_high", 2.0))),
    ("dip_atr_from_high_max", float(base.get("dip_atr_from_high", 2.0))),
    ("dip_lookback_high_min", int(base.get("dip_lookback_high", 60))),
    ("dip_lookback_high_max", int(base.get("dip_lookback_high", 60))),
    ("dip_rsi_max_min", float(base.get("dip_rsi_max", 55.0))),
    ("dip_rsi_max_max", float(base.get("dip_rsi_max", 55.0))),
    ("dip_confirm_min", int(bool(base.get("dip_confirm", False)))),
    ("dip_confirm_max", int(bool(base.get("dip_confirm", False)))),
    ("dip_cooldown_min", int(base.get("dip_cooldown_days", 5))),
    ("dip_cooldown_max", int(base.get("dip_cooldown_days", 5))),
]:
    ea_cfg.setdefault(_k, _v)

with st.expander("Evolutionary search (EA) controls", expanded=True):
    st.markdown("**Search behaviour**")
    search_cols = st.columns(3)
    with search_cols[0]:
        ea_cfg["generations"] = st.number_input("Generations", 1, 200, int(ea_cfg["generations"]), 1)
    with search_cols[1]:
        ea_cfg["pop_size"] = st.number_input("Population", 2, 400, int(ea_cfg["pop_size"]), 1)
    with search_cols[2]:
        ea_cfg["min_trades"] = st.number_input("Min trades (gate)", 0, 200, int(ea_cfg["min_trades"]), 1)

    st.markdown("**Parallelism**")
    ea_cfg["n_jobs"] = st.number_input(
        "Jobs (EA)",
        1,
        max(1, (os.cpu_count() or 2)),
        int(ea_cfg["n_jobs"]),
        1,
        help="Worker processes dedicated to the evolutionary search.",
    )

    st.markdown("**Parameter bounds**")
    bounds_cols = st.columns(3)
    with bounds_cols[0]:
        bnm_lo = st.number_input("breakout_n min", 1, 400, int(ea_cfg["breakout_n_min"]), 1)
        bnm_hi = st.number_input("breakout_n max", bnm_lo, 400, int(ea_cfg["breakout_n_max"]), 1)
        enm_lo = st.number_input("exit_n min", 1, 400, int(ea_cfg["exit_n_min"]), 1)
        enm_hi = st.number_input("exit_n max", enm_lo, 400, int(ea_cfg["exit_n_max"]), 1)
    with bounds_cols[1]:
        atm_lo = st.number_input("atr_n min", 1, 200, int(ea_cfg["atr_n_min"]), 1)
        atm_hi = st.number_input("atr_n max", atm_lo, 200, int(ea_cfg["atr_n_max"]), 1)
        atm_mul_lo = st.number_input("atr_multiple min", 0.1, 20.0, float(ea_cfg["atr_multiple_min"]), 0.1)
        atm_mul_hi = st.number_input("atr_multiple max", atm_mul_lo, 20.0, float(ea_cfg["atr_multiple_max"]), 0.1)
    with bounds_cols[2]:
        tpm_lo = st.number_input("tp_multiple min", 0.1, 20.0, float(ea_cfg["tp_multiple_min"]), 0.1)
        tpm_hi = st.number_input("tp_multiple max", tpm_lo, 20.0, float(ea_cfg["tp_multiple_max"]), 0.1)
        hold_lo = st.number_input("hold min", 1, 600, int(ea_cfg["hold_min"]), 1)
        hold_hi = st.number_input("hold max", hold_lo, 600, int(ea_cfg["hold_max"]), 1)

    st.markdown("**Dip parameter bounds**")
    dip_bounds = st.columns(3)
    with dip_bounds[0]:
        trend_lo = st.number_input("trend_ma min", 20, 600, int(ea_cfg["trend_ma_min"]), 1)
        trend_hi = st.number_input("trend_ma max", trend_lo, 600, int(ea_cfg["trend_ma_max"]), 1)
        dlh_lo = st.number_input("dip_lookback_high min", 5, 600, int(ea_cfg["dip_lookback_high_min"]), 1)
        dlh_hi = st.number_input("dip_lookback_high max", dlh_lo, 600, int(ea_cfg["dip_lookback_high_max"]), 1)
    with dip_bounds[1]:
        dah_lo = st.number_input("dip_atr_from_high min", 0.0, 20.0, float(ea_cfg["dip_atr_from_high_min"]), 0.1)
        dah_hi = st.number_input("dip_atr_from_high max", dah_lo, 20.0, float(ea_cfg["dip_atr_from_high_max"]), 0.1)
        drs_lo = st.number_input("dip_rsi_max min", 0.0, 100.0, float(ea_cfg["dip_rsi_max_min"]), 1.0)
        drs_hi = st.number_input("dip_rsi_max max", drs_lo, 100.0, float(ea_cfg["dip_rsi_max_max"]), 1.0)
    with dip_bounds[2]:
        dcf_lo = st.number_input("dip_confirm min", 0, 1, int(ea_cfg["dip_confirm_min"]), 1)
        dcf_hi = st.number_input("dip_confirm max", dcf_lo, 1, int(ea_cfg["dip_confirm_max"]), 1)
        dcd_lo = st.number_input("dip_cooldown_days min", 0, 240, int(ea_cfg["dip_cooldown_min"]), 1)
        dcd_hi = st.number_input("dip_cooldown_days max", dcd_lo, 240, int(ea_cfg["dip_cooldown_max"]), 1)

    ea_cfg["breakout_n_min"], ea_cfg["breakout_n_max"] = int(bnm_lo), int(bnm_hi)
    ea_cfg["exit_n_min"], ea_cfg["exit_n_max"] = int(enm_lo), int(enm_hi)
    ea_cfg["atr_n_min"], ea_cfg["atr_n_max"] = int(atm_lo), int(atm_hi)
    ea_cfg["atr_multiple_min"], ea_cfg["atr_multiple_max"] = float(atm_mul_lo), float(atm_mul_hi)
    ea_cfg["tp_multiple_min"], ea_cfg["tp_multiple_max"] = float(tpm_lo), float(tpm_hi)
    ea_cfg["hold_min"], ea_cfg["hold_max"] = int(hold_lo), int(hold_hi)
    ea_cfg["trend_ma_min"], ea_cfg["trend_ma_max"] = int(trend_lo), int(trend_hi)
    ea_cfg["dip_atr_from_high_min"], ea_cfg["dip_atr_from_high_max"] = float(dah_lo), float(dah_hi)
    ea_cfg["dip_lookback_high_min"], ea_cfg["dip_lookback_high_max"] = int(dlh_lo), int(dlh_hi)
    ea_cfg["dip_rsi_max_min"], ea_cfg["dip_rsi_max_max"] = float(drs_lo), float(drs_hi)
    ea_cfg["dip_confirm_min"], ea_cfg["dip_confirm_max"] = int(dcf_lo), int(dcf_hi)
    ea_cfg["dip_cooldown_min"], ea_cfg["dip_cooldown_max"] = int(dcd_lo), int(dcd_hi)

with st.expander("Strategy parameter defaults (optional)", expanded=False):
    entry_modes = ["breakout", "dip"]
    entry_mode_clean = str(base.get("entry_mode", "breakout")).strip().lower()
    if entry_mode_clean not in entry_modes:
        entry_mode_clean = "breakout"
    base["entry_mode"] = st.radio(
        "Entry mode",
        entry_modes,
        index=entry_modes.index(entry_mode_clean),
        horizontal=True,
        help="Choose the entry logic to seed training runs.",
    )

    base_cols = st.columns(2)
    with base_cols[0]:
        base["breakout_n"] = st.number_input(
            "breakout_n",
            5,
            300,
            base["breakout_n"],
            1,
            help="Lookback used for breakout entry signal.",
        )
        base["exit_n"] = st.number_input(
            "exit_n",
            4,
            300,
            base["exit_n"],
            1,
            help="Lookback used for breakout exit/stop logic.",
        )
        base["atr_n"] = st.number_input(
            "atr_n",
            5,
            60,
            base["atr_n"],
            1,
            help="ATR window length.",
        )
        base["atr_multiple"] = st.number_input(
            "atr_multiple",
            0.5,
            10.0,
            float(base["atr_multiple"]),
            0.1,
            help="ATR multiple for stop distance.",
        )
        base["tp_multiple"] = st.number_input(
            "tp_multiple",
            0.2,
            10.0,
            float(base["tp_multiple"]),
            0.1,
            help="Take-profit multiple (vs ATR or entry logic).",
        )
        base["holding_period_limit"] = st.number_input(
            "holding_period_limit",
            1,
            400,
            base["holding_period_limit"],
            1,
            help="Max bars to hold a position.",
        )
    with base_cols[1]:
        base["risk_per_trade"] = st.number_input(
            "risk_per_trade",
            0.0005,
            0.05,
            float(base["risk_per_trade"]),
            0.0005,
            format="%.4f",
            help="Fraction of equity risked per trade.",
        )
        base["use_trend_filter"] = st.checkbox(
            "use_trend_filter",
            value=bool(base["use_trend_filter"]),
            help="Optional trend filter gate.",
        )
        base["sma_fast"] = st.number_input(
            "sma_fast",
            5,
            100,
            base["sma_fast"],
            1,
            help="Fast MA length (if trend filter used).",
        )
        base["sma_slow"] = st.number_input(
            "sma_slow",
            10,
            200,
            base["sma_slow"],
            1,
            help="Slow MA length (if trend filter used).",
        )
        base["sma_long"] = st.number_input(
            "sma_long",
            100,
            400,
            base["sma_long"],
            1,
            help="Long MA length (if trend filter used).",
        )
        base["long_slope_len"] = st.number_input(
            "long_slope_len",
            5,
            60,
            base["long_slope_len"],
            1,
            help="Slope window for long MA trend check.",
        )
        base["cost_bps"] = st.number_input(
            "cost_bps",
            0.0,
            20.0,
            float(base["cost_bps"]),
            0.1,
            help="Per-trade cost (basis points).",
        )
        base["execution"] = st.selectbox(
            "Execution",
            ["close"],
            index=0,
            help="Execution price proxy used in backtest.",
        )

    dip_active = str(base.get("entry_mode", "breakout")).strip().lower() == "dip"
    with st.expander("Dip Settings", expanded=dip_active):
        if dip_active:
            base["trend_ma"] = st.number_input(
                "trend_ma",
                20,
                400,
                int(base.get("trend_ma", 200)),
                1,
                help="Minimum trend moving average length to qualify dip entries.",
            )
            base["dip_atr_from_high"] = st.number_input(
                "dip_atr_from_high",
                0.0,
                10.0,
                float(base.get("dip_atr_from_high", 2.0)),
                0.1,
                help="ATR distance from recent high to trigger a dip setup.",
            )
            base["dip_lookback_high"] = st.number_input(
                "dip_lookback_high",
                5,
                400,
                int(base.get("dip_lookback_high", 60)),
                1,
                help="Lookback window for defining the reference high.",
            )
            base["dip_rsi_max"] = st.number_input(
                "dip_rsi_max",
                0.0,
                100.0,
                float(base.get("dip_rsi_max", 55.0)),
                1.0,
                help="Upper RSI bound required to qualify a dip entry.",
            )
            base["dip_confirm"] = st.checkbox(
                "dip_confirm",
                value=bool(base.get("dip_confirm", False)),
                help="Require confirmation (e.g., reversal bar) before dip entry.",
            )
            base["dip_cooldown_days"] = st.number_input(
                "dip_cooldown_days",
                0,
                60,
                int(base.get("dip_cooldown_days", 5)),
                1,
                help="Bars to wait between dip entries for the same symbol.",
            )
        else:
            st.info("Switch entry mode to 'Dip' to configure dip-specific defaults.")

with st.expander("Advanced run settings", expanded=False):
    folds = st.number_input("CV folds", 2, 10, 4, 1, help="Cross-validation splits for the base model trainer.")
    equity = st.number_input("Starting equity ($)", 1000.0, 1_000_000.0, 10_000.0, 100.0,
                             help="Starting equity for per-symbol runs.")
    min_trades = st.number_input("Min trades (valid)", 0, 200, 2, 1,
                                 help="Minimum total trades needed for a run to be considered valid.")
    max_procs = os.cpu_count() or 8
    n_jobs = st.slider("Jobs (processes)", 1, max(1, max_procs - 1), min(8, max(1, max_procs - 1)))

st.caption("Tip: If Alpaca SIP limits or YF rate limits bite, reduce Jobs to 1–4.")

st.divider()
st.subheader("3️⃣ Train & Monitor")

run_btn = st.button(
    "🚀 Train (portfolio)",
    type="primary",
    help="Runs the portfolio-level base trainer with the configuration above.",
    use_container_width=True,
)

st.markdown("### Live evaluations")
st.caption("Recent EA evaluations (rolling window)")
eval_table_placeholder = st.empty()

st.markdown("### Best candidate so far")
best_score_col, best_params_col = st.columns([1, 1.8], gap="large")
with best_score_col:
    best_score_placeholder = st.empty()
with best_params_col:
    best_params_placeholder = st.empty()

st.markdown("### Holdout equity (outside training window)")
holdout_chart_placeholder = st.empty()
holdout_status_placeholder = st.empty()

st.markdown("### Generation summary")
gen_summary_placeholder = st.empty()

# --- Pull any previous run artifacts back into the UI ---
live_rows_state = st.session_state.get("adapter_live_rows") or []
if live_rows_state:
    eval_table_placeholder.dataframe(pd.DataFrame(live_rows_state), width="stretch", height=380)
else:
    eval_table_placeholder.info("No evaluations yet. Run the EA to populate this table.")

best_tracker_state = st.session_state.get("adapter_best_tracker") or {}
best_score_val = best_tracker_state.get("score")
best_delta_val = best_tracker_state.get("delta")
if isinstance(best_score_val, (int, float)) and best_score_val not in (float("-inf"), float("inf")):
    best_score_placeholder.metric(
        "Best score",
        f"{best_score_val:.3f}",
        delta=None if best_delta_val is None else f"{best_delta_val:+.3f}",
    )
else:
    best_score_placeholder.metric("Best score", "—")

best_params_state = best_tracker_state.get("params") or {}
if best_params_state:
    df_params_state = pd.DataFrame(
        {"param": list(best_params_state.keys()), "value": [best_params_state[k] for k in best_params_state.keys()]}
    )
    best_params_placeholder.dataframe(df_params_state.set_index("param"), width="stretch", height=220)
else:
    best_params_placeholder.info("Waiting for evaluations…")

holdout_history_state = st.session_state.get("adapter_holdout_history") or []
# (chart rendered by holdout_chart helper)
# ---------- Results hydration & status ----------
holdout_status_state = st.session_state.get("adapter_holdout_status") or ("info",
                                                                          "Holdout equity will appear when a best candidate is found.")
status_kind, status_msg = holdout_status_state
if status_kind == "success":
    holdout_status_placeholder.success(status_msg)
elif status_kind == "warning":
    holdout_status_placeholder.warning(status_msg)
elif status_kind == "error":
    holdout_status_placeholder.error(status_msg)
else:
    holdout_status_placeholder.info(status_msg)

gen_history_state = st.session_state.get("adapter_gen_history") or []
if gen_history_state:
    gen_summary_placeholder.dataframe(pd.DataFrame(gen_history_state[-12:]), width="stretch", height=180)
else:
    gen_summary_placeholder.info("No generations have completed yet.")

# ------------------------ EA RUN (refactored) ------------------------
if run_btn:
    if not tickers:
        st.error("This portfolio has no tickers.")
        st.stop()

    # Resolve modules used during EA
    try:
        evo = _safe_import("src.optimization.evolutionary")
        progmod = _safe_import("src.utils.progress")  # optional UI progress sink
    except Exception as e:
        st.error(f"Import error: {e}")
        st.stop()

    prog = st.progress(0.0, text="Preparing EA search…")
    status = st.empty()

    # Reset per-run UI/session state
    live_rows: list[dict[str, Any]] = []
    gen_history: list[dict[str, Any]] = []
    best_tracker: dict[str, Any] = {"score": float("-inf"), "params": {}, "delta": None}

    st.session_state["adapter_live_rows"] = []
    st.session_state["adapter_gen_history"] = []
    st.session_state["adapter_best_tracker"] = dict(best_tracker)

    eval_table_placeholder.empty()
    best_score_placeholder.metric("Best score", "—")
    best_params_placeholder.info("Waiting for evaluations…")

    if not all([train_start_dt, train_end_dt, holdout_start_dt, holdout_end_dt]):
        st.error("Unable to resolve training/holdout windows from portfolio metadata. Refresh the page or rebuild the portfolio.")
        st.stop()

    def _ensure_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    start = _ensure_utc(train_start_dt)
    holdout_start = _ensure_utc(holdout_start_dt)
    holdout_end = _ensure_utc(holdout_end_dt)
    end = min(_ensure_utc(train_end_dt), holdout_start - timedelta(days=1))
    if end < start:
        end = start

    st.session_state["adapter_holdout_status"] = (
        "info",
        f"Testing on holdout window {holdout_start.date().isoformat()} → {holdout_end.date().isoformat()}",
    )

    def _hc_engine(params, data, starting_equity):
        series = _portfolio_equity_curve(
            strategy_dotted,
            tickers,
            holdout_start,
            holdout_end,
            float(starting_equity),
            params,
        )
        return pd.DataFrame({"equity": series})

    init_chart(
        placeholder=holdout_chart_placeholder,
        starting_equity=float(equity),
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        loader_fn=lambda **kwargs: {},  # unused by _hc_engine
        engine_fn=_hc_engine,
        symbols=tickers,
        max_curves=8,
    )
    st.session_state["_hc_last_score"] = None

    # ---- Build EA param space from UI bounds ----
    cfg = ea_cfg

    def _clamp_int(lo, hi):
        return (int(lo), int(max(lo, hi)))

    def _clamp_float(lo, hi):
        return (float(lo), float(max(float(lo), float(hi))))

    param_space = {
        "breakout_n": _clamp_int(cfg["breakout_n_min"], cfg["breakout_n_max"]),
        "exit_n": _clamp_int(cfg["exit_n_min"], cfg["exit_n_max"]),
        "atr_n": _clamp_int(cfg["atr_n_min"], cfg["atr_n_max"]),
        "atr_multiple": _clamp_float(cfg["atr_multiple_min"], cfg["atr_multiple_max"]),
        "tp_multiple": _clamp_float(cfg["tp_multiple_min"], cfg["tp_multiple_max"]),
        "holding_period_limit": _clamp_int(cfg["hold_min"], cfg["hold_max"]),
    }

    if str(base.get("entry_mode", "")).strip().lower() == "dip":
        param_space.update(
            {
                "trend_ma": _clamp_int(cfg["trend_ma_min"], cfg["trend_ma_max"]),
                "dip_atr_from_high": _clamp_float(
                    cfg["dip_atr_from_high_min"], cfg["dip_atr_from_high_max"]
                ),
                "dip_lookback_high": _clamp_int(
                    cfg["dip_lookback_high_min"], cfg["dip_lookback_high_max"]
                ),
                "dip_rsi_max": _clamp_float(cfg["dip_rsi_max_min"], cfg["dip_rsi_max_max"]),
                "dip_confirm": _clamp_int(cfg["dip_confirm_min"], cfg["dip_confirm_max"]),
                "dip_cooldown_days": _clamp_int(
                    cfg["dip_cooldown_min"], cfg["dip_cooldown_max"]
                ),
            }
        )

    # Optional richer progress sink
    ui_cb = getattr(progmod, "ui_progress", lambda *_args, **_kw: (lambda *_a, **_k: None))(st)
    # Track best-of-generation and last plotted score
    gen_best = {"score": float("-inf"), "params": {}}
    st.session_state.setdefault("_hc_last_score", None)

    def _cb(evt, ctx):
        # evt: "generation_start", "individual_evaluated", "generation_end", "done"
        try:
            ui_cb(evt, ctx)
        except Exception:
            pass
        if evt == "generation_start":
            gen = int(ctx.get("gen", 0))
            # Reset generation tracker for best-of-generation
            gen_best["score"] = float("-inf")
            gen_best["params"] = {}
            status.info(f"Generation {gen + 1} starting (population={ctx.get('pop_size', 'n/a')})")
            prog.progress(
                min(0.9, 0.1 + (gen / max(1, cfg['generations'])) * 0.8),
                text=f"EA generation {gen + 1}/{cfg['generations']}…",
            )
        elif evt == "individual_evaluated":
            metrics = ctx.get("metrics", {}) or {}
            row = {
                "gen": ctx.get("gen"),
                "idx": ctx.get("idx"),
                "score": ctx.get("score"),
                "trades": int(metrics.get("trades", 0) or 0),
                "cagr": metrics.get("cagr"),
                "calmar": metrics.get("calmar"),
                "sharpe": metrics.get("sharpe"),
            }
            live_rows.append(row)
            live_rows[:] = live_rows[-60:]
            eval_table_placeholder.dataframe(pd.DataFrame(live_rows), width="stretch", height=380)
            st.session_state["adapter_live_rows"] = list(live_rows)

            score = ctx.get("score")
            if isinstance(score, (int, float)):
                # Track best within this generation
                try:
                    cur_best = gen_best.get("score")
                except Exception:
                    cur_best = float("-inf")
                if cur_best in (None, float("-inf")) or float(score) > float(cur_best):
                    gen_best["score"] = float(score)
                    gen_best["params"] = dict(ctx.get("params") or {})
                prev = best_tracker.get("score")
                if prev in (None, float("-inf")) or score > prev:
                    delta = None if prev in (None, float("-inf")) else float(score) - float(prev)
                    best_tracker.update({"score": float(score), "params": dict(ctx.get("params") or {}), "delta": delta})
                    best_score_placeholder.metric(
                        "Best score",
                        f"{best_tracker['score']:.3f}",
                        delta=None if delta is None else f"{delta:+.3f}",
                    )
                    if best_tracker["params"]:
                        dfp = pd.DataFrame({"param": list(best_tracker["params"].keys()),
                                            "value": [best_tracker["params"][k] for k in best_tracker["params"].keys()]})
                        best_params_placeholder.dataframe(dfp.set_index("param"), width="stretch", height=220)
                    st.session_state["adapter_best_tracker"] = dict(best_tracker)

        elif evt == "generation_end":
            gen_history.append({
                "generation": ctx.get("gen"),
                "best_score": ctx.get("best_score"),
                "avg_score": ctx.get("avg_score"),
                "avg_trades": ctx.get("avg_trades"),
                "no_trades_pct": ctx.get("pct_no_trades"),
            })
            gen_summary_placeholder.dataframe(pd.DataFrame(gen_history[-12:]), width="stretch", height=180)
            st.session_state["adapter_gen_history"] = list(gen_history)
            # Determine gen-best from ctx first, then fallback to accumulator
            best_score_gen = ctx.get("best_score")
            if not isinstance(best_score_gen, (int, float)):
                best_score_gen = gen_best.get("score", float("-inf"))
            best_params_gen = ctx.get("best_params") or gen_best.get("params") or dict(ctx.get("params") or {})
            last_plotted = st.session_state.get("_hc_last_score")

            # Always push Gen 0; otherwise only if score improves
            should_push = (last_plotted is None) or (isinstance(best_score_gen, (int, float)) and best_score_gen > last_plotted)
            if should_push and isinstance(best_score_gen, (int, float)) and best_params_gen:
                try:
                    on_generation_end(int(ctx.get("gen", 0)), float(best_score_gen), dict(best_params_gen))
                    st.session_state["_hc_last_score"] = float(best_score_gen)
                except Exception:
                    pass

            # Progress update
            gen_idx = int(ctx.get("gen", 0))
            total_gens = int(cfg.get("generations", 1)) or 1
            p = min(0.95, 0.1 + ((gen_idx + 1) / total_gens) * 0.8)
            label = (
                f"EA gen {gen_idx} done; best={best_score_gen:.3f}"
                if isinstance(best_score_gen, (int, float)) else
                "EA evolving…"
            )
            prog.progress(p, text=label)

        elif evt == "done":
            elapsed = ctx.get("elapsed_sec")
            if isinstance(elapsed, (int, float)):
                status.success(f"EA completed in {elapsed:.1f}s")
                prog.progress(1.0, text="EA completed")

    # EA logging: timestamped JSONL under storage/logs/ea
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("storage", "logs", "ea")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{ts}_ea.jsonl")

    coverage_start_iso = coverage_start_ts.date().isoformat() if coverage_start_ts is not None else None
    coverage_end_iso = coverage_end_ts.date().isoformat() if coverage_end_ts is not None else None

    # Log the holdout (simulation) window so tools can detect train→test boundary
    try:
        TrainingLogger(log_file).log(
            "holdout_meta",
            {
                "gen": 0,
                "portfolio": port_name,
                "tickers": list(tickers),
                "holdout_start": str(holdout_start),
                "holdout_end": str(holdout_end),
                "starting_equity": float(equity),
                "train_start": str(start),
                "train_end": str(end),
                "train_days": int(train_days),
                "test_days": int(holdout_days),
                "train_fraction": float(train_fraction_actual),
                "train_share_pct": round(train_fraction_actual * 100.0, 3),
                "test_fraction": float(holdout_fraction_actual),
                "test_share_pct": round(holdout_fraction_actual * 100.0, 3),
                "coverage": {
                    "start": coverage_start_iso,
                    "end": coverage_end_iso,
                    "days": int(coverage_total_days),
                },
                "per_symbol_ranges": per_symbol_ranges,
                "data_cache_root": data_cache_root,
                "data_shards": data_shards_daily,
                "costs": {
                    "cost_bps": float(base.get("cost_bps", 0.0)),
                },
                "source": "StrategyAdapter",
            },
        )
    except Exception:
        # Logging is best-effort; never block the run if the file isn't writable yet
        pass

    # Run EA search (helper updates chart live via on_generation_end)
    try:
        top = evo.evolutionary_search(
            strategy_dotted=strategy_dotted,
            tickers=tickers,
            start=start,
            end=end,
            starting_equity=float(equity),
            param_space=param_space,
            generations=int(cfg["generations"]),
            pop_size=int(cfg["pop_size"]),
            min_trades=int(cfg["min_trades"]),
            n_jobs=int(n_jobs),
            progress_cb=_cb,
            log_file=log_file,
        )
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    if not top:
        st.error("EA returned no candidates.")
        st.stop()

    dip_mode = str(base.get("entry_mode", "")).strip().lower() == "dip"
    if dip_mode:
        enriched_top: list[tuple[dict[str, Any], float]] = []
        for params, score in top:
            merged = dict(params)
            merged.setdefault("entry_mode", "dip")
            enriched_top.append((merged, score))
        top = enriched_top

    best_params, best_score = top[0]
    st.session_state["ea_best_params"] = dict(best_params)
    st.session_state["ea_portfolio"] = port_name
    st.session_state["ea_strategy"] = strategy_dotted
    st.session_state["ea_top_results"] = list(top)

    prog.progress(0.95, text="Rendering EA leaderboard…")

    rows = []
    for params, score in top[: min(50, len(top))]:
        r = {"score": float(score)}
        leaderboard_keys = (
            "breakout_n",
            "exit_n",
            "atr_n",
            "atr_multiple",
            "tp_multiple",
            "holding_period_limit",
            "trend_ma",
            "dip_atr_from_high",
            "dip_lookback_high",
            "dip_rsi_max",
            "dip_confirm",
            "dip_cooldown_days",
            "entry_mode",
        )
        r.update({k: params.get(k) for k in leaderboard_keys if k in params})
        rows.append(r)
    lb = pd.DataFrame(rows).sort_values("score", ascending=False)
    st.markdown("**EA leaderboard (top candidates)**")
    st.dataframe(lb, width="stretch", height=360)

    st.session_state["ea_log_file"] = log_file
    st.success(f"EA complete. Best score={best_score:.3f}.")

    st.session_state["adapter_holdout_status"] = (
        "success",
        f"Holdout window {holdout_start.date().isoformat()} → {holdout_end.date().isoformat()} "
        f"({int(holdout_days)} days, {holdout_fraction_actual * 100:.1f}% of coverage) logged to {os.path.basename(log_file)}.",
    )

    latest_training_meta = {
        "latest_training": {
            "train_start": start.date().isoformat(),
            "train_end": end.date().isoformat(),
            "holdout_start": holdout_start.date().isoformat(),
            "holdout_end": holdout_end.date().isoformat(),
            "train_fraction": float(train_fraction_actual),
            "test_fraction": float(holdout_fraction_actual),
            "train_share_pct": round(train_fraction_actual * 100.0, 3),
            "test_share_pct": round(holdout_fraction_actual * 100.0, 3),
            "train_days": int(train_days),
            "test_days": int(holdout_days),
            "coverage_start": coverage_start_iso,
            "coverage_end": coverage_end_iso,
            "coverage_days": int(coverage_total_days),
            "data_cache_root": data_cache_root,
            "ea_log_file": log_file,
            "updated_at": _utc_now().isoformat(),
        }
    }
    try:
        append_to_portfolio(port_name, tickers, meta_update=latest_training_meta)
    except Exception:
        pass



# --- Always-available Save EA Best Params section ---
st.divider()
st.subheader("Save EA Best Params")

ea_best = st.session_state.get("ea_best_params") or {}
if not ea_best:
    st.info("Run training to produce EA params, then save them here.")
else:
    with st.expander("EA best parameters", expanded=False):
        st.json(ea_best)

    portfolio_to_save = st.session_state.get("ea_portfolio") or port_name
    strategy_to_save = st.session_state.get("ea_strategy") or strategy_dotted

    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        do_save_always = st.button(
            "💾 Save EA Best Params",
            type="primary",
            use_container_width=True,
            key="save_ea_btn"
        )
    with col_s2:
        st.markdown(
            f"Saving for **{portfolio_to_save}** using `{strategy_to_save}` strategy.\n"
            "This overwrites any previously saved EA-scoped parameters."
        )

    if do_save_always:
        try:
            saved_path = save_strategy_params(
                portfolio=portfolio_to_save,
                strategy=strategy_to_save,
                params=ea_best,
                scope="ea",
            )
            st.success(f"Saved EA params for '{portfolio_to_save}' → {saved_path or '(path not returned)'}")
        except Exception as e:
            st.error(f"Save failed: {e}")
