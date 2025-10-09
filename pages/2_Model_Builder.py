# pages/2_Model_Builder.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.training_logger import TrainingLogger
from src.data.portfolio_prefetch import intersection_range
from src.storage import (
    append_to_portfolio,
    get_benchmark_total_return,
    list_portfolios,
    load_portfolio,
    save_strategy_params,
)

DIP_STRATEGY_MODULE = "src.models.atr_dip_breakout"
PARAM_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "storage"
    / "params"
    / "model_builder_profiles.json"
)
STRATEGY_OPTIONS: List[Tuple[str, str]] = [
    ("ATR Breakout", "src.models.atr_breakout"),
    ("ATR + Buy-the-Dip Overlay", DIP_STRATEGY_MODULE),
]

BASE_DEFAULTS_BY_STRATEGY: Dict[str, Dict[str, Any]] = {
    "src.models.atr_breakout": {
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
        "vol_target_enabled": False,
        "vol_target_target_pct": 0.0,
        "vol_target_atr_window": 14,
        "vol_target_min_leverage": 0.0,
        "vol_target_max_leverage": 1.0,
        "trend_filter_ma": 0,
        "trend_filter_slope_lookback": 0,
        "trend_filter_slope_threshold": 0.0,
        "trend_filter_exit": False,
    },
    DIP_STRATEGY_MODULE: {
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
        "entry_mode": "dip",
        "trend_ma": 200,
        "dip_atr_from_high": 2.0,
        "dip_lookback_high": 60,
        "dip_rsi_max": 55.0,
        "dip_confirm": False,
        "dip_cooldown_days": 5,
        "vol_target_enabled": False,
        "vol_target_target_pct": 0.0,
        "vol_target_atr_window": 14,
        "vol_target_min_leverage": 0.0,
        "vol_target_max_leverage": 1.0,
        "trend_filter_ma": 0,
        "trend_filter_slope_lookback": 0,
        "trend_filter_slope_threshold": 0.0,
        "trend_filter_exit": False,
    },
}

BASE_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "breakout_n": (5, 300),
    "exit_n": (4, 300),
    "atr_n": (5, 60),
    "atr_multiple": (0.5, 10.0),
    "tp_multiple": (0.2, 10.0),
    "holding_period_limit": (1, 400),
    "risk_per_trade": (0.0005, 0.05),
    "sma_fast": (5, 100),
    "sma_slow": (10, 200),
    "sma_long": (100, 400),
    "long_slope_len": (5, 60),
    "cost_bps": (0.0, 20.0),
    "vol_target_target_pct": (0.0, 0.1),
    "vol_target_atr_window": (1, 200),
    "vol_target_min_leverage": (0.0, 5.0),
    "vol_target_max_leverage": (0.0, 5.0),
    "trend_filter_ma": (0, 400),
    "trend_filter_slope_lookback": (0, 200),
    "trend_filter_slope_threshold": (0.0, 0.02),
}

DIP_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "dip_atr_from_high": (0.0, 20.0),
    "dip_lookback_high": (5, 600),
    "dip_cooldown_days": (0, 240),
    "dip_rsi_max": (0.0, 100.0),
    "trend_ma": (20, 600),
}

EA_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "pop_size": (8, 400),
    "generations": (1, 300),
    "tournament_k": (2, 400),
    "fitness_patience": (0, 100),
    "min_trades": (0, 500),
    "breakout_n_min": (1, 400),
    "breakout_n_max": (1, 400),
    "exit_n_min": (1, 400),
    "exit_n_max": (1, 400),
    "atr_n_min": (1, 200),
    "atr_n_max": (1, 200),
    "atr_multiple_min": (0.1, 20.0),
    "atr_multiple_max": (0.1, 20.0),
    "tp_multiple_min": (0.1, 20.0),
    "tp_multiple_max": (0.1, 20.0),
    "hold_min": (1, 600),
    "hold_max": (1, 600),
    "vol_target_target_pct_min": (0.0, 0.1),
    "vol_target_target_pct_max": (0.0, 0.1),
    "vol_target_atr_window_min": (1, 200),
    "vol_target_atr_window_max": (1, 200),
    "vol_target_min_leverage_min": (0.0, 2.0),
    "vol_target_min_leverage_max": (0.0, 2.0),
    "vol_target_max_leverage_min": (0.5, 5.0),
    "vol_target_max_leverage_max": (0.5, 5.0),
    "trend_filter_ma_min": (0, 400),
    "trend_filter_ma_max": (0, 400),
    "trend_filter_slope_lookback_min": (0, 120),
    "trend_filter_slope_lookback_max": (0, 120),
    "trend_filter_slope_threshold_min": (0.0, 0.02),
    "trend_filter_slope_threshold_max": (0.0, 0.02),
    "trend_filter_exit_min": (0, 1),
    "trend_filter_exit_max": (0, 1),
}

EA_PARAM_MIN_MAX_PAIRS: List[Tuple[str, str]] = [
    ("breakout_n_min", "breakout_n_max"),
    ("exit_n_min", "exit_n_max"),
    ("atr_n_min", "atr_n_max"),
    ("atr_multiple_min", "atr_multiple_max"),
    ("tp_multiple_min", "tp_multiple_max"),
    ("hold_min", "hold_max"),
    ("vol_target_target_pct_min", "vol_target_target_pct_max"),
    ("vol_target_atr_window_min", "vol_target_atr_window_max"),
    ("vol_target_min_leverage_min", "vol_target_min_leverage_max"),
    ("vol_target_max_leverage_min", "vol_target_max_leverage_max"),
    ("trend_filter_ma_min", "trend_filter_ma_max"),
    ("trend_filter_slope_lookback_min", "trend_filter_slope_lookback_max"),
    ("trend_filter_slope_threshold_min", "trend_filter_slope_threshold_max"),
    ("trend_filter_exit_min", "trend_filter_exit_max"),
]

EA_DIP_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "trend_ma_min": (20, 600),
    "trend_ma_max": (20, 600),
    "dip_lookback_high_min": (5, 600),
    "dip_lookback_high_max": (5, 600),
    "dip_atr_from_high_min": (0.0, 20.0),
    "dip_atr_from_high_max": (0.0, 20.0),
    "dip_rsi_max_min": (0.0, 100.0),
    "dip_rsi_max_max": (0.0, 100.0),
    "dip_confirm_min": (0, 1),
    "dip_confirm_max": (0, 1),
    "dip_cooldown_min": (0, 240),
    "dip_cooldown_max": (0, 240),
}

EA_DIP_PARAM_MIN_MAX_PAIRS: List[Tuple[str, str]] = [
    ("trend_ma_min", "trend_ma_max"),
    ("dip_lookback_high_min", "dip_lookback_high_max"),
    ("dip_atr_from_high_min", "dip_atr_from_high_max"),
    ("dip_rsi_max_min", "dip_rsi_max_max"),
    ("dip_confirm_min", "dip_confirm_max"),
    ("dip_cooldown_min", "dip_cooldown_max"),
]

EA_DEFAULTS_BASE: Dict[str, Any] = {
    "generations": 50,
    "pop_size": 64,
    "selection_method": "tournament",
    "tournament_k": 3,
    "replacement": "mu+lambda",
    "elitism_fraction": 0.05,
    "crossover_rate": 0.85,
    "crossover_op": "blend",
    "mutation_rate": 0.10,
    "mutation_scale": 0.20,
    "mutation_scheme": "gaussian",
    "anneal_mutation": True,
    "anneal_floor": 0.05,
    "fitness_patience": 8,
    "seed": None,
    "workers": None,
    "shuffle_eval": True,
    "genewise_clip": True,
    "min_trades": 12,
    "n_jobs": max(1, (os.cpu_count() or 2)),
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
    "vol_target_target_pct_min": 0.0,
    "vol_target_target_pct_max": 0.02,
    "vol_target_atr_window_min": 10,
    "vol_target_atr_window_max": 35,
    "vol_target_min_leverage_min": 0.4,
    "vol_target_min_leverage_max": 0.9,
    "vol_target_max_leverage_min": 1.0,
    "vol_target_max_leverage_max": 1.8,
    "trend_filter_ma_min": 50,
    "trend_filter_ma_max": 200,
    "trend_filter_slope_lookback_min": 8,
    "trend_filter_slope_lookback_max": 35,
    "trend_filter_slope_threshold_min": 0.0001,
    "trend_filter_slope_threshold_max": 0.001,
    "trend_filter_exit_min": 0,
    "trend_filter_exit_max": 1,
}

EA_DEFAULTS_BY_STRATEGY: Dict[str, Dict[str, Any]] = {
    "src.models.atr_breakout": dict(EA_DEFAULTS_BASE),
    DIP_STRATEGY_MODULE: {
        **EA_DEFAULTS_BASE,
        "trend_ma_min": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["trend_ma"],
        "trend_ma_max": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["trend_ma"],
        "dip_atr_from_high_min": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_atr_from_high"],
        "dip_atr_from_high_max": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_atr_from_high"],
        "dip_lookback_high_min": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_lookback_high"],
        "dip_lookback_high_max": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_lookback_high"],
        "dip_rsi_max_min": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_rsi_max"],
        "dip_rsi_max_max": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_rsi_max"],
        "dip_confirm_min": int(bool(BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_confirm"])),
        "dip_confirm_max": int(bool(BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_confirm"])),
        "dip_cooldown_min": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_cooldown_days"],
        "dip_cooldown_max": BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]["dip_cooldown_days"],
    },
}


def _load_param_profiles() -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not PARAM_PROFILE_PATH.exists():
        return {}
    try:
        with PARAM_PROFILE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


PARAM_PROFILES = _load_param_profiles()


def _profiles_for(category: str) -> Dict[str, Dict[str, Any]]:
    data = PARAM_PROFILES.get(category, {})
    if isinstance(data, dict):
        return data
    return {}


def _split_profile_payload(profile: Dict[str, Any]) -> tuple[Dict[str, Any], str | None]:
    if not isinstance(profile, dict):
        return {}, None
    data = {k: v for k, v in profile.items() if not str(k).startswith("_")}
    description = profile.get("_description")
    return data, description


def _profile_state_key(category: str, strategy_key: str | None) -> str:
    return f"{category}::{strategy_key or '__global__'}"


def _profile_widget_key(category: str, strategy_key: str | None) -> str:
    safe_strategy = re.sub(r"[^0-9a-zA-Z_]+", "_", (strategy_key or "global"))
    return f"{category}_profile_{safe_strategy}"


def _profile_selectbox(
    category: str,
    strategy_key: str | None,
    profiles: Dict[str, Dict[str, Any]],
    label: str,
    *,
    default_selection: str | None = None,
):
    options: List[str] = ["Custom"] + list(profiles.keys())
    widget_key = _profile_widget_key(category, strategy_key)
    default_option = default_selection if default_selection in options else options[0]
    if widget_key not in st.session_state or st.session_state[widget_key] not in options:
        st.session_state[widget_key] = default_option
    previous = st.session_state[widget_key]
    selected = st.selectbox(label, options=options, key=widget_key)
    changed = selected != previous
    return selected, changed


def _maybe_apply_profile(
    category: str,
    strategy_key: str | None,
    selection: str,
    changed: bool,
    apply_cb,
) -> bool:
    profile_state = _ss_get_dict("adapter_profile_state", {})
    last_applied_map = profile_state.setdefault("last_applied", {})
    state_key = _profile_state_key(category, strategy_key)
    last_applied = last_applied_map.get(state_key)
    if selection == "Custom":
        if last_applied != "Custom":
            last_applied_map[state_key] = "Custom"
        return False
    if changed or last_applied != selection:
        apply_cb()
        last_applied_map[state_key] = selection
        return True
    return False


def _apply_ea_profile(profile: Dict[str, Any], target: Dict[str, Any]) -> None:
    mapping = {
        "population_size": ("pop_size", int),
        "generations": ("generations", int),
        "crossover_prob": ("crossover_rate", float),
        "mutation_prob": ("mutation_rate", float),
        "mutation_sigma": ("mutation_scale", float),
        "elite_frac": ("elitism_fraction", float),
        "tournament_k": ("tournament_k", int),
        "early_stopping_gens": ("fitness_patience", int),
    }
    for src, (dest, caster) in mapping.items():
        if src in profile and profile[src] is not None:
            target[dest] = caster(profile[src])
    if "seed" in profile:
        seed_value = profile.get("seed")
        target["seed"] = None if seed_value in (None, "") else int(seed_value)


def _apply_bounds_profile(profile: Dict[str, Any], target: Dict[str, Any]) -> None:
    mapping = {
        "entry_lookback_n_min": ("breakout_n_min", int),
        "entry_lookback_n_max": ("breakout_n_max", int),
        "exit_lookback_n_min": ("exit_n_min", int),
        "exit_lookback_n_max": ("exit_n_max", int),
        "atr_period_min": ("atr_n_min", int),
        "atr_period_max": ("atr_n_max", int),
        "stop_atr_mult_min": ("atr_multiple_min", float),
        "stop_atr_mult_max": ("atr_multiple_max", float),
        "take_profit_atr_mult_min": ("tp_multiple_min", float),
        "take_profit_atr_mult_max": ("tp_multiple_max", float),
        "trailing_atr_mult_min": ("trailing_atr_mult_min", float),
        "trailing_atr_mult_max": ("trailing_atr_mult_max", float),
        "max_hold_days_min": ("hold_min", int),
        "max_hold_days_max": ("hold_max", int),
        "vol_target_pct_min": ("vol_target_target_pct_min", float),
        "vol_target_pct_max": ("vol_target_target_pct_max", float),
        "vol_target_window_min": ("vol_target_atr_window_min", int),
        "vol_target_window_max": ("vol_target_atr_window_max", int),
        "vol_target_min_leverage_min": ("vol_target_min_leverage_min", float),
        "vol_target_min_leverage_max": ("vol_target_min_leverage_max", float),
        "vol_target_max_leverage_min": ("vol_target_max_leverage_min", float),
        "vol_target_max_leverage_max": ("vol_target_max_leverage_max", float),
        "trend_filter_ma_min": ("trend_filter_ma_min", int),
        "trend_filter_ma_max": ("trend_filter_ma_max", int),
        "trend_filter_slope_lookback_min": ("trend_filter_slope_lookback_min", int),
        "trend_filter_slope_lookback_max": ("trend_filter_slope_lookback_max", int),
        "trend_filter_slope_threshold_min": ("trend_filter_slope_threshold_min", float),
        "trend_filter_slope_threshold_max": ("trend_filter_slope_threshold_max", float),
        "trend_filter_exit_min": ("trend_filter_exit_min", int),
        "trend_filter_exit_max": ("trend_filter_exit_max", int),
    }
    for src, (dest, caster) in mapping.items():
        if src in profile and profile[src] is not None:
            target[dest] = caster(profile[src])


def _apply_dip_profile(profile: Dict[str, Any], target: Dict[str, Any]) -> None:
    mapping = {
        "trend_ma": ("trend_ma", int),
        "dip_atr_from_high": ("dip_atr_from_high", float),
        "dip_lookback_high": ("dip_lookback_high", int),
        "dip_rsi_max": ("dip_rsi_max", float),
        "dip_cooldown_days": ("dip_cooldown_days", int),
    }
    for src, (dest, caster) in mapping.items():
        if src in profile and profile[src] is not None:
            target[dest] = caster(profile[src])
    if "dip_confirm" in profile:
        target["dip_confirm"] = bool(profile.get("dip_confirm"))


def _apply_strategy_profile(profile: Dict[str, Any], target: Dict[str, Any]) -> None:
    mapping = {
        "entry_lookback_n": ("breakout_n", int),
        "exit_lookback_n": ("exit_n", int),
        "atr_period": ("atr_n", int),
        "stop_atr_mult": ("atr_multiple", float),
        "take_profit_atr_mult": ("tp_multiple", float),
        "trailing_atr_mult": ("trailing_atr_mult", float),
    }
    for src, (dest, caster) in mapping.items():
        if src in profile and profile[src] is not None:
            target[dest] = caster(profile[src])
    if "risk_per_trade_pct" in profile and profile["risk_per_trade_pct"] is not None:
        target["risk_per_trade"] = float(profile["risk_per_trade_pct"]) / 100.0
    if "allow_short" in profile:
        target["allow_short"] = bool(profile["allow_short"])
    if "slippage_bps" in profile and profile["slippage_bps"] is not None:
        slip = float(profile["slippage_bps"])
        target["cost_bps"] = slip
        target["slippage_bps"] = slip
    if "fee_per_trade" in profile and profile["fee_per_trade"] is not None:
        target["per_trade_fee"] = float(profile["fee_per_trade"])
    if "trend_filter" in profile and isinstance(profile["trend_filter"], dict):
        tf = profile["trend_filter"]
        target["use_trend_filter"] = bool(tf.get("enable", False))
        if tf.get("ma_fast") is not None:
            target["sma_fast"] = int(tf["ma_fast"])
        if tf.get("ma_slow") is not None:
            target["sma_slow"] = int(tf["ma_slow"])
        if tf.get("ma_long") is not None:
            target["sma_long"] = int(tf["ma_long"])
        if tf.get("slope_window") is not None:
            target["long_slope_len"] = int(tf["slope_window"])
        if tf.get("ma") is not None:
            target["trend_filter_ma"] = int(tf["ma"])
        if tf.get("slope_lookback") is not None:
            target["trend_filter_slope_lookback"] = int(tf["slope_lookback"])
        if tf.get("slope_threshold") is not None:
            target["trend_filter_slope_threshold"] = float(tf["slope_threshold"])
        if tf.get("exit_on_fail") is not None:
            target["trend_filter_exit"] = bool(tf["exit_on_fail"])
    if "vol_target" in profile and isinstance(profile["vol_target"], dict):
        vt = profile["vol_target"]
        target["vol_target_enabled"] = bool(vt.get("enable", False))
        if vt.get("target_pct") is not None:
            target["vol_target_target_pct"] = float(vt["target_pct"])
        if vt.get("atr_window") is not None:
            target["vol_target_atr_window"] = int(vt["atr_window"])
        if vt.get("min_leverage") is not None:
            target["vol_target_min_leverage"] = float(vt["min_leverage"])
        if vt.get("max_leverage") is not None:
            target["vol_target_max_leverage"] = float(vt["max_leverage"])


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


def _bounded_numeric_default(
    store: Dict[str, Any],
    key: str,
    fallback: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
):
    """Clamp a numeric default into the UI's valid range and coerce to the fallback type."""

    raw = store.get(key, fallback)
    # Determine the target type from the fallback; fall back to float for safety.
    target_type = float
    if isinstance(fallback, bool):
        target_type = int
    elif isinstance(fallback, int) and not isinstance(fallback, bool):
        target_type = int
    elif isinstance(fallback, float):
        target_type = float

    def _cast(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("missing value")
        try:
            return target_type(value)
        except (TypeError, ValueError):
            if target_type is int:
                return int(float(value))
            raise

    try:
        value = _cast(raw)
    except (TypeError, ValueError):
        value = _cast(fallback)

    if min_value is not None and value < min_value:
        value = target_type(min_value)
    if max_value is not None and value > max_value:
        value = target_type(max_value)

    store[key] = value
    return value


def _ensure_min_leq_max(store: Dict[str, Any], min_key: str, max_key: str) -> None:
    """Ensure paired min/max entries remain ordered for Streamlit inputs."""

    if min_key not in store or max_key not in store:
        return
    if store[max_key] < store[min_key]:
        target_type = int if isinstance(store[max_key], bool) else type(store[max_key])
        store[max_key] = target_type(store[min_key])


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


HOLDOUT_EQUITY_KEY = "adapter_holdout_equity_series"
HOLDOUT_BENCHMARK_KEY = "adapter_holdout_benchmark_series"
HOLDOUT_RETURNS_KEY = "adapter_holdout_returns_series"
HOLDOUT_TRADES_KEY = "adapter_holdout_trades"

HOLDOUT_CHART_MARGIN = dict(l=80, r=30, t=40, b=30)


@dataclass
class PortfolioHoldoutResult:
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ratio: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    per_symbol_equity: Dict[str, pd.Series] = field(default_factory=dict)
    per_symbol_ratio: Dict[str, pd.Series] = field(default_factory=dict)
    trades: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


def _ensure_dt_index(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            return series
    if series.index.tz is not None:
        try:
            series.index = series.index.tz_convert(None)
        except Exception:
            series.index = series.index.tz_localize(None)
    return series.sort_index()


def _flatten_trades(trades: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for symbol, trade_list in (trades or {}).items():
        if not isinstance(trade_list, list):
            continue
        for trade in trade_list:
            if not isinstance(trade, dict):
                continue
            entry = trade.get("entry_time") or trade.get("entry_dt")
            exit_ = trade.get("exit_time") or trade.get("exit_dt")
            if entry is None or exit_ is None:
                continue
            entry_ts = pd.to_datetime(entry, errors="coerce")
            exit_ts = pd.to_datetime(exit_, errors="coerce")
            if pd.isna(entry_ts) or pd.isna(exit_ts):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "entry_time": entry_ts.tz_localize(None) if getattr(entry_ts, "tzinfo", None) else entry_ts,
                    "exit_time": exit_ts.tz_localize(None) if getattr(exit_ts, "tzinfo", None) else exit_ts,
                    "return_pct": float(trade.get("return_pct", 0.0) or 0.0),
                    "holding_days": int(trade.get("holding_days", 0) or 0),
                    "quantity": float(trade.get("quantity", trade.get("qty", 0.0)) or 0.0),
                    "net_pnl": float(trade.get("net_pnl", 0.0) or 0.0),
                    "notional_entry": float(trade.get("notional_entry", trade.get("notional", 0.0)) or 0.0),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "return_pct"])
    df = pd.DataFrame(rows)
    df = df.sort_values("entry_time")
    return df


def _render_holdout_equity_chart(
    placeholder: st.delta_generator.DeltaGenerator,
    equity: pd.Series | None,
    benchmark: pd.Series | None,
    x_range: Tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> None:
    if equity is None or equity.empty:
        placeholder.info("Holdout equity will appear when a best candidate is available.")
        return

    equity = _ensure_dt_index(equity.copy())
    strategy_values = np.asarray([float(v) for v in equity.values])
    fig = go.Figure()

    if benchmark is not None and not benchmark.empty:
        benchmark = _ensure_dt_index(benchmark.copy())
        bench_aligned = benchmark.reindex(equity.index).ffill().bfill()
        bench_values = np.asarray([float(v) if pd.notna(v) else np.nan for v in bench_aligned.values])

        if np.isfinite(bench_values).any():
            diff = strategy_values - bench_values
            outperform_mask = diff >= 0
            underperform_mask = diff < 0

            def _add_fill(mask: np.ndarray, name: str, color: str) -> None:
                if not mask.any():
                    return
                upper = np.where(mask, np.maximum(strategy_values, bench_values), None)
                lower = np.where(mask, np.minimum(strategy_values, bench_values), None)
                custom_diff = np.where(mask, diff, None)
                fig.add_trace(
                    go.Scatter(
                        x=list(equity.index),
                        y=upper,
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        connectgaps=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=list(equity.index),
                        y=lower,
                        mode="lines",
                        fill="tonexty",
                        fillcolor=color,
                        line=dict(width=0),
                        name=name,
                        showlegend=True,
                        connectgaps=False,
                        customdata=custom_diff,
                        hovertemplate="Date=%{x|%Y-%m-%d}<br>Diff=$%{customdata:.2f}<extra></extra>",
                    )
                )

            _add_fill(outperform_mask, "Outperformance", "rgba(40, 167, 69, 0.25)")
            _add_fill(underperform_mask, "Underperformance", "rgba(220, 53, 69, 0.25)")

            fig.add_trace(
                go.Scatter(
                    x=list(bench_aligned.index),
                    y=bench_values,
                    mode="lines",
                    name="Benchmark",
                    line=dict(width=1.8, color="#6c757d", dash="dash"),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=list(equity.index),
            y=strategy_values,
            mode="lines",
            name="Strategy",
            line=dict(width=2.2, color="#1f78b4"),
        )
    )

    fig.update_layout(
        title="Holdout equity (anchored at starting capital)",
        margin=HOLDOUT_CHART_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        yaxis_title_standoff=10,
    )
    if x_range is not None:
        fig.update_layout(xaxis=dict(range=[x_range[0], x_range[1]]))
    placeholder.plotly_chart(fig, use_container_width=True)


def _render_holdout_heatmap(
    placeholder: st.delta_generator.DeltaGenerator,
    returns: pd.Series | None,
    x_range: Tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> None:
    container = placeholder.container()

    if returns is None or returns.dropna().empty:
        container.info("Rolling performance heatmap will populate after a completed holdout run.")
        return

    returns = _ensure_dt_index(returns.copy())
    windows = [5, 10, 20, 60]
    metrics: Dict[str, pd.Series] = {}
    for window in windows:
        if len(returns) < window:
            metrics[f"{window}-day"] = pd.Series(np.nan, index=returns.index)
            continue
        roll_prod = (1.0 + returns).rolling(window).apply(lambda arr: float(np.prod(arr)), raw=True)
        roll_prod = roll_prod.where(roll_prod > 0.0)
        ann_return = roll_prod.pow(252 / window) - 1.0
        metrics[f"{window}-day"] = ann_return * 100.0

    heatmap_df = pd.DataFrame(metrics).dropna(how="all")
    if heatmap_df.empty:
        container.info("Not enough overlapping holdout data to compute rolling performance windows yet.")
        return

    heatmap_df = heatmap_df.fillna(np.nan)
    z = heatmap_df.T.values
    x = list(heatmap_df.index)
    y = list(heatmap_df.columns)
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                colorscale="RdYlGn",
                colorbar=dict(title="Ann. return (%)"),
                zmid=0,
                hovertemplate="Window=%{y}<br>Date=%{x|%Y-%m-%d}<br>Ann. return=%{z:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Rolling performance heatmap",
        margin=HOLDOUT_CHART_MARGIN,
        xaxis_title="Date",
        yaxis_title="Rolling window",
        yaxis_title_standoff=10,
    )
    if x_range is not None:
        fig.update_layout(xaxis=dict(range=[x_range[0], x_range[1]]))
    container.plotly_chart(fig, use_container_width=True)
    container.markdown(
        """
        **How to read this heatmap**

        * Each row represents a rolling window (e.g., 5-day) of compounded strategy returns.
        * Colors encode the annualized return for that window—greens indicate positive momentum, yellows are flat, and reds
          highlight stretches of drawdown.
        * Line up the vertical date bands with the equity curve above and trade timeline below to spot which clusters of
          trades produced the strongest or weakest performance at different horizons.
        """
    )


def _render_trade_timeline(
    placeholder: st.delta_generator.DeltaGenerator,
    trades_df: pd.DataFrame | None,
    max_trades: int = 200,
    x_range: Tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> None:
    if trades_df is None or trades_df.empty:
        placeholder.info("Trade timeline will display once closed holds appear in the holdout window.")
        return

    df = trades_df.copy()
    df = df.dropna(subset=["entry_time", "exit_time"])
    if df.empty:
        placeholder.info("Trade timeline will display once closed holds appear in the holdout window.")
        return
    df = df.sort_values("entry_time")
    if len(df) > max_trades:
        df = df.iloc[-max_trades:]

    df["return_pct_display"] = df["return_pct"] * 100.0
    df["duration_days"] = (df["exit_time"] - df["entry_time"]).dt.days.clip(lower=0)

    fig = px.timeline(
        df,
        x_start="entry_time",
        x_end="exit_time",
        y="symbol",
        color="return_pct_display",
        color_continuous_scale="RdYlGn",
        hover_data={
            "return_pct_display": ":.2f",
            "duration_days": True,
            "quantity": ":.2f",
            "net_pnl": ":.2f",
        },
    )
    fig.update_layout(
        title="Trade lifecycle timeline",
        margin=HOLDOUT_CHART_MARGIN,
        coloraxis_colorbar=dict(title="Return (%)"),
        xaxis_title="Date",
        yaxis_title="Symbol",
        yaxis_title_standoff=10,
    )
    if x_range is not None:
        fig.update_layout(xaxis=dict(range=[x_range[0], x_range[1]]))
    fig.update_yaxes(autorange="reversed")
    fig.update_coloraxes(cmid=0)
    placeholder.plotly_chart(fig, use_container_width=True)


def _resolve_holdout_x_range(
    equity: pd.Series | None,
    returns: pd.Series | None,
    trades_df: pd.DataFrame | None,
) -> Tuple[pd.Timestamp, pd.Timestamp] | None:
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    if equity is not None and not equity.empty:
        idx = _ensure_dt_index(equity.copy()).index
        if len(idx):
            spans.append((idx.min().to_pydatetime(), idx.max().to_pydatetime()))

    if returns is not None and not returns.dropna().empty:
        idx = _ensure_dt_index(returns.copy()).index
        if len(idx):
            spans.append((idx.min().to_pydatetime(), idx.max().to_pydatetime()))

    if trades_df is not None and not trades_df.empty:
        df = trades_df.dropna(subset=["entry_time", "exit_time"])
        if not df.empty:
            entry_min = pd.to_datetime(df["entry_time"], errors="coerce").dropna()
            exit_max = pd.to_datetime(df["exit_time"], errors="coerce").dropna()
            if not entry_min.empty and not exit_max.empty:
                spans.append((entry_min.min().to_pydatetime(), exit_max.max().to_pydatetime()))

    if not spans:
        return None

    start = min(span[0] for span in spans)
    end = max(span[1] for span in spans)
    if start == end:
        end = end + timedelta(days=1)
    return (start, end)


def _portfolio_equity_curve(
        strategy_dotted: str,
        tickers: List[str],
        start,
        end,
        starting_equity: float,
        params: Dict[str, Any],
) -> PortfolioHoldoutResult:
    """Simulate aggregated portfolio equity for the given params on [start, end)."""

    try:
        mod = _safe_import(strategy_dotted)
        run = getattr(mod, "run_strategy")
    except Exception:
        return PortfolioHoldoutResult()

    base_equity = float(starting_equity)
    per_symbol_equity: Dict[str, pd.Series] = {}
    per_symbol_ratio: Dict[str, pd.Series] = {}
    trades_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    for sym in tickers:
        try:
            result = run(sym, start, end, starting_equity, params)
        except Exception:
            continue

        if isinstance(result, dict):
            trades_payload = result.get("trades")
            if isinstance(trades_payload, list):
                trades_by_symbol[sym] = [dict(t) for t in trades_payload if isinstance(t, dict)]
            eq = result.get("equity")
        else:
            eq = None

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

        eq = _ensure_dt_index(eq)
        if eq.empty:
            continue

        first_value = float(eq.iloc[0]) if len(eq) else 0.0
        if abs(first_value) < 1e-9:
            anchored = eq + base_equity
        else:
            anchored = eq

        anchored = anchored.astype(float)
        anchored = _ensure_dt_index(anchored)
        if anchored.empty:
            continue

        start_val = float(anchored.iloc[0]) if len(anchored) else 0.0
        if abs(start_val) < 1e-9:
            continue

        ratio = (anchored / start_val).astype(float)
        per_symbol_equity[sym] = anchored
        per_symbol_ratio[sym] = ratio

    if not per_symbol_ratio:
        return PortfolioHoldoutResult(trades=trades_by_symbol)

    ratio_df = pd.DataFrame(per_symbol_ratio).sort_index()
    ratio_df = ratio_df.ffill().dropna(how="all")
    if ratio_df.empty:
        return PortfolioHoldoutResult(trades=trades_by_symbol)

    portfolio_ratio = ratio_df.mean(axis=1, skipna=True)
    portfolio_ratio = portfolio_ratio.dropna()
    if portfolio_ratio.empty:
        return PortfolioHoldoutResult(trades=trades_by_symbol)
    portfolio_ratio.name = "portfolio_ratio"

    portfolio_equity = portfolio_ratio * base_equity
    portfolio_equity.name = "portfolio_equity"

    aligned_equity: Dict[str, pd.Series] = {}
    aligned_ratio: Dict[str, pd.Series] = {}
    for sym, series in per_symbol_equity.items():
        aligned_equity[sym] = series.reindex(portfolio_equity.index).ffill()
    for sym, series in per_symbol_ratio.items():
        aligned_ratio[sym] = series.reindex(portfolio_ratio.index).ffill()

    return PortfolioHoldoutResult(
        equity=portfolio_equity,
        ratio=portfolio_ratio,
        per_symbol_equity=aligned_equity,
        per_symbol_ratio=aligned_ratio,
        trades=trades_by_symbol,
    )


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

cache_only_hint = bool(data_shards_daily)
cache_env_values = {"cache", "cache_only", "disk", "offline"}
current_provider = os.environ.get("DATA_PROVIDER")
default_cache_only = current_provider in cache_env_values or cache_only_hint
cache_only_state = st.checkbox(
    "Use cached OHLCV only (skip live API fetches)",
    value=st.session_state.get("_adapter_cache_only", default_cache_only),
    help=(
        "When enabled the trainer will read OHLCV from disk cache and raise an error if data is missing. "
        "Disable to allow automatic Alpaca/Yahoo requests."
    ),
)
st.session_state["_adapter_cache_only"] = cache_only_state

if cache_only_state:
    if current_provider not in cache_env_values:
        st.session_state["_adapter_prev_data_provider"] = current_provider
    os.environ["DATA_PROVIDER"] = "cache_only"
    st.caption("Cache-only mode active — remote providers are disabled for training runs.")
else:
    if current_provider in cache_env_values:
        prev_provider = st.session_state.get("_adapter_prev_data_provider")
        if prev_provider:
            os.environ["DATA_PROVIDER"] = prev_provider
        else:
            os.environ.pop("DATA_PROVIDER", None)

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

strategy_labels = [label for label, _ in STRATEGY_OPTIONS]
strategy_modules = [module for _, module in STRATEGY_OPTIONS]
default_strategy = st.session_state.get("adapter_strategy", strategy_modules[0])
try:
    default_strategy_index = strategy_modules.index(default_strategy)
except ValueError:
    default_strategy_index = 0
selected_label = st.selectbox(
    "Strategy",
    strategy_labels,
    index=default_strategy_index,
    help="Select the strategy module to adapt.",
)
strategy_dotted = strategy_modules[strategy_labels.index(selected_label)]
st.session_state["adapter_strategy"] = strategy_dotted
dip_strategy = strategy_dotted == DIP_STRATEGY_MODULE

base_defaults_map = {key: dict(value) for key, value in BASE_DEFAULTS_BY_STRATEGY.items()}
base_map = _ss_get_dict("adapter_base_params_map", base_defaults_map)
if strategy_dotted not in base_map:
    base_map[strategy_dotted] = dict(BASE_DEFAULTS_BY_STRATEGY[strategy_dotted])
base = base_map[strategy_dotted]
strategy_defaults = BASE_DEFAULTS_BY_STRATEGY[strategy_dotted]
base["entry_mode"] = "dip" if dip_strategy else "breakout"
if dip_strategy:
    for field, value in BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE].items():
        base.setdefault(field, value)

for field, bounds in BASE_PARAM_RANGES.items():
    if field in strategy_defaults:
        _bounded_numeric_default(
            base,
            field,
            strategy_defaults[field],
            min_value=bounds[0],
            max_value=bounds[1],
        )

if dip_strategy:
    dip_defaults = BASE_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]
    for field, bounds in DIP_PARAM_RANGES.items():
        _bounded_numeric_default(
            base,
            field,
            dip_defaults[field],
            min_value=bounds[0],
            max_value=bounds[1],
        )

ea_defaults_map = {key: dict(value) for key, value in EA_DEFAULTS_BY_STRATEGY.items()}
ea_cfg_map = _ss_get_dict("ea_cfg_map", ea_defaults_map)
if strategy_dotted not in ea_cfg_map:
    ea_cfg_map[strategy_dotted] = dict(EA_DEFAULTS_BY_STRATEGY[strategy_dotted])
ea_cfg = ea_cfg_map[strategy_dotted]

if dip_strategy:
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

for _k, _v in {
    "selection_method": "tournament",
    "tournament_k": 3,
    "replacement": "mu+lambda",
    "elitism_fraction": 0.05,
    "crossover_rate": 0.85,
    "crossover_op": "blend",
    "mutation_rate": 0.10,
    "mutation_scale": 0.20,
    "mutation_scheme": "gaussian",
    "anneal_mutation": True,
    "anneal_floor": 0.05,
    "fitness_patience": 8,
    "seed": None,
    "workers": None,
    "shuffle_eval": True,
    "genewise_clip": True,
}.items():
    ea_cfg.setdefault(_k, _v)

ea_defaults = EA_DEFAULTS_BY_STRATEGY[strategy_dotted]
for field, bounds in EA_PARAM_RANGES.items():
    if field in ea_defaults:
        _bounded_numeric_default(
            ea_cfg,
            field,
            ea_defaults[field],
            min_value=bounds[0],
            max_value=bounds[1],
        )

ea_jobs_max = max(1, (os.cpu_count() or 2))
if "n_jobs" in ea_defaults:
    _bounded_numeric_default(
        ea_cfg,
        "n_jobs",
        ea_defaults["n_jobs"],
        min_value=1,
        max_value=ea_jobs_max,
    )

ea_cfg["tournament_k"] = int(
    max(2, min(int(ea_cfg.get("tournament_k", 3)), int(ea_cfg.get("pop_size", 8)))))

for min_key, max_key in EA_PARAM_MIN_MAX_PAIRS:
    if min_key in ea_defaults and max_key in ea_defaults:
        _ensure_min_leq_max(ea_cfg, min_key, max_key)

if dip_strategy:
    dip_ea_defaults = EA_DEFAULTS_BY_STRATEGY[DIP_STRATEGY_MODULE]
    for field, bounds in EA_DIP_PARAM_RANGES.items():
        _bounded_numeric_default(
            ea_cfg,
            field,
            dip_ea_defaults[field],
            min_value=bounds[0],
            max_value=bounds[1],
        )
    for min_key, max_key in EA_DIP_PARAM_MIN_MAX_PAIRS:
        _ensure_min_leq_max(ea_cfg, min_key, max_key)

with st.expander("EA Parameters", expanded=False):
    ea_profiles = _profiles_for("ea_parameters")
    if ea_profiles:
        selection, changed = _profile_selectbox(
            "ea_parameters",
            strategy_dotted,
            ea_profiles,
            "EA parameter profile",
            default_selection="Balanced Default",
        )
        profile_raw = ea_profiles.get(selection, {})
        profile_data, profile_description = _split_profile_payload(profile_raw)
        _maybe_apply_profile(
            "ea_parameters",
            strategy_dotted,
            selection,
            changed,
            lambda data=profile_data: _apply_ea_profile(data, ea_cfg),
        )
        if selection != "Custom":
            if profile_description:
                st.caption(profile_description)
            st.caption(f"Profile '{selection}' loaded for EA parameters.")

    primary_cols = st.columns(3)
    with primary_cols[0]:
        ea_cfg["pop_size"] = st.number_input(
            "Population size",
            min_value=8,
            max_value=400,
            value=int(ea_cfg.get("pop_size", 64)),
            step=1,
            help="Population per generation. Larger explores more, costs more time.",
        )
        ea_cfg["generations"] = st.number_input(
            "Generations",
            min_value=1,
            max_value=300,
            value=int(ea_cfg.get("generations", 50)),
            step=1,
            help="Number of generations (iterations) to evolve.",
        )
        ea_cfg["selection_method"] = st.selectbox(
            "Selection method",
            options=["tournament", "rank", "roulette"],
            index=["tournament", "rank", "roulette"].index(str(ea_cfg.get("selection_method", "tournament"))),
            help="Parent selection pressure and bias.",
        )
    with primary_cols[1]:
        tournament_disabled = ea_cfg["selection_method"] != "tournament"
        ea_cfg["tournament_k"] = st.number_input(
            "Tournament k",
            min_value=2,
            max_value=max(2, int(ea_cfg["pop_size"])),
            value=int(ea_cfg.get("tournament_k", 3)),
            step=1,
            help="Higher k = stronger selection pressure.",
            disabled=tournament_disabled,
        )
        replacement_labels = {"generational": "generational", "mu+lambda": "μ+λ"}
        current_replacement = replacement_labels.get(str(ea_cfg.get("replacement", "mu+lambda")), "μ+λ")
        selected_replacement = st.selectbox(
            "Replacement",
            options=["generational", "μ+λ"],
            index=["generational", "μ+λ"].index(current_replacement),
            help="How survivors are chosen into the next gen.",
        )
        ea_cfg["replacement"] = "mu+lambda" if selected_replacement == "μ+λ" else "generational"
        ea_cfg["elitism_fraction"] = st.slider(
            "Elitism fraction",
            min_value=0.0,
            max_value=0.20,
            value=float(ea_cfg.get("elitism_fraction", 0.05)),
            step=0.01,
            help="Top X% copied unchanged to next generation.",
        )
    with primary_cols[2]:
        ea_cfg["crossover_rate"] = st.slider(
            "Crossover rate",
            min_value=0.50,
            max_value=1.0,
            value=float(ea_cfg.get("crossover_rate", 0.85)),
            step=0.01,
            help="Chance to recombine parents into offspring.",
        )
        ea_cfg["crossover_op"] = st.selectbox(
            "Crossover operator",
            options=["blend", "sbx", "one_point"],
            index=["blend", "sbx", "one_point"].index(str(ea_cfg.get("crossover_op", "blend"))),
            help="Recombination operator for real/mixed genomes.",
        )
        ea_cfg["mutation_rate"] = st.slider(
            "Mutation rate",
            min_value=0.02,
            max_value=0.30,
            value=float(ea_cfg.get("mutation_rate", 0.10)),
            step=0.01,
            help="Per-gene chance to mutate. Higher = more exploration.",
        )
        ea_cfg["mutation_scale"] = st.slider(
            "Mutation scale",
            min_value=0.05,
            max_value=0.50,
            value=float(ea_cfg.get("mutation_scale", 0.20)),
            step=0.01,
            help="Typical mutation step as fraction of parameter range.",
        )

    secondary_cols = st.columns(3)
    with secondary_cols[0]:
        ea_cfg["mutation_scheme"] = st.selectbox(
            "Mutation scheme",
            options=["gaussian", "polynomial", "uniform_reset"],
            index=["gaussian", "polynomial", "uniform_reset"].index(str(ea_cfg.get("mutation_scheme", "gaussian"))),
            help="Distribution used for mutations.",
        )
        ea_cfg["anneal_mutation"] = st.checkbox(
            "Anneal mutation",
            value=bool(ea_cfg.get("anneal_mutation", True)),
            help="Gradually reduce mutation scale across generations.",
        )
    with secondary_cols[1]:
        ea_cfg["anneal_floor"] = st.slider(
            "Anneal floor",
            min_value=0.0,
            max_value=0.20,
            value=float(ea_cfg.get("anneal_floor", 0.05)),
            step=0.01,
            help="Minimum mutation scale when annealing is enabled.",
            disabled=not ea_cfg["anneal_mutation"],
        )
        ea_cfg["fitness_patience"] = st.number_input(
            "Fitness patience",
            min_value=0,
            max_value=100,
            value=int(ea_cfg.get("fitness_patience", 8)),
            step=1,
            help="Early stop if best score doesn’t improve for N generations.",
        )
    with secondary_cols[2]:
        seed_default = "" if ea_cfg.get("seed") in (None, "") else str(ea_cfg.get("seed"))
        seed_str = st.text_input(
            "Seed (optional)",
            value=seed_default,
            help="Set for reproducible runs (may be approximate with multiprocessing).",
        )
        try:
            ea_cfg["seed"] = int(seed_str.strip()) if seed_str.strip() else None
        except ValueError:
            ea_cfg["seed"] = None
        workers_default = "" if ea_cfg.get("workers") in (None, "") else str(ea_cfg.get("workers"))
        workers_str = st.text_input(
            "Workers (optional)",
            value=workers_default,
            help="Parallel workers. Leave blank for auto.",
        )
        try:
            ea_cfg["workers"] = int(workers_str.strip()) if workers_str.strip() else None
        except ValueError:
            ea_cfg["workers"] = None

    guard_cols = st.columns(3)
    with guard_cols[0]:
        ea_cfg["min_trades"] = st.number_input(
            "Min trades (gate)",
            min_value=0,
            max_value=500,
            value=int(ea_cfg.get("min_trades", 12)),
            step=1,
            help="Discard chromosomes with fewer executed trades.",
        )
    with guard_cols[1]:
        ea_cfg["n_jobs"] = st.number_input(
            "Jobs (EA)",
            min_value=1,
            max_value=max(1, (os.cpu_count() or 2)),
            value=int(ea_cfg.get("n_jobs", 1)),
            step=1,
            help="Worker processes dedicated to the evolutionary search.",
        )
    with guard_cols[2]:
        ea_cfg["shuffle_eval"] = st.checkbox(
            "Shuffle evaluations",
            value=bool(ea_cfg.get("shuffle_eval", True)),
            help="Randomize eval order to reduce hot-spot skew in parallel runs.",
        )

    ea_cfg["genewise_clip"] = True

with st.expander("Optimization parameter bounds", expanded=True):
    bounds_profiles = _profiles_for("optimization_bounds")
    if bounds_profiles:
        selection, changed = _profile_selectbox(
            "optimization_bounds",
            strategy_dotted,
            bounds_profiles,
            "Bounds profile",
            default_selection="Active-Tight Risk",
        )
        profile_raw = bounds_profiles.get(selection, {})
        profile_data, profile_description = _split_profile_payload(profile_raw)
        _maybe_apply_profile(
            "optimization_bounds",
            strategy_dotted,
            selection,
            changed,
            lambda data=profile_data: _apply_bounds_profile(data, ea_cfg),
        )
        if selection != "Custom":
            if profile_description:
                st.caption(profile_description)
            st.caption(f"Profile '{selection}' loaded for optimization bounds.")

    bounds_cols = st.columns(3)
    with bounds_cols[0]:
        bnm_lo = st.number_input(
            "breakout_n min",
            1,
            400,
            int(ea_cfg["breakout_n_min"]),
            1,
            help="Lower breakout lookback bound searched by the EA.",
        )
        bnm_hi = st.number_input(
            "breakout_n max",
            int(bnm_lo),
            400,
            int(ea_cfg["breakout_n_max"]),
            1,
            help="Upper breakout lookback bound searched by the EA.",
        )
        enm_lo = st.number_input(
            "exit_n min",
            1,
            400,
            int(ea_cfg["exit_n_min"]),
            1,
            help="Smallest exit lookback considered during evolution.",
        )
        enm_hi = st.number_input(
            "exit_n max",
            int(enm_lo),
            400,
            int(ea_cfg["exit_n_max"]),
            1,
            help="Largest exit lookback considered during evolution.",
        )
    with bounds_cols[1]:
        atm_lo = st.number_input(
            "atr_n min",
            1,
            200,
            int(ea_cfg["atr_n_min"]),
            1,
            help="Minimum ATR window for volatility sizing.",
        )
        atm_hi = st.number_input(
            "atr_n max",
            int(atm_lo),
            200,
            int(ea_cfg["atr_n_max"]),
            1,
            help="Maximum ATR window for volatility sizing.",
        )
        atm_mul_lo = st.number_input(
            "atr_multiple min",
            0.1,
            20.0,
            float(ea_cfg["atr_multiple_min"]),
            0.1,
            help="Smallest ATR multiple sampled for stops.",
        )
        atm_mul_hi = st.number_input(
            "atr_multiple max",
            float(atm_mul_lo),
            20.0,
            float(ea_cfg["atr_multiple_max"]),
            0.1,
            help="Largest ATR multiple sampled for stops.",
        )
    with bounds_cols[2]:
        tpm_lo = st.number_input(
            "tp_multiple min",
            0.1,
            20.0,
            float(ea_cfg["tp_multiple_min"]),
            0.1,
            help="Lower bound for take-profit multiples.",
        )
        tpm_hi = st.number_input(
            "tp_multiple max",
            float(tpm_lo),
            20.0,
            float(ea_cfg["tp_multiple_max"]),
            0.1,
            help="Upper bound for take-profit multiples.",
        )
        hold_lo = st.number_input(
            "hold min",
            1,
            600,
            int(ea_cfg["hold_min"]),
            1,
            help="Shortest holding period allowed during tuning.",
        )
        hold_hi = st.number_input(
            "hold max",
            int(hold_lo),
            600,
            int(ea_cfg["hold_max"]),
            1,
            help="Longest holding period allowed during tuning.",
        )

    ea_cfg["breakout_n_min"], ea_cfg["breakout_n_max"] = int(bnm_lo), int(bnm_hi)
    ea_cfg["exit_n_min"], ea_cfg["exit_n_max"] = int(enm_lo), int(enm_hi)
    ea_cfg["atr_n_min"], ea_cfg["atr_n_max"] = int(atm_lo), int(atm_hi)
    ea_cfg["atr_multiple_min"], ea_cfg["atr_multiple_max"] = float(atm_mul_lo), float(atm_mul_hi)
    ea_cfg["tp_multiple_min"], ea_cfg["tp_multiple_max"] = float(tpm_lo), float(tpm_hi)
    ea_cfg["hold_min"], ea_cfg["hold_max"] = int(hold_lo), int(hold_hi)

if dip_strategy:
    with st.expander("Buy-the-Dip Parameters", expanded=True):
        dip_profiles = _profiles_for("buy_the_dip")
        if dip_profiles:
            selection, changed = _profile_selectbox(
                "buy_the_dip",
                strategy_dotted,
                dip_profiles,
                "Dip overlay profile",
                default_selection="Deep Corrections Only",
            )
            profile_raw = dip_profiles.get(selection, {})
            profile_data, profile_description = _split_profile_payload(profile_raw)
            _maybe_apply_profile(
                "buy_the_dip",
                strategy_dotted,
                selection,
                changed,
                lambda data=profile_data: _apply_dip_profile(data, base),
            )
            if selection != "Custom":
                if profile_description:
                    st.caption(profile_description)
                st.caption(f"Profile '{selection}' loaded for dip overlay parameters.")

        st.markdown("**Optimization bounds**")
        dip_bounds = st.columns(3)
        with dip_bounds[0]:
            trend_lo = st.number_input(
                "trend_ma min",
                20,
                600,
                int(ea_cfg.get("trend_ma_min", base.get("trend_ma", 200))),
                1,
                help="Minimum trend moving average length to qualify dip entries.",
            )
            trend_hi = st.number_input(
                "trend_ma max",
                int(trend_lo),
                600,
                int(ea_cfg.get("trend_ma_max", base.get("trend_ma", 200))),
                1,
                help="Maximum trend moving average length tested for dip entries.",
            )
            dlh_lo = st.number_input(
                "dip_lookback_high min",
                5,
                600,
                int(ea_cfg.get("dip_lookback_high_min", base.get("dip_lookback_high", 60))),
                1,
                help="Shortest window for measuring prior highs in dip mode.",
            )
            dlh_hi = st.number_input(
                "dip_lookback_high max",
                int(dlh_lo),
                600,
                int(ea_cfg.get("dip_lookback_high_max", base.get("dip_lookback_high", 60))),
                1,
                help="Longest window for measuring prior highs in dip mode.",
            )
        with dip_bounds[1]:
            dah_lo = st.number_input(
                "dip_atr_from_high min",
                0.0,
                20.0,
                float(ea_cfg.get("dip_atr_from_high_min", base.get("dip_atr_from_high", 2.0))),
                0.1,
                help="Smallest ATR pullback from recent highs to trigger dip entries.",
            )
            dah_hi = st.number_input(
                "dip_atr_from_high max",
                float(dah_lo),
                20.0,
                float(ea_cfg.get("dip_atr_from_high_max", base.get("dip_atr_from_high", 2.0))),
                0.1,
                help="Largest ATR pullback from highs to consider for dip entries.",
            )
            drs_lo = st.number_input(
                "dip_rsi_max min",
                0.0,
                100.0,
                float(ea_cfg.get("dip_rsi_max_min", base.get("dip_rsi_max", 55.0))),
                1.0,
                help="Lower bound for RSI filter during dip setup qualification.",
            )
            drs_hi = st.number_input(
                "dip_rsi_max max",
                float(drs_lo),
                100.0,
                float(ea_cfg.get("dip_rsi_max_max", base.get("dip_rsi_max", 55.0))),
                1.0,
                help="Upper bound for RSI filter during dip setup qualification.",
            )
        with dip_bounds[2]:
            dcf_lo = st.number_input(
                "dip_confirm min",
                0,
                1,
                int(ea_cfg.get("dip_confirm_min", int(bool(base.get("dip_confirm", False))))),
                1,
                help="Minimum confirmation flag (0/1) permitted for dip entries.",
            )
            dcf_hi = st.number_input(
                "dip_confirm max",
                int(dcf_lo),
                1,
                int(ea_cfg.get("dip_confirm_max", int(bool(base.get("dip_confirm", False))))),
                1,
                help="Maximum confirmation flag (0/1) permitted for dip entries.",
            )
            dcd_lo = st.number_input(
                "dip_cooldown_days min",
                0,
                240,
                int(ea_cfg.get("dip_cooldown_min", base.get("dip_cooldown_days", 5))),
                1,
                help="Minimum cooldown period (in days) between dip entries.",
            )
            dcd_hi = st.number_input(
                "dip_cooldown_days max",
                int(dcd_lo),
                240,
                int(ea_cfg.get("dip_cooldown_max", base.get("dip_cooldown_days", 5))),
                1,
                help="Maximum cooldown period (in days) between dip entries.",
            )

        ea_cfg["trend_ma_min"], ea_cfg["trend_ma_max"] = int(trend_lo), int(trend_hi)
        ea_cfg["dip_lookback_high_min"], ea_cfg["dip_lookback_high_max"] = int(dlh_lo), int(dlh_hi)
        ea_cfg["dip_atr_from_high_min"], ea_cfg["dip_atr_from_high_max"] = float(dah_lo), float(dah_hi)
        ea_cfg["dip_rsi_max_min"], ea_cfg["dip_rsi_max_max"] = float(drs_lo), float(drs_hi)
        ea_cfg["dip_confirm_min"], ea_cfg["dip_confirm_max"] = int(dcf_lo), int(dcf_hi)
        ea_cfg["dip_cooldown_min"], ea_cfg["dip_cooldown_max"] = int(dcd_lo), int(dcd_hi)

        st.markdown("**Default dip behaviour**")
        dip_cols = st.columns(2)
        with dip_cols[0]:
            base["dip_atr_from_high"] = st.number_input(
                "dip_atr_from_high",
                0.0,
                20.0,
                float(base.get("dip_atr_from_high", 2.0)),
                0.1,
                help="Minimum ATR pullback from highs required to enter.",
            )
            base["dip_lookback_high"] = st.number_input(
                "dip_lookback_high",
                5,
                600,
                int(base.get("dip_lookback_high", 60)),
                1,
                help="Lookback window for recent highs when detecting dips.",
            )
            base["dip_cooldown_days"] = st.number_input(
                "dip_cooldown_days",
                0,
                240,
                int(base.get("dip_cooldown_days", 5)),
                1,
                help="Cooldown (days) before allowing another dip entry.",
            )
        with dip_cols[1]:
            base["dip_rsi_max"] = st.number_input(
                "dip_rsi_max",
                0.0,
                100.0,
                float(base.get("dip_rsi_max", 55.0)),
                1.0,
                help="RSI threshold dip setups must stay below.",
            )
            base["dip_confirm"] = st.checkbox(
                "dip_confirm",
                value=bool(base.get("dip_confirm", False)),
                help="Require confirmation (close above prior close) before entry.",
            )
        base["trend_ma"] = st.number_input(
            "trend_ma",
            20,
            600,
            int(base.get("trend_ma", 200)),
            1,
            help="Trend moving average length for dip qualification.",
        )
else:
    st.info("Select the ATR + Buy-the-Dip Overlay strategy to configure dip parameters.")

with st.expander("Strategy parameter defaults (optional)", expanded=False):
    strategy_profiles = _profiles_for("strategy_defaults")
    if strategy_profiles:
        selection, changed = _profile_selectbox(
            "strategy_defaults",
            strategy_dotted,
            strategy_profiles,
            "Strategy defaults profile",
            default_selection="Classic ATR Breakout",
        )
        profile_raw = strategy_profiles.get(selection, {})
        profile_data, profile_description = _split_profile_payload(profile_raw)
        _maybe_apply_profile(
            "strategy_defaults",
            strategy_dotted,
            selection,
            changed,
            lambda data=profile_data: _apply_strategy_profile(data, base),
        )
        if selection != "Custom":
            if profile_description:
                st.caption(profile_description)
            st.caption(f"Profile '{selection}' loaded for base strategy parameters.")

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
            0,
            100,
            base["sma_fast"],
            1,
            help="Fast MA length (if trend filter used).",
        )
        base["sma_slow"] = st.number_input(
            "sma_slow",
            0,
            200,
            base["sma_slow"],
            1,
            help="Slow MA length (if trend filter used).",
        )
        base["sma_long"] = st.number_input(
            "sma_long",
            0,
            400,
            base["sma_long"],
            1,
            help="Long MA length (if trend filter used).",
        )
        base["long_slope_len"] = st.number_input(
            "long_slope_len",
            0,
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

    # Dip defaults are now managed in the Buy-the-Dip Parameters section above.

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

st.markdown("### Holdout diagnostics")
holdout_equity_placeholder = st.empty()
holdout_heatmap_placeholder = st.empty()
holdout_status_placeholder = st.empty()

st.markdown("### Trade lifecycle timeline")
holdout_timeline_placeholder = st.empty()

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

# ---------- Results hydration & status ----------
equity_series_state = st.session_state.get(HOLDOUT_EQUITY_KEY)
benchmark_series_state = st.session_state.get(HOLDOUT_BENCHMARK_KEY)
returns_series_state = st.session_state.get(HOLDOUT_RETURNS_KEY)
trades_df_state = st.session_state.get(HOLDOUT_TRADES_KEY)

x_axis_range = _resolve_holdout_x_range(equity_series_state, returns_series_state, trades_df_state)

_render_holdout_equity_chart(
    holdout_equity_placeholder,
    equity_series_state,
    benchmark_series_state,
    x_axis_range,
)
_render_holdout_heatmap(
    holdout_heatmap_placeholder,
    returns_series_state,
    x_axis_range,
)
_render_trade_timeline(
    holdout_timeline_placeholder,
    trades_df_state,
    x_range=x_axis_range,
)

holdout_status_state = st.session_state.get("adapter_holdout_status") or (
    "info",
    "Holdout diagnostics will appear once a best candidate evaluates on the holdout window.",
)
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

    st.session_state[HOLDOUT_EQUITY_KEY] = None
    st.session_state[HOLDOUT_RETURNS_KEY] = None
    st.session_state[HOLDOUT_BENCHMARK_KEY] = None
    st.session_state[HOLDOUT_TRADES_KEY] = None

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

    st.session_state["_hc_last_score"] = None

    def _load_benchmark_equity(target_index: pd.DatetimeIndex) -> pd.Series | None:
        try:
            bench = get_benchmark_total_return(start=holdout_start, end=holdout_end)
        except Exception:
            bench = None
        if bench is None or bench.empty:
            return None
        bench = _ensure_dt_index(bench.astype(float))
        bench = bench.reindex(target_index.union(bench.index)).ffill()
        bench = bench.reindex(target_index).ffill().dropna()
        if bench.empty:
            return None
        bench_equity = bench * float(equity)
        bench_equity.name = "benchmark_equity"
        return bench_equity

    def _update_holdout_visuals(gen_idx: int, best_score_val: float, params: Dict[str, Any]) -> None:
        try:
            result = _portfolio_equity_curve(
                strategy_dotted,
                tickers,
                holdout_start,
                holdout_end,
                float(equity),
                params,
            )
        except Exception:
            return

        equity_series = getattr(result, "equity", pd.Series(dtype=float))
        if equity_series is None or equity_series.empty:
            return

        equity_series = _ensure_dt_index(equity_series.astype(float))
        returns_series = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        benchmark_series = _load_benchmark_equity(equity_series.index)
        trades_df = _flatten_trades(getattr(result, "trades", {}))

        st.session_state[HOLDOUT_EQUITY_KEY] = equity_series
        st.session_state[HOLDOUT_RETURNS_KEY] = returns_series
        st.session_state[HOLDOUT_BENCHMARK_KEY] = benchmark_series
        st.session_state[HOLDOUT_TRADES_KEY] = trades_df

        status_suffix = f" (Gen {gen_idx}, score={best_score_val:.3f})" if isinstance(best_score_val, (int, float)) else ""
        st.session_state["adapter_holdout_status"] = (
            "success",
            f"Holdout window {holdout_start.date().isoformat()} → {holdout_end.date().isoformat()}{status_suffix}",
        )

        _render_holdout_equity_chart(holdout_equity_placeholder, equity_series, benchmark_series)
        _render_holdout_heatmap(holdout_heatmap_placeholder, returns_series)
        _render_trade_timeline(holdout_timeline_placeholder, trades_df)

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

    param_space.update(
        {
            "vol_target_target_pct": _clamp_float(
                cfg["vol_target_target_pct_min"], cfg["vol_target_target_pct_max"]
            ),
            "vol_target_atr_window": _clamp_int(
                cfg["vol_target_atr_window_min"], cfg["vol_target_atr_window_max"]
            ),
            "vol_target_min_leverage": _clamp_float(
                cfg["vol_target_min_leverage_min"], cfg["vol_target_min_leverage_max"]
            ),
            "vol_target_max_leverage": _clamp_float(
                cfg["vol_target_max_leverage_min"], cfg["vol_target_max_leverage_max"]
            ),
            "trend_filter_ma": _clamp_int(
                cfg["trend_filter_ma_min"], cfg["trend_filter_ma_max"]
            ),
            "trend_filter_slope_lookback": _clamp_int(
                cfg["trend_filter_slope_lookback_min"], cfg["trend_filter_slope_lookback_max"]
            ),
            "trend_filter_slope_threshold": _clamp_float(
                cfg["trend_filter_slope_threshold_min"], cfg["trend_filter_slope_threshold_max"]
            ),
            "trend_filter_exit": _clamp_int(
                cfg["trend_filter_exit_min"], cfg["trend_filter_exit_max"]
            ),
        }
    )

    if dip_strategy:
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

    ea_config_payload = {
        "pop_size": int(cfg.get("pop_size", 64)),
        "generations": int(cfg.get("generations", 50)),
        "selection_method": str(cfg.get("selection_method", "tournament")),
        "tournament_k": max(2, int(cfg.get("tournament_k", 3) or 3)),
        "replacement": str(cfg.get("replacement", "mu+lambda")),
        "elitism_fraction": float(cfg.get("elitism_fraction", 0.05)),
        "crossover_rate": float(cfg.get("crossover_rate", 0.85)),
        "crossover_op": str(cfg.get("crossover_op", "blend")),
        "mutation_rate": float(cfg.get("mutation_rate", 0.10)),
        "mutation_scale": float(cfg.get("mutation_scale", 0.20)),
        "mutation_scheme": str(cfg.get("mutation_scheme", "gaussian")),
        "genewise_clip": bool(cfg.get("genewise_clip", True)),
        "anneal_mutation": bool(cfg.get("anneal_mutation", True)),
        "anneal_floor": float(cfg.get("anneal_floor", 0.05)),
        "fitness_patience": int(cfg.get("fitness_patience", 8)),
        "no_improve_tol": None,
        "seed": cfg.get("seed"),
        "workers": cfg.get("workers"),
        "shuffle_eval": bool(cfg.get("shuffle_eval", True)),
    }

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
                    _update_holdout_visuals(
                        int(ctx.get("gen", 0)),
                        float(best_score_gen),
                        dict(best_params_gen),
                    )
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
        TrainingLogger(log_file, tags={"model_key": strategy_dotted}).log(
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
            test_start=holdout_start,
            test_end=holdout_end,
            min_trades=int(cfg["min_trades"]),
            n_jobs=int(n_jobs),
            progress_cb=_cb,
            log_file=log_file,
            config=ea_config_payload,
        )
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    if not top:
        st.error("EA returned no candidates.")
        st.stop()

    if dip_strategy:
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
    leaderboard_keys = [
        "breakout_n",
        "exit_n",
        "atr_n",
        "atr_multiple",
        "tp_multiple",
        "holding_period_limit",
    ]
    if dip_strategy:
        leaderboard_keys.extend(
            [
                "trend_ma",
                "dip_atr_from_high",
                "dip_lookback_high",
                "dip_rsi_max",
                "dip_confirm",
                "dip_cooldown_days",
                "entry_mode",
            ]
        )
    for params, score in top[: min(50, len(top))]:
        r = {"score": float(score)}
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
