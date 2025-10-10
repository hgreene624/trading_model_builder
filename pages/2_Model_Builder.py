# pages/2_Model_Builder.py
from __future__ import annotations

import json
import math
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

from src.data.portfolio_prefetch import intersection_range
from src.models._warmup import apply_disable_warmup_flag
from src.utils.training_logger import TrainingLogger
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
        "use_risk_reward_sizing": False,
        "risk_reward_min_scale": 0.5,
        "risk_reward_max_scale": 1.5,
        "risk_reward_target": 1.5,
        "risk_reward_sensitivity": 0.5,
        "risk_reward_fallback": 1.0,
        "size_mode": "rr_portfolio",
        "size_base_fraction": 0.005,
        "size_rr_slope": 0.003,
        "size_min_fraction": 0.001,
        "size_rr_cap_fraction": 0.05,
        "rr_floor": 0.5,
        "leverage_cap": 1.0,
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
        "use_risk_reward_sizing": False,
        "risk_reward_min_scale": 0.5,
        "risk_reward_max_scale": 1.5,
        "risk_reward_target": 1.5,
        "risk_reward_sensitivity": 0.5,
        "risk_reward_fallback": 1.0,
        "size_mode": "rr_portfolio",
        "size_base_fraction": 0.005,
        "size_rr_slope": 0.003,
        "size_min_fraction": 0.001,
        "size_rr_cap_fraction": 0.05,
        "rr_floor": 0.5,
        "leverage_cap": 1.0,
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
    "risk_reward_min_scale": (0.0, 1.0),
    "risk_reward_max_scale": (0.5, 3.0),
    "risk_reward_target": (0.5, 5.0),
    "risk_reward_sensitivity": (0.0, 2.0),
    "risk_reward_fallback": (0.5, 5.0),
    "size_base_fraction": (0.0, 0.1),
    "size_rr_slope": (0.0, 0.05),
    "size_min_fraction": (0.0, 0.05),
    "size_rr_cap_fraction": (0.005, 0.5),
    "rr_floor": (0.0, 5.0),
    "leverage_cap": (1.0, 2.0),
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
    "risk_per_trade_min": (0.0005, 0.05),
    "risk_per_trade_max": (0.0005, 0.05),
    "use_risk_reward_sizing_min": (0, 1),
    "use_risk_reward_sizing_max": (0, 1),
    "risk_reward_min_scale_min": (0.0, 1.0),
    "risk_reward_min_scale_max": (0.0, 1.5),
    "risk_reward_max_scale_min": (0.5, 3.0),
    "risk_reward_max_scale_max": (0.5, 4.0),
    "risk_reward_target_min": (0.5, 3.0),
    "risk_reward_target_max": (0.5, 6.0),
    "risk_reward_sensitivity_min": (0.0, 2.0),
    "risk_reward_sensitivity_max": (0.0, 2.0),
    "risk_reward_fallback_min": (0.5, 3.0),
    "risk_reward_fallback_max": (0.5, 6.0),
    "size_base_fraction_min": (0.0, 0.02),
    "size_base_fraction_max": (0.0, 0.08),
    "size_rr_slope_min": (0.0, 0.01),
    "size_rr_slope_max": (0.0, 0.03),
    "size_min_fraction_min": (0.0, 0.01),
    "size_min_fraction_max": (0.0, 0.03),
    "size_rr_cap_fraction_min": (0.005, 0.08),
    "size_rr_cap_fraction_max": (0.01, 0.2),
    "rr_floor_min": (0.0, 1.5),
    "rr_floor_max": (0.0, 3.0),
    "leverage_cap_min": (1.0, 1.5),
    "leverage_cap_max": (1.0, 2.5),
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
    ("risk_per_trade_min", "risk_per_trade_max"),
    ("use_risk_reward_sizing_min", "use_risk_reward_sizing_max"),
    ("risk_reward_min_scale_min", "risk_reward_min_scale_max"),
    ("risk_reward_max_scale_min", "risk_reward_max_scale_max"),
    ("risk_reward_target_min", "risk_reward_target_max"),
    ("risk_reward_sensitivity_min", "risk_reward_sensitivity_max"),
    ("risk_reward_fallback_min", "risk_reward_fallback_max"),
    ("size_base_fraction_min", "size_base_fraction_max"),
    ("size_rr_slope_min", "size_rr_slope_max"),
    ("size_min_fraction_min", "size_min_fraction_max"),
    ("size_rr_cap_fraction_min", "size_rr_cap_fraction_max"),
    ("rr_floor_min", "rr_floor_max"),
    ("leverage_cap_min", "leverage_cap_max"),
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
    "risk_per_trade_min": 0.002,
    "risk_per_trade_max": 0.02,
    "use_risk_reward_sizing_min": 0,
    "use_risk_reward_sizing_max": 1,
    "risk_reward_min_scale_min": 0.4,
    "risk_reward_min_scale_max": 1.0,
    "risk_reward_max_scale_min": 1.0,
    "risk_reward_max_scale_max": 2.5,
    "risk_reward_target_min": 1.0,
    "risk_reward_target_max": 3.0,
    "risk_reward_sensitivity_min": 0.0,
    "risk_reward_sensitivity_max": 1.0,
    "risk_reward_fallback_min": 1.0,
    "risk_reward_fallback_max": 3.0,
    "size_base_fraction_min": 0.002,
    "size_base_fraction_max": 0.015,
    "size_rr_slope_min": 0.0005,
    "size_rr_slope_max": 0.006,
    "size_min_fraction_min": 0.0005,
    "size_min_fraction_max": 0.004,
    "size_rr_cap_fraction_min": 0.02,
    "size_rr_cap_fraction_max": 0.08,
    "rr_floor_min": 0.5,
    "rr_floor_max": 1.5,
    "leverage_cap_min": 1.0,
    "leverage_cap_max": 1.4,
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
        "risk_per_trade_min": ("risk_per_trade_min", float),
        "risk_per_trade_max": ("risk_per_trade_max", float),
        "use_risk_reward_sizing_min": ("use_risk_reward_sizing_min", int),
        "use_risk_reward_sizing_max": ("use_risk_reward_sizing_max", int),
        "risk_reward_min_scale_min": ("risk_reward_min_scale_min", float),
        "risk_reward_min_scale_max": ("risk_reward_min_scale_max", float),
        "risk_reward_max_scale_min": ("risk_reward_max_scale_min", float),
        "risk_reward_max_scale_max": ("risk_reward_max_scale_max", float),
        "risk_reward_target_min": ("risk_reward_target_min", float),
        "risk_reward_target_max": ("risk_reward_target_max", float),
        "risk_reward_sensitivity_min": ("risk_reward_sensitivity_min", float),
        "risk_reward_sensitivity_max": ("risk_reward_sensitivity_max", float),
        "risk_reward_fallback_min": ("risk_reward_fallback_min", float),
        "risk_reward_fallback_max": ("risk_reward_fallback_max", float),
        "size_base_fraction_min": ("size_base_fraction_min", float),
        "size_base_fraction_max": ("size_base_fraction_max", float),
        "size_rr_slope_min": ("size_rr_slope_min", float),
        "size_rr_slope_max": ("size_rr_slope_max", float),
        "size_min_fraction_min": ("size_min_fraction_min", float),
        "size_min_fraction_max": ("size_min_fraction_max", float),
        "size_rr_cap_fraction_min": ("size_rr_cap_fraction_min", float),
        "size_rr_cap_fraction_max": ("size_rr_cap_fraction_max", float),
        "rr_floor_min": ("rr_floor_min", float),
        "rr_floor_max": ("rr_floor_max", float),
        "leverage_cap_min": ("leverage_cap_min", float),
        "leverage_cap_max": ("leverage_cap_max", float),
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
        "use_risk_reward_sizing": ("use_risk_reward_sizing", bool),
        "risk_reward_min_scale": ("risk_reward_min_scale", float),
        "risk_reward_max_scale": ("risk_reward_max_scale", float),
        "risk_reward_target": ("risk_reward_target", float),
        "risk_reward_sensitivity": ("risk_reward_sensitivity", float),
        "risk_reward_fallback": ("risk_reward_fallback", float),
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
    capital_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    capital_allocations: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_allocation: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    capital_weight_requests: pd.DataFrame = field(default_factory=pd.DataFrame)
    capital_allocation_requests: pd.DataFrame = field(default_factory=pd.DataFrame)
    leverage_cap: float = 1.0


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
                    "portfolio_weight_fraction": (
                        float(trade.get("portfolio_weight_fraction"))
                        if trade.get("portfolio_weight_fraction") is not None
                        else np.nan
                    ),
                    "portfolio_notional_entry": (
                        float(trade.get("portfolio_notional_entry"))
                        if trade.get("portfolio_notional_entry") is not None
                        else np.nan
                    ),
                    "portfolio_requested_notional": (
                        float(trade.get("portfolio_requested_notional"))
                        if trade.get("portfolio_requested_notional") is not None
                        else np.nan
                    ),
                    "portfolio_requested_fraction": (
                        float(trade.get("portfolio_requested_fraction"))
                        if trade.get("portfolio_requested_fraction") is not None
                        else np.nan
                    ),
                    "requested_notional_entry": (
                        float(trade.get("requested_notional_entry"))
                        if trade.get("requested_notional_entry") is not None
                        else np.nan
                    ),
                    "requested_weight_fraction": (
                        float(trade.get("requested_weight_fraction"))
                        if trade.get("requested_weight_fraction") is not None
                        else np.nan
                    ),
                    "entry_equity_snapshot": (
                        float(trade.get("entry_equity_snapshot"))
                        if trade.get("entry_equity_snapshot") is not None
                        else np.nan
                    ),
                    "risk_reward_ratio": (
                        float(trade.get("risk_reward_ratio"))
                        if trade.get("risk_reward_ratio") is not None
                        else np.nan
                    ),
                    "risk_reward_scale": (
                        float(trade.get("risk_reward_scale"))
                        if trade.get("risk_reward_scale") is not None
                        else np.nan
                    ),
                    "risk_reward_sizing": bool(trade.get("risk_reward_sizing", False)),
                    "risk_per_share": (
                        float(trade.get("risk_per_share"))
                        if trade.get("risk_per_share") is not None
                        else np.nan
                    ),
                    "reward_per_share": (
                        float(trade.get("reward_per_share"))
                        if trade.get("reward_per_share") is not None
                        else np.nan
                    ),
                    "volatility_scale": (
                        float(trade.get("volatility_scale"))
                        if trade.get("volatility_scale") is not None
                        else np.nan
                    ),
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
    if "portfolio_notional_entry" in df.columns and df["portfolio_notional_entry"].notna().any():
        df["abs_notional"] = df["portfolio_notional_entry"].abs().fillna(0.0)
    else:
        df["abs_notional"] = df["notional_entry"].abs()
    if "portfolio_requested_notional" in df.columns:
        zero_mask = df["abs_notional"] <= 0.0
        if zero_mask.any():
            df.loc[zero_mask, "abs_notional"] = df.loc[zero_mask, "portfolio_requested_notional"].abs().fillna(0.0)
    max_notional = df["abs_notional"].max()
    if max_notional and max_notional > 0:
        df["size_fraction"] = df["abs_notional"] / max_notional
    else:
        df["size_fraction"] = 0.0
    df["size_fraction_display"] = df["size_fraction"].clip(0.0, 1.0)
    df["bar_width"] = 0.35 + 0.55 * df["size_fraction_display"]

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
            "notional_entry": ":.2f",
            "portfolio_notional_entry": ":.2f",
            "portfolio_weight_fraction": ":.2f",
            "portfolio_requested_notional": ":.2f",
            "portfolio_requested_fraction": ":.2f",
            "requested_notional_entry": ":.2f",
            "requested_weight_fraction": ":.2f",
            "entry_equity_snapshot": ":.2f",
            "risk_reward_ratio": ":.2f",
            "risk_reward_scale": ":.2f",
            "risk_reward_sizing": True,
            "size_fraction_display": ":.1%",
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
    if fig.data:
        widths = df["bar_width"].tolist()
        fig.update_traces(width=widths, selector=dict(type="bar"))
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
    params_template: Dict[str, Any] = {}
    if isinstance(params, dict):
        params_template = dict(params)
    params_template.setdefault("model_key", strategy_dotted)
    try:
        leverage_cap = float(params_template.get("leverage_cap", 1.0))
    except (TypeError, ValueError):
        leverage_cap = 1.0
    if not math.isfinite(leverage_cap) or leverage_cap < 1.0:
        leverage_cap = 1.0
    per_symbol_equity: Dict[str, pd.Series] = {}
    per_symbol_ratio: Dict[str, pd.Series] = {}
    trades_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    for sym in tickers:
        try:
            run_params = apply_disable_warmup_flag(
                params_template, disable_warmup=False
            )
            result = run(sym, start, end, starting_equity, run_params)
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

    ratio_df = ratio_df.astype(float)
    returns_df = (
        ratio_df.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    notional_request_df = pd.DataFrame(0.0, index=ratio_df.index, columns=ratio_df.columns, dtype=float)

    def _normalize_ts(value: Any) -> pd.Timestamp | None:
        ts = pd.to_datetime(value, errors="coerce")
        if ts is None or pd.isna(ts):
            return None
        if getattr(ts, "tzinfo", None) is not None:
            try:
                ts = ts.tz_convert(None)
            except Exception:
                ts = ts.tz_localize(None)
        return ts

    for sym, trade_list in trades_by_symbol.items():
        if sym not in notional_request_df.columns:
            continue
        series = per_symbol_equity.get(sym)
        if series is None or series.empty:
            continue
        for trade in trade_list or []:
            if not isinstance(trade, dict):
                continue
            entry_ts = _normalize_ts(trade.get("entry_time") or trade.get("entry_dt"))
            exit_ts = _normalize_ts(trade.get("exit_time") or trade.get("exit_dt"))
            if entry_ts is None or exit_ts is None:
                continue
            if exit_ts < entry_ts:
                entry_ts, exit_ts = exit_ts, entry_ts
            notional_candidates = [
                trade.get("requested_notional_entry"),
                trade.get("portfolio_requested_notional"),
                trade.get("portfolio_notional_entry"),
                trade.get("notional_entry"),
                trade.get("notional"),
            ]
            notional = 0.0
            for candidate in notional_candidates:
                try:
                    value = float(candidate)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                value = abs(value)
                if value > 0.0:
                    notional = value
                    break
            if notional <= 0.0:
                qty = float(trade.get("quantity") or trade.get("qty") or 0.0)
                price = float(trade.get("entry_price") or trade.get("decision_price") or 0.0)
                notional = abs(qty * price)
            if notional <= 0.0:
                continue
            mask = (notional_request_df.index >= entry_ts) & (notional_request_df.index <= exit_ts)
            if not mask.any():
                continue
            notional_request_df.loc[mask, sym] += notional

    returns_df = returns_df.reindex(notional_request_df.index).fillna(0.0)

    notional_request_df = (
        notional_request_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    weights_fraction = pd.DataFrame(
        0.0, index=notional_request_df.index, columns=notional_request_df.columns, dtype=float
    )
    cash_series = pd.Series(1.0, index=weights_fraction.index, dtype=float)
    gross_series = pd.Series(0.0, index=weights_fraction.index, dtype=float)
    allocations_df = pd.DataFrame(0.0, index=weights_fraction.index, columns=weights_fraction.columns, dtype=float)
    cash_notional_series = pd.Series(0.0, index=weights_fraction.index, dtype=float)
    requested_weights_df = pd.DataFrame(
        0.0, index=weights_fraction.index, columns=weights_fraction.columns, dtype=float
    )
    requested_cash_series = pd.Series(1.0, index=weights_fraction.index, dtype=float)
    requested_gross_series = pd.Series(0.0, index=weights_fraction.index, dtype=float)

    equity_values: List[float] = []
    current_equity = float(base_equity if math.isfinite(base_equity) else 0.0)
    if current_equity <= 0.0:
        try:
            first_symbol_equity = next(
                float(series.iloc[0])
                for series in per_symbol_equity.values()
                if isinstance(series, pd.Series) and len(series) > 0 and math.isfinite(float(series.iloc[0]))
            )
        except StopIteration:
            first_symbol_equity = 0.0
        current_equity = float(first_symbol_equity)
    if current_equity <= 0.0:
        current_equity = 1.0
    prev_weights = pd.Series(0.0, index=weights_fraction.columns, dtype=float)

    for idx, _ in enumerate(weights_fraction.index):
        if idx > 0:
            period_vector = returns_df.iloc[idx]
            try:
                period_return = float((period_vector * prev_weights).sum())
            except Exception:
                period_return = 0.0
            if not math.isfinite(period_return):
                period_return = 0.0
            current_equity *= 1.0 + period_return
        equity_values.append(current_equity)

        notional_row = notional_request_df.iloc[idx].clip(lower=0.0).fillna(0.0)
        equity_divisor = current_equity if current_equity > 0.0 else 0.0
        if equity_divisor > 0.0:
            requested_weights_row = (
                (notional_row / equity_divisor)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
        else:
            requested_weights_row = pd.Series(0.0, index=weights_fraction.columns, dtype=float)
        requested_weights_row = requested_weights_row.clip(lower=0.0)
        requested_weights_df.iloc[idx] = requested_weights_row
        requested_gross = float(requested_weights_row.sum())
        requested_gross_series.iloc[idx] = requested_gross
        requested_cash_series.iloc[idx] = float(1.0 - requested_gross)

        requested_total = float(notional_row.sum())
        available_capital = current_equity * leverage_cap if current_equity > 0.0 else 0.0
        if requested_total > 0.0 and available_capital > 0.0:
            if requested_total <= available_capital + 1e-12:
                funded_row = notional_row.copy()
            else:
                scale = available_capital / requested_total if requested_total > 0.0 else 0.0
                scale = max(0.0, min(1.0, scale))
                funded_row = notional_row * scale
        else:
            funded_row = pd.Series(0.0, index=weights_fraction.columns, dtype=float)

        if equity_divisor > 0.0:
            weights_row = (
                (funded_row / equity_divisor)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
        else:
            weights_row = pd.Series(0.0, index=weights_fraction.columns, dtype=float)
        weights_row = weights_row.clip(lower=0.0)
        weights_fraction.iloc[idx] = weights_row

        funded_total = float(funded_row.sum())
        gross = (funded_total / equity_divisor) if equity_divisor > 0.0 else 0.0
        gross_series.iloc[idx] = float(gross)
        cash_value = current_equity - funded_total
        min_cash_value = -current_equity * (leverage_cap - 1.0) if leverage_cap > 1.0 else 0.0
        if cash_value < min_cash_value:
            cash_value = float(min_cash_value)
        if not math.isfinite(cash_value):
            cash_value = 0.0
        cash_ratio = (cash_value / current_equity) if equity_divisor > 0.0 else 0.0
        cash_series.iloc[idx] = float(cash_ratio)
        allocations_df.iloc[idx] = funded_row
        cash_notional_series.iloc[idx] = float(cash_value)
        prev_weights = weights_row

    weights_effective = weights_fraction.shift(1)
    if not weights_effective.empty:
        weights_effective.iloc[0, :] = weights_fraction.iloc[0, :]
    weights_effective = weights_effective.fillna(0.0)

    allocations_effective = allocations_df.shift(1)
    if not allocations_effective.empty:
        allocations_effective.iloc[0, :] = allocations_df.iloc[0, :]
    allocations_effective = allocations_effective.fillna(0.0)

    cash_effective = cash_series.shift(1)
    if len(cash_effective):
        cash_effective.iloc[0] = cash_series.iloc[0]
    cash_effective = cash_effective.fillna(0.0)

    cash_notional_effective = cash_notional_series.shift(1)
    if len(cash_notional_effective):
        cash_notional_effective.iloc[0] = cash_notional_series.iloc[0]
    cash_notional_effective = cash_notional_effective.fillna(0.0)

    gross_effective = gross_series.shift(1)
    if len(gross_effective):
        gross_effective.iloc[0] = gross_series.iloc[0]
    gross_effective = gross_effective.fillna(0.0)

    weights_effective["__cash__"] = cash_effective.astype(float)
    weights_effective["__gross_exposure__"] = gross_effective.astype(float)

    allocations_effective["__cash__"] = cash_notional_effective.astype(float)

    weight_requests = requested_weights_df.copy()
    weight_requests["__cash__"] = requested_cash_series.astype(float)
    weight_requests["__gross_exposure__"] = requested_gross_series.astype(float)

    notional_request_df = notional_request_df.fillna(0.0)

    portfolio_equity = pd.Series(equity_values, index=weights_fraction.index, dtype=float)
    portfolio_equity.name = "portfolio_equity"

    ratio_divisor = base_equity if base_equity and math.isfinite(base_equity) else None
    if not ratio_divisor or ratio_divisor == 0.0:
        first_valid = next((float(v) for v in equity_values if math.isfinite(v) and v != 0.0), 1.0)
        ratio_divisor = first_valid if first_valid else 1.0
    portfolio_ratio = portfolio_equity / float(ratio_divisor)
    portfolio_ratio = portfolio_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(portfolio_ratio):
        first_ratio = float(portfolio_ratio.iloc[0])
        if not math.isfinite(first_ratio) or first_ratio <= 0.0:
            portfolio_ratio.iloc[0] = 1.0
    portfolio_ratio.name = "portfolio_ratio"

    def _lookup_portfolio_equity(ts: pd.Timestamp | None) -> float | None:
        if ts is None or portfolio_equity.empty:
            return None
        try:
            if ts in portfolio_equity.index:
                value = float(portfolio_equity.loc[ts])
                if math.isfinite(value):
                    return value
            mask = portfolio_equity.index <= ts
            if not mask.any():
                return None
            value = float(portfolio_equity.loc[mask].iloc[-1])
            if math.isfinite(value):
                return value
        except Exception:
            return None
        return None

    def _locate_allocation_index(ts: pd.Timestamp | None) -> pd.Timestamp | None:
        if ts is None or allocations_df.empty:
            return None
        idx = allocations_df.index
        try:
            if ts in idx:
                return ts
            pos = idx.searchsorted(ts)
            if 0 <= pos < len(idx):
                return idx[pos]
        except Exception:
            return None
        return None

    base_equity_fallback = float(base_equity if math.isfinite(base_equity) else 0.0)
    for sym, trade_list in trades_by_symbol.items():
        if not isinstance(trade_list, list):
            continue
        for trade in trade_list:
            if not isinstance(trade, dict):
                continue
            entry_ts = _normalize_ts(trade.get("entry_time") or trade.get("entry_dt"))
            entry_equity_portfolio = _lookup_portfolio_equity(entry_ts)
            if entry_equity_portfolio is None or entry_equity_portfolio <= 0.0:
                entry_equity_portfolio = base_equity_fallback
            if entry_equity_portfolio is None or entry_equity_portfolio <= 0.0:
                continue
            allocation_idx = _locate_allocation_index(entry_ts)
            if allocation_idx is None:
                continue
            try:
                funded_value = float(allocations_df.loc[allocation_idx, sym])
            except Exception:
                funded_value = 0.0
            try:
                requested_value = float(notional_request_df.loc[allocation_idx, sym])
            except Exception:
                requested_value = 0.0
            funded_value = abs(funded_value)
            requested_value = abs(requested_value)
            if funded_value <= 0.0 and requested_value <= 0.0:
                continue
            weight_fraction = (
                funded_value / entry_equity_portfolio
                if funded_value > 0.0 and entry_equity_portfolio > 0.0
                else 0.0
            )
            requested_fraction = (
                requested_value / entry_equity_portfolio
                if requested_value > 0.0 and entry_equity_portfolio > 0.0
                else 0.0
            )
            if math.isfinite(weight_fraction) and weight_fraction > 0.0:
                trade["portfolio_weight_fraction"] = float(weight_fraction)
            if math.isfinite(requested_fraction) and requested_fraction > 0.0:
                trade["portfolio_requested_fraction"] = float(requested_fraction)
                trade["requested_weight_fraction"] = float(requested_fraction)
            if funded_value > 0.0:
                trade["portfolio_notional_entry"] = float(funded_value)
            if requested_value > 0.0:
                trade["portfolio_requested_notional"] = float(requested_value)
                trade["requested_notional_entry"] = float(requested_value)

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
        capital_weights=weights_effective,
        capital_allocations=allocations_effective,
        cash_allocation=cash_effective,
        capital_weight_requests=weight_requests,
        capital_allocation_requests=notional_request_df,
        leverage_cap=float(leverage_cap),
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

    risk_bound_cols = st.columns(3)
    with risk_bound_cols[0]:
        rptr_lo = st.number_input(
            "risk_per_trade min",
            0.0005,
            0.5,
            float(ea_cfg.get("risk_per_trade_min", 0.002)),
            0.0005,
            format="%.4f",
            help="Lower bound for fraction of equity risked per trade during EA runs.",
        )
        rptr_hi = st.number_input(
            "risk_per_trade max",
            float(rptr_lo),
            0.5,
            float(ea_cfg.get("risk_per_trade_max", 0.02)),
            0.0005,
            format="%.4f",
            help="Upper bound for fraction of equity risked per trade during EA runs.",
        )
        use_min_default = int(ea_cfg.get("use_risk_reward_sizing_min", 0))
        use_max_default = int(ea_cfg.get("use_risk_reward_sizing_max", 1))
        use_options = [0, 1]
        use_min_idx = use_options.index(use_min_default) if use_min_default in use_options else 0
        use_max_idx = use_options.index(use_max_default) if use_max_default in use_options else 1
        use_min = st.selectbox(
            "risk/reward sizing min",
            options=use_options,
            index=use_min_idx,
            help="Minimum boolean (0=off,1=on) allowed for risk/reward sizing in the EA genome.",
            format_func=lambda x: f"{x} ({'on' if x else 'off'})",
        )
        use_max = st.selectbox(
            "risk/reward sizing max",
            options=use_options,
            index=use_max_idx,
            help="Maximum boolean (0=off,1=on) allowed for risk/reward sizing in the EA genome.",
            format_func=lambda x: f"{x} ({'on' if x else 'off'})",
        )
        if use_max < use_min:
            use_max = use_min
    with risk_bound_cols[1]:
        rrmin_lo = st.number_input(
            "rr_min_scale min",
            0.0,
            2.0,
            float(ea_cfg.get("risk_reward_min_scale_min", 0.4)),
            0.1,
            help="Smallest multiplier clamp when sizing by reward/risk.",
        )
        rrmin_hi = st.number_input(
            "rr_min_scale max",
            float(rrmin_lo),
            2.0,
            float(ea_cfg.get("risk_reward_min_scale_max", 1.0)),
            0.1,
            help="Largest multiplier clamp for the lower bound.",
        )
        rrmax_lo = st.number_input(
            "rr_max_scale min",
            0.5,
            5.0,
            float(ea_cfg.get("risk_reward_max_scale_min", 1.0)),
            0.1,
            help="Minimum multiplier allowed for the upper clamp.",
        )
    with risk_bound_cols[2]:
        rrmax_hi = st.number_input(
            "rr_max_scale max",
            float(rrmax_lo),
            5.0,
            float(ea_cfg.get("risk_reward_max_scale_max", 2.5)),
            0.1,
            help="Maximum multiplier allowed for the upper clamp.",
        )
        target_lo = st.number_input(
            "rr_target min",
            0.1,
            10.0,
            float(ea_cfg.get("risk_reward_target_min", 1.0)),
            0.1,
            help="Smallest reward/risk ratio the EA can treat as neutral.",
        )
        target_hi = st.number_input(
            "rr_target max",
            float(target_lo),
            10.0,
            float(ea_cfg.get("risk_reward_target_max", 3.0)),
            0.1,
            help="Largest reward/risk ratio the EA can treat as neutral.",
        )

    rr_cols_secondary = st.columns(3)
    with rr_cols_secondary[0]:
        sens_lo = st.number_input(
            "rr_sensitivity min",
            0.0,
            3.0,
            float(ea_cfg.get("risk_reward_sensitivity_min", 0.0)),
            0.05,
            help="Lower bound on multiplier slope when scaling by reward/risk.",
        )
        sens_hi = st.number_input(
            "rr_sensitivity max",
            float(sens_lo),
            3.0,
            float(ea_cfg.get("risk_reward_sensitivity_max", 1.0)),
            0.05,
            help="Upper bound on multiplier slope when scaling by reward/risk.",
        )
    with rr_cols_secondary[1]:
        fallback_lo = st.number_input(
            "rr_fallback min",
            0.1,
            10.0,
            float(ea_cfg.get("risk_reward_fallback_min", 1.0)),
            0.1,
            help="Minimum assumed reward/risk ratio when no explicit target exists.",
        )
        fallback_hi = st.number_input(
            "rr_fallback max",
            float(fallback_lo),
            10.0,
            float(ea_cfg.get("risk_reward_fallback_max", 3.0)),
            0.1,
            help="Maximum assumed reward/risk ratio when no explicit target exists.",
        )
    with rr_cols_secondary[2]:
        st.markdown(
            """<div style='height:2.2em'></div>""",
            unsafe_allow_html=True,
        )

    size_bounds_primary = st.columns(3)
    with size_bounds_primary[0]:
        sbf_lo = st.number_input(
            "size_base_fraction min",
            float(EA_PARAM_RANGES["size_base_fraction_min"][0]),
            float(EA_PARAM_RANGES["size_base_fraction_min"][1]),
            float(ea_cfg.get("size_base_fraction_min", 0.002)),
            0.0005,
            format="%.4f",
            help="Minimum base capital fraction the EA can request per trade.",
        )
        sbf_hi = st.number_input(
            "size_base_fraction max",
            float(sbf_lo),
            float(EA_PARAM_RANGES["size_base_fraction_max"][1]),
            float(ea_cfg.get("size_base_fraction_max", 0.015)),
            0.0005,
            format="%.4f",
            help="Maximum base capital fraction the EA can request per trade.",
        )
    with size_bounds_primary[1]:
        slope_lo = st.number_input(
            "size_rr_slope min",
            float(EA_PARAM_RANGES["size_rr_slope_min"][0]),
            float(EA_PARAM_RANGES["size_rr_slope_min"][1]),
            float(ea_cfg.get("size_rr_slope_min", 0.0005)),
            0.0005,
            format="%.4f",
            help="Lower bound on incremental sizing per unit reward/risk.",
        )
        slope_hi = st.number_input(
            "size_rr_slope max",
            float(max(slope_lo, EA_PARAM_RANGES["size_rr_slope_max"][0])),
            float(EA_PARAM_RANGES["size_rr_slope_max"][1]),
            float(ea_cfg.get("size_rr_slope_max", 0.006)),
            0.0005,
            format="%.4f",
            help="Upper bound on incremental sizing per unit reward/risk.",
        )
    with size_bounds_primary[2]:
        minfrac_lo = st.number_input(
            "size_min_fraction min",
            float(EA_PARAM_RANGES["size_min_fraction_min"][0]),
            float(EA_PARAM_RANGES["size_min_fraction_min"][1]),
            float(ea_cfg.get("size_min_fraction_min", 0.0005)),
            0.0005,
            format="%.4f",
            help="Smallest minimum capital fraction the EA can enforce.",
        )
        minfrac_hi = st.number_input(
            "size_min_fraction max",
            float(max(minfrac_lo, EA_PARAM_RANGES["size_min_fraction_max"][0])),
            float(EA_PARAM_RANGES["size_min_fraction_max"][1]),
            float(ea_cfg.get("size_min_fraction_max", 0.004)),
            0.0005,
            format="%.4f",
            help="Largest minimum capital fraction the EA can enforce.",
        )

    size_bounds_secondary = st.columns(3)
    with size_bounds_secondary[0]:
        cap_lo = st.number_input(
            "size_rr_cap_fraction min",
            float(EA_PARAM_RANGES["size_rr_cap_fraction_min"][0]),
            float(EA_PARAM_RANGES["size_rr_cap_fraction_min"][1]),
            float(ea_cfg.get("size_rr_cap_fraction_min", 0.02)),
            0.001,
            format="%.4f",
            help="Minimum per-trade cap fraction allowed in the EA search.",
        )
        cap_hi = st.number_input(
            "size_rr_cap_fraction max",
            float(max(cap_lo, EA_PARAM_RANGES["size_rr_cap_fraction_max"][0])),
            float(EA_PARAM_RANGES["size_rr_cap_fraction_max"][1]),
            float(ea_cfg.get("size_rr_cap_fraction_max", 0.08)),
            0.001,
            format="%.4f",
            help="Maximum per-trade cap fraction allowed in the EA search.",
        )
    with size_bounds_secondary[1]:
        floor_lo = st.number_input(
            "rr_floor min",
            float(EA_PARAM_RANGES["rr_floor_min"][0]),
            float(EA_PARAM_RANGES["rr_floor_min"][1]),
            float(ea_cfg.get("rr_floor_min", 0.5)),
            0.1,
            help="Smallest reward/risk floor considered by the EA.",
        )
        floor_hi = st.number_input(
            "rr_floor max",
            float(max(floor_lo, EA_PARAM_RANGES["rr_floor_max"][0])),
            float(EA_PARAM_RANGES["rr_floor_max"][1]),
            float(ea_cfg.get("rr_floor_max", 1.5)),
            0.1,
            help="Largest reward/risk floor considered by the EA.",
        )
    with size_bounds_secondary[2]:
        lev_lo = st.number_input(
            "leverage_cap min",
            float(EA_PARAM_RANGES["leverage_cap_min"][0]),
            float(EA_PARAM_RANGES["leverage_cap_min"][1]),
            float(ea_cfg.get("leverage_cap_min", 1.0)),
            0.05,
            help="Minimum portfolio leverage the EA can explore.",
        )
        lev_hi = st.number_input(
            "leverage_cap max",
            float(max(lev_lo, EA_PARAM_RANGES["leverage_cap_max"][0])),
            float(EA_PARAM_RANGES["leverage_cap_max"][1]),
            float(ea_cfg.get("leverage_cap_max", 1.4)),
            0.05,
            help="Maximum portfolio leverage the EA can explore.",
        )

    ea_cfg["risk_per_trade_min"], ea_cfg["risk_per_trade_max"] = float(rptr_lo), float(rptr_hi)
    ea_cfg["use_risk_reward_sizing_min"], ea_cfg["use_risk_reward_sizing_max"] = int(use_min), int(use_max)
    ea_cfg["risk_reward_min_scale_min"], ea_cfg["risk_reward_min_scale_max"] = float(rrmin_lo), float(rrmin_hi)
    ea_cfg["risk_reward_max_scale_min"], ea_cfg["risk_reward_max_scale_max"] = float(rrmax_lo), float(rrmax_hi)
    ea_cfg["risk_reward_target_min"], ea_cfg["risk_reward_target_max"] = float(target_lo), float(target_hi)
    ea_cfg["risk_reward_sensitivity_min"], ea_cfg["risk_reward_sensitivity_max"] = float(sens_lo), float(sens_hi)
    ea_cfg["risk_reward_fallback_min"], ea_cfg["risk_reward_fallback_max"] = float(fallback_lo), float(fallback_hi)
    ea_cfg["size_base_fraction_min"], ea_cfg["size_base_fraction_max"] = float(sbf_lo), float(sbf_hi)
    ea_cfg["size_rr_slope_min"], ea_cfg["size_rr_slope_max"] = float(slope_lo), float(slope_hi)
    ea_cfg["size_min_fraction_min"], ea_cfg["size_min_fraction_max"] = float(minfrac_lo), float(minfrac_hi)
    ea_cfg["size_rr_cap_fraction_min"], ea_cfg["size_rr_cap_fraction_max"] = float(cap_lo), float(cap_hi)
    ea_cfg["rr_floor_min"], ea_cfg["rr_floor_max"] = float(floor_lo), float(floor_hi)
    ea_cfg["leverage_cap_min"], ea_cfg["leverage_cap_max"] = float(lev_lo), float(lev_hi)

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
        base["use_risk_reward_sizing"] = st.checkbox(
            "use_risk_reward_sizing",
            value=bool(base.get("use_risk_reward_sizing", False)),
            help="Enable ATR-based risk sizing that scales position size by the trade's reward/risk ratio.",
        )
        rr_disabled = not base["use_risk_reward_sizing"]
        base["risk_reward_min_scale"] = st.number_input(
            "risk_reward_min_scale",
            0.0,
            5.0,
            float(base.get("risk_reward_min_scale", 0.5)),
            0.1,
            help="Lower clamp for the risk/reward sizing multiplier.",
            disabled=rr_disabled,
        )
        base["risk_reward_max_scale"] = st.number_input(
            "risk_reward_max_scale",
            0.0,
            5.0,
            float(base.get("risk_reward_max_scale", 1.5)),
            0.1,
            help="Upper clamp for the risk/reward sizing multiplier.",
            disabled=rr_disabled,
        )
        base["risk_reward_target"] = st.number_input(
            "risk_reward_target",
            0.1,
            10.0,
            float(base.get("risk_reward_target", 1.5)),
            0.1,
            help="Reward/risk ratio considered neutral (multiplier=1).",
            disabled=rr_disabled,
        )
        base["risk_reward_sensitivity"] = st.number_input(
            "risk_reward_sensitivity",
            0.0,
            3.0,
            float(base.get("risk_reward_sensitivity", 0.5)),
            0.1,
            help="Multiplier slope per unit change in reward/risk vs target.",
            disabled=rr_disabled,
        )
        base["risk_reward_fallback"] = st.number_input(
            "risk_reward_fallback",
            0.1,
            10.0,
            float(base.get("risk_reward_fallback", 1.0)),
            0.1,
            help="Assumed reward/risk ratio when no explicit take-profit is defined.",
            disabled=rr_disabled,
        )
        size_mode_options = ["rr_portfolio", "legacy"]
        current_mode = str(base.get("size_mode", "rr_portfolio"))
        if current_mode not in size_mode_options:
            current_mode = "rr_portfolio"
        base["size_mode"] = st.selectbox(
            "size_mode",
            options=size_mode_options,
            index=size_mode_options.index(current_mode),
            help="Choose between portfolio-aware sizing (rr_portfolio) and legacy per-symbol sizing.",
        )
        base["size_base_fraction"] = st.number_input(
            "size_base_fraction",
            float(BASE_PARAM_RANGES["size_base_fraction"][0]),
            float(BASE_PARAM_RANGES["size_base_fraction"][1]),
            float(base.get("size_base_fraction", 0.005)),
            0.0005,
            format="%.4f",
            help="Baseline fraction of equity requested for each new trade.",
        )
        base["size_rr_slope"] = st.number_input(
            "size_rr_slope",
            float(BASE_PARAM_RANGES["size_rr_slope"][0]),
            float(BASE_PARAM_RANGES["size_rr_slope"][1]),
            float(base.get("size_rr_slope", 0.003)),
            0.0005,
            format="%.4f",
            help="Incremental requested fraction per +1.0 of reward/risk.",
        )
        base["size_min_fraction"] = st.number_input(
            "size_min_fraction",
            float(BASE_PARAM_RANGES["size_min_fraction"][0]),
            float(BASE_PARAM_RANGES["size_min_fraction"][1]),
            float(base.get("size_min_fraction", 0.001)),
            0.0005,
            format="%.4f",
            help="Minimum fraction of equity to request even for weak reward/risk setups.",
        )
        base["size_rr_cap_fraction"] = st.number_input(
            "size_rr_cap_fraction",
            float(BASE_PARAM_RANGES["size_rr_cap_fraction"][0]),
            float(BASE_PARAM_RANGES["size_rr_cap_fraction"][1]),
            float(base.get("size_rr_cap_fraction", 0.05)),
            0.001,
            format="%.4f",
            help="Ceiling fraction of equity requested per trade.",
        )
        base["rr_floor"] = st.number_input(
            "rr_floor",
            float(BASE_PARAM_RANGES["rr_floor"][0]),
            float(BASE_PARAM_RANGES["rr_floor"][1]),
            float(base.get("rr_floor", 0.5)),
            0.1,
            help="Reward/risk threshold below which minimum sizing is used.",
        )
        base["leverage_cap"] = st.number_input(
            "leverage_cap",
            float(BASE_PARAM_RANGES["leverage_cap"][0]),
            float(BASE_PARAM_RANGES["leverage_cap"][1]),
            float(base.get("leverage_cap", 1.0)),
            0.05,
            help="Maximum leverage the holdout allocator will use when funding concurrent trades.",
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
        "risk_per_trade": _clamp_float(cfg["risk_per_trade_min"], cfg["risk_per_trade_max"]),
        "use_risk_reward_sizing": _clamp_int(
            cfg["use_risk_reward_sizing_min"], cfg["use_risk_reward_sizing_max"]
        ),
        "risk_reward_min_scale": _clamp_float(
            cfg["risk_reward_min_scale_min"], cfg["risk_reward_min_scale_max"]
        ),
        "risk_reward_max_scale": _clamp_float(
            cfg["risk_reward_max_scale_min"], cfg["risk_reward_max_scale_max"]
        ),
        "risk_reward_target": _clamp_float(
            cfg["risk_reward_target_min"], cfg["risk_reward_target_max"]
        ),
        "risk_reward_sensitivity": _clamp_float(
            cfg["risk_reward_sensitivity_min"], cfg["risk_reward_sensitivity_max"]
        ),
        "risk_reward_fallback": _clamp_float(
            cfg["risk_reward_fallback_min"], cfg["risk_reward_fallback_max"]
        ),
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
                params_payload = ctx.get("resolved_params") or ctx.get("params") or {}
                if isinstance(params_payload, dict):
                    params_payload = dict(params_payload)
                else:
                    params_payload = {}
                if cur_best in (None, float("-inf")) or float(score) > float(cur_best):
                    gen_best["score"] = float(score)
                    gen_best["params"] = dict(params_payload)
                prev = best_tracker.get("score")
                if prev in (None, float("-inf")) or score > prev:
                    delta = None if prev in (None, float("-inf")) else float(score) - float(prev)
                    best_tracker.update({"score": float(score), "params": dict(params_payload), "delta": delta})
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
            best_params_gen = (
                ctx.get("top_params_resolved")
                or ctx.get("best_params")
                or gen_best.get("params")
                or dict(ctx.get("params") or {})
            )
            last_plotted = st.session_state.get("_hc_last_score")

            # Always push Gen 0; otherwise only if score improves
            should_push = (last_plotted is None) or (isinstance(best_score_gen, (int, float)) and best_score_gen > last_plotted)
            params_for_plot = dict(best_params_gen) if isinstance(best_params_gen, dict) else {}
            if should_push and isinstance(best_score_gen, (int, float)) and params_for_plot:
                try:
                    _update_holdout_visuals(
                        int(ctx.get("gen", 0)),
                        float(best_score_gen),
                        params_for_plot,
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
        "risk_per_trade",
        "use_risk_reward_sizing",
        "risk_reward_min_scale",
        "risk_reward_max_scale",
        "risk_reward_target",
        "risk_reward_sensitivity",
        "risk_reward_fallback",
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
