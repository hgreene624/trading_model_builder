# src/storage.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# --------------------------- roots & helpers ---------------------------

ROOT = Path.cwd() / "storage"  # project-local storage root
PORTFOLIOS_DIR = ROOT / "portfolios"
PORTFOLIO_MODELS_DIR = ROOT / "portfolio_models"
BASE_METRICS_DIR = ROOT / "base_metrics"
SIMULATIONS_DIR = ROOT / "simulations"

for p in (ROOT, PORTFOLIOS_DIR, PORTFOLIO_MODELS_DIR, BASE_METRICS_DIR, SIMULATIONS_DIR):
    p.mkdir(parents=True, exist_ok=True)


def _json_default(o: Any):
    """Safe JSON conversion for dataclasses & numpy-ish types."""
    try:
        import numpy as np  # local import in case not installed yet

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass

    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (set,)):
        return list(o)
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    return str(o)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _now_stamp() -> str:
    # e.g. 2025-09-10T12-34-56
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


# --------------------------- portfolios --------------------------------

def _portfolio_path(name: str) -> Path:
    # Each portfolio is one JSON file: {"name":..., "tickers":[...], "meta":{...}}
    safe = name.strip().replace("/", "_")
    return PORTFOLIOS_DIR / f"{safe}.json"


def list_portfolios() -> List[str]:
    return sorted([p.stem for p in PORTFOLIOS_DIR.glob("*.json")])


def load_portfolio(name: str) -> Dict[str, Any]:
    return read_json(_portfolio_path(name), default={"name": name, "tickers": [], "meta": {}}) or {"name": name, "tickers": [], "meta": {}}


def save_portfolio(name: str, tickers: Iterable[str], meta: Optional[Dict[str, Any]] = None) -> Path:
    obj = {
        "name": name,
        "tickers": sorted(set([t.strip().upper() for t in tickers if t and str(t).strip()])),
        "meta": meta or {},
        "updated": _now_stamp(),
    }
    return write_json(_portfolio_path(name), obj)


# ----------------------- portfolio models (general models) --------------------

def portfolio_models_dir(portfolio: str) -> Path:
    safe = portfolio.strip().replace("/", "_")
    d = PORTFOLIO_MODELS_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_portfolio_models(portfolio: str) -> List[str]:
    d = portfolio_models_dir(portfolio)
    return sorted([p.stem for p in d.glob("*.json")])


def load_portfolio_model(portfolio: str, model_name: str) -> Dict[str, Any]:
    d = portfolio_models_dir(portfolio)
    return read_json(d / f"{model_name}.json", default={}) or {}


def save_portfolio_model(portfolio: str, model_name: str, payload: Dict[str, Any]) -> Path:
    d = portfolio_models_dir(portfolio)
    return write_json(d / f"{model_name}.json", payload)


# ----------------------- base-model metrics snapshots -------------------------

def base_model_path(portfolio: str) -> Path:
    """Directory where base-model metric snapshots for a portfolio live."""
    safe = portfolio.strip().replace("/", "_")
    d = BASE_METRICS_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_base_metrics_ctx(ctx: Dict[str, Any]) -> Path:
    """
    Save the computed metrics/priors context blob so the Base-Model Lab can reload
    without recomputing. One file per snapshot.
    """
    port = (ctx.get("port") or "unknown").strip()
    d = base_model_path(port)
    snap = {
        "meta": {
            "port": port,
            "created": _now_stamp(),
            "windows": ctx.get("windows", {}),
            "tickers": ctx.get("tickers", []),
            "errors": ctx.get("errors", []),
        },
        # To keep files smaller, store pri_df / sel_df as records (list of dicts)
        "pri_df": (ctx.get("pri_df") or pd.DataFrame()).reset_index().to_dict(orient="records") if "pd" in globals() else [],
        "sel_df": (ctx.get("sel_df") or pd.DataFrame()).reset_index().to_dict(orient="records") if "pd" in globals() else [],
        "priors": ctx.get("priors", {}),
    }
    path = d / f"metrics__{_now_stamp()}.json"
    return write_json(path, snap)


def load_latest_base_metrics(portfolio: str) -> Optional[Dict[str, Any]]:
    d = base_model_path(portfolio)
    files = sorted(d.glob("metrics__*.json"))
    if not files:
        return None
    # Latest by name (timestamp embedded)
    return read_json(files[-1], default=None)


# ----------------------------- simulations -----------------------------------

def simulations_dir() -> Path:
    SIMULATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SIMULATIONS_DIR


def list_simulations() -> List[str]:
    d = simulations_dir()
    return sorted([p.stem for p in d.glob("*.json")])


def load_simulation(name: str) -> Dict[str, Any]:
    return read_json(simulations_dir() / f"{name}.json", default={}) or {}


def save_simulation(name: str, payload: Dict[str, Any]) -> Path:
    return write_json(simulations_dir() / f"{name}.json", payload)


# ------------------------- (optional) datasets/meta ---------------------------

def datasets_dir() -> Path:
    d = ROOT / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Avoid mypy error for pandas in save_base_metrics_ctx when Streamlit runs before imports
try:
    import pandas as pd  # noqa: E402
except Exception:
    pd = None  # type: ignore