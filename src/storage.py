# src/storage.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


# ────────────────────────────────────────────────────────────────────────────────
# Core paths & utilities
# ────────────────────────────────────────────────────────────────────────────────

def get_data_dir() -> Path:
    """
    Base data directory. Defaults to ./data, override with env DATA_DIR.
    """
    base = os.environ.get("DATA_DIR", "data")
    p = Path(base).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> str:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return str(path)


def _clean_filename(name: str) -> str:
    keep = "-_.() "
    return "".join(ch for ch in name if ch.isalnum() or ch in keep).strip()


# ────────────────────────────────────────────────────────────────────────────────
# Portfolio storage (JSON)
# ────────────────────────────────────────────────────────────────────────────────

def _portfolios_dir() -> Path:
    return _ensure_dir(get_data_dir() / "portfolios")


def list_portfolios() -> List[str]:
    """
    Returns portfolio names (without .json).
    """
    base = _portfolios_dir()
    return sorted(p.stem for p in base.glob("*.json"))


def load_portfolio(name: str) -> Optional[Dict[str, Any]]:
    """
    Loads a portfolio JSON payload or None.
    """
    name = _clean_filename(name)
    path = _portfolios_dir() / f"{name}.json"
    return _read_json(path)


def save_portfolio(name: str, payload: Dict[str, Any]) -> str:
    """
    Saves (creates/updates) a portfolio JSON.
    Ensures 'tickers' are unique while preserving order.
    """
    name = _clean_filename(name)
    path = _portfolios_dir() / f"{name}.json"

    tickers = list(payload.get("tickers", []))
    if tickers:
        # preserve order while deduplicating
        tickers = list(dict.fromkeys(tickers))
        payload["tickers"] = tickers

    # inject metadata
    meta = dict(payload.get("meta", {}))
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["meta"] = meta

    return _write_json_atomic(path, payload)


# Back-compat convenience alias used in older pages
def create_portfolio(name: str, tickers: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> str:
    payload: Dict[str, Any] = {
        "name": name,
        "tickers": tickers or [],
        "meta": meta or {},
    }
    return save_portfolio(name, payload)


# ────────────────────────────────────────────────────────────────────────────────
# Portfolio models (per-portfolio saved models)
# ────────────────────────────────────────────────────────────────────────────────

def _models_dir(portfolio: str) -> Path:
    portfolio = _clean_filename(portfolio)
    return _ensure_dir(get_data_dir() / "models" / portfolio)


def list_portfolio_models(portfolio: str) -> List[str]:
    """
    Returns list of model filenames (without directory), newest first.
    """
    d = _models_dir(portfolio)
    items = list(d.glob("*.json"))
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in items]


def save_portfolio_model(portfolio: str, model_name: str, payload: Dict[str, Any]) -> str:
    """
    Saves a model JSON under data/models/<portfolio>/<model_name>.json
    """
    portfolio = _clean_filename(portfolio)
    model_name = _clean_filename(model_name)
    if not model_name.endswith(".json"):
        model_name += ".json"
    path = _models_dir(portfolio) / model_name

    # add basic metadata if missing
    meta = dict(payload.get("meta", {}))
    meta.setdefault("portfolio", portfolio)
    meta.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    payload["meta"] = meta

    return _write_json_atomic(path, payload)


def load_portfolio_model(portfolio: str, model_filename: str) -> Optional[Dict[str, Any]]:
    """
    Loads a saved portfolio model by filename (e.g. 'MyModel.json').
    """
    portfolio = _clean_filename(portfolio)
    path = _models_dir(portfolio) / model_filename
    return _read_json(path)


# ────────────────────────────────────────────────────────────────────────────────
# Training logs (Base Model Lab / General Trainer)
# ────────────────────────────────────────────────────────────────────────────────

def _logs_dir(portfolio: str) -> Path:
    portfolio = _clean_filename(portfolio)
    return _ensure_dir(get_data_dir() / "logs" / portfolio)


def save_training_log(portfolio: str, payload: Dict[str, Any], filename: Optional[str] = None) -> str:
    """
    Persists a training log JSON for a given portfolio.
    Default filename is 'base_model_training_log_YYYYMMDD-HHMMSS.json'.
    Returns the path string.
    """
    d = _logs_dir(portfolio)
    if filename:
        filename = _clean_filename(filename)
        if not filename.endswith(".json"):
            filename += ".json"
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"base_model_training_log_{ts}.json"
    path = d / filename
    return _write_json_atomic(path, payload)


def list_training_logs(portfolio: str) -> List[str]:
    """
    Returns list of log file paths (absolute), newest first.
    """
    d = _logs_dir(portfolio)
    files = list(d.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def load_training_log(path_or_portfolio: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    If called with a full path, loads that file.
    If called with (portfolio, filename), loads from data/logs/<portfolio>/<filename>.
    """
    if filename is None:
        # assume full path
        path = Path(path_or_portfolio)
    else:
        path = _logs_dir(path_or_portfolio) / filename
    return _read_json(path)


# ────────────────────────────────────────────────────────────────────────────────
# Optional helpers some pages expect (safe no-ops)
# ────────────────────────────────────────────────────────────────────────────────

def list_index_cache() -> List[str]:
    """
    Optional: return cached index metadata files if your Portfolio page uses them.
    Safe to leave minimal.
    """
    d = _ensure_dir(get_data_dir() / "indexes")
    return sorted(p.name for p in d.glob("*.json"))


def save_index_members(index_key: str, payload: Dict[str, Any]) -> str:
    """
    Optional: persist index constituents metadata used by Portfolio builder.
    """
    index_key = _clean_filename(index_key)
    d = _ensure_dir(get_data_dir() / "indexes")
    path = d / f"{index_key}.json"
    return _write_json_atomic(path, payload)


def load_index_members(index_key: str) -> Optional[Dict[str, Any]]:
    index_key = _clean_filename(index_key)
    d = _ensure_dir(get_data_dir() / "indexes")
    path = d / f"{index_key}.json"
    return _read_json(path)

# ────────────────────────────────────────────────────────────────────────────────
# Legacy simulation helpers (back-compat shims for older pages)
# ────────────────────────────────────────────────────────────────────────────────

def _sims_dir() -> Path:
    return _ensure_dir(get_data_dir() / "simulations")

def list_simulations() -> List[str]:
    """
    Returns list of saved simulation JSON filenames (newest first),
    stored under data/simulations/.
    """
    d = _sims_dir()
    files = list(d.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]

def save_simulation(name: str, payload: Dict[str, Any]) -> str:
    """
    Saves a simulation run to data/simulations/<name>.json.
    """
    name = _clean_filename(name)
    if not name.endswith(".json"):
        name += ".json"
    path = _sims_dir() / name

    meta = dict(payload.get("meta", {}))
    meta.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    payload["meta"] = meta

    return _write_json_atomic(path, payload)

def load_simulation(name_or_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads a simulation JSON by filename (from data/simulations)
    or by absolute/relative path if a path is provided.
    """
    p = Path(name_or_path)
    if p.suffix.lower() == ".json" and p.exists():
        return _read_json(p)

    # treat as a filename under simulations/
    name = _clean_filename(name_or_path)
    if not name.endswith(".json"):
        name += ".json"
    return _read_json(_sims_dir() / name)