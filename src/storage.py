# src/storage.py
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── folders ────────────────────────────────────────────────────────────────────
ROOT = Path(os.getenv("DATA_ROOT", ".")).resolve()
DATA_DIR = ROOT / "data"
PORTFOLIOS_DIR = DATA_DIR / "portfolios"
MODELS_DIR = DATA_DIR / "portfolio_models"
LOGS_DIR = DATA_DIR / "logs"

for d in (PORTFOLIOS_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────
def _norm_symbols(symbols: List[str]) -> List[str]:
    return sorted({str(s).strip().upper() for s in symbols or [] if str(s).strip()})

def _atomic_write(path: Path, payload: Dict[str, Any]) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)
    return str(path)

# ── portfolios ────────────────────────────────────────────────────────────────
def list_portfolios() -> List[str]:
    return sorted(p.stem for p in PORTFOLIOS_DIR.glob("*.json"))

def load_portfolio(name: str) -> Optional[Dict[str, Any]]:
    p = PORTFOLIOS_DIR / f"{name}.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_portfolio(name: str, tickers: List[str], meta: Optional[Dict[str, Any]] = None) -> str:
    payload = {
        "name": name,
        "tickers": _norm_symbols(tickers),
        "meta": meta or {},
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "version": 1,
    }
    return _atomic_write(PORTFOLIOS_DIR / f"{name}.json", payload)

# Backwards-compat: some pages may still call create_portfolio / append_to_portfolio
def create_portfolio(name: str, tickers: List[str], meta: Optional[Dict[str, Any]] = None) -> str:
    return save_portfolio(name, tickers, meta)

def append_to_portfolio(name: str, more_tickers: List[str], meta_update: Optional[Dict[str, Any]] = None) -> str:
    existing = load_portfolio(name) or {"tickers": [], "meta": {}}
    merged = _norm_symbols((existing.get("tickers") or []) + (more_tickers or []))
    meta = existing.get("meta", {}) or {}
    if meta_update:
        meta.update(meta_update)
    return save_portfolio(name, merged, meta)

# ── portfolio models ──────────────────────────────────────────────────────────
def list_portfolio_models(portfolio: str) -> List[str]:
    d = MODELS_DIR / portfolio
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))

def save_portfolio_model(portfolio: str, model_name: str, payload: Dict[str, Any]) -> str:
    d = MODELS_DIR / portfolio
    d.mkdir(parents=True, exist_ok=True)
    return _atomic_write(d / f"{model_name}.json", payload)

# ── training logs ─────────────────────────────────────────────────────────────
def save_training_log(portfolio: str, log_payload: Dict[str, Any]) -> str:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOGS_DIR / f"base_model_log_{portfolio}_{ts}.json"
    return _atomic_write(path, log_payload)