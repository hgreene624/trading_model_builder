# src/storage.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# -----------------------------------------------------------------------------
# Core paths & utils
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Portfolio storage (unified, back-compat with legacy "storage/portfolios")
# -----------------------------------------------------------------------------

# New canonical dir
def _portfolios_dir() -> Path:
    return _ensure_dir(get_data_dir() / "portfolios")

# Legacy dir support (read/upgrade on load)
_LEGACY_PORT_ROOT = Path("storage/portfolios")

def _legacy_path_for(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-", ".")).strip()
    return _LEGACY_PORT_ROOT / f"{safe}.json"

def _legacy_txt_path_for(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-", ".")).strip()
    return _LEGACY_PORT_ROOT / f"{safe}.txt"

def _canonical_portfolio_path(name: str) -> Path:
    name = _clean_filename(name)
    return _portfolios_dir() / f"{name}.json"

def list_portfolios() -> list[str]:
    """
    Return portfolio names discovered under storage/portfolios, A→Z.
    Supports legacy .parquet files for discovery only.
    """
    from pathlib import Path
    PORT_DIR = Path("storage") / "portfolios"
    PORT_DIR.mkdir(parents=True, exist_ok=True)

    names = set()
    for p in PORT_DIR.glob("*.json"):
        names.add(p.stem)
    for p in PORT_DIR.glob("*.parquet"):
        names.add(p.stem)  # legacy discovery

    return sorted(names)

def _upgrade_legacy_if_needed(name: str) -> Optional[Dict[str, Any]]:
    """
    If legacy JSON/TXT exists and new JSON does not, read legacy, normalize, and
    write to canonical location. Returns normalized payload or None.
    """
    new_path = _canonical_portfolio_path(name)
    if new_path.exists():
        return _read_json(new_path)

    # Legacy JSON
    jpath = _legacy_path_for(name)
    if jpath.exists():
        try:
            data = _read_json(jpath) or {}
            tickers = [str(t).upper() for t in (data.get("tickers") or [])]
            meta = data.get("meta") or {}
            payload = {"name": name, "tickers": sorted(set(tickers)), "meta": meta}
            _write_json_atomic(new_path, payload)
            return payload
        except Exception:
            pass

    # Legacy TXT (one symbol per line)
    tpath = _legacy_txt_path_for(name)
    if tpath.exists():
        lines = [ln.strip().upper() for ln in tpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        payload = {"name": name, "tickers": sorted(set(lines)), "meta": {"upgraded_from": "txt"}}
        _write_json_atomic(new_path, payload)
        return payload

    return None

def load_portfolio(name: str) -> dict | None:
    """
    Load a portfolio by name from storage/portfolios/<name>.json
    If a legacy .parquet exists, load it read-only (symbols from 'symbol' col or first col).
    Returns:
      {"name": str, "tickers": [str], "meta": dict, "saved_at": iso} or None
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    PORT_DIR = Path("storage") / "portfolios"
    PORT_DIR.mkdir(parents=True, exist_ok=True)

    def _norm_syms(xs):
        seen, out = set(), []
        for x in xs or []:
            s = str(x).strip().upper()
            if s and s not in seen:
                seen.add(s); out.append(s)
        return out

    json_path = PORT_DIR / f"{name}.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "name": data.get("name") or name,
            "tickers": _norm_syms(data.get("tickers", [])),
            "meta": data.get("meta") or {},
            "saved_at": data.get("saved_at") or datetime.now(timezone.utc).isoformat(),
        }

    # Legacy parquet (read-only)
    pq_path = PORT_DIR / f"{name}.parquet"
    if pq_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq_path)
            if "symbol" in df.columns:
                syms = df["symbol"].astype(str).tolist()
            else:
                syms = df.iloc[:, 0].astype(str).tolist()
            return {
                "name": name,
                "tickers": _norm_syms(syms),
                "meta": {"source": "legacy_parquet"},
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            return None

    return None

def save_portfolio(name: str, tickers: list[str], meta: dict | None = None) -> dict:
    """
    Create/overwrite storage/portfolios/<name>.json with {name,tickers,meta,saved_at}.
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    PORT_DIR = Path("storage") / "portfolios"
    PORT_DIR.mkdir(parents=True, exist_ok=True)

    seen, syms = set(), []
    for t in tickers or []:
        s = str(t).strip().upper()
        if s and s not in seen:
            seen.add(s); syms.append(s)

    payload = {
        "name": name,
        "tickers": syms,
        "meta": meta or {},
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = PORT_DIR / f"{name}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return payload

def append_to_portfolio(name: str, tickers: list[str], meta_update: dict | None = None) -> dict:
    """
    Append tickers & merge meta into storage/portfolios/<name>.json (create if missing).
    De-duplicates tickers, uppercases symbols, preserves existing meta keys unless overridden.
    """
    current = load_portfolio(name) or {"name": name, "tickers": [], "meta": {}}
    base_syms = current.get("tickers", [])
    new_syms = tickers or []

    seen, merged = set(), []
    for t in list(base_syms) + list(new_syms):
        s = str(t).strip().upper()
        if s and s not in seen:
            seen.add(s); merged.append(s)

    meta = dict(current.get("meta") or {})
    if meta_update:
        meta.update(meta_update)

    return save_portfolio(name, merged, meta=meta)

# -----------------------------------------------------------------------------
# Portfolio models (per-portfolio saved models)
# -----------------------------------------------------------------------------

def _models_dir(portfolio: str) -> Path:
    portfolio = _clean_filename(portfolio)
    return _ensure_dir(get_data_dir() / "models" / portfolio)

def list_portfolio_models(portfolio: str) -> List[str]:
    d = _models_dir(portfolio)
    items = list(d.glob("*.json"))
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in items]

def save_portfolio_model(portfolio: str, model_name: str, payload: Dict[str, Any]) -> str:
    portfolio = _clean_filename(portfolio)
    model_name = _clean_filename(model_name)
    if not model_name.endswith(".json"):
        model_name += ".json"
    path = _models_dir(portfolio) / model_name

    meta = dict(payload.get("meta", {}))
    meta.setdefault("portfolio", portfolio)
    meta.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    payload["meta"] = meta

    return _write_json_atomic(path, payload)

def load_portfolio_model(portfolio: str, model_filename: str) -> Optional[Dict[str, Any]]:
    portfolio = _clean_filename(portfolio)
    path = _models_dir(portfolio) / model_filename
    return _read_json(path)

# -----------------------------------------------------------------------------
# Training logs (Base Model Lab / General Trainer)
# -----------------------------------------------------------------------------

def _logs_dir(portfolio: str) -> Path:
    portfolio = _clean_filename(portfolio)
    return _ensure_dir(get_data_dir() / "logs" / portfolio)

def save_training_log(portfolio: str, payload: Dict[str, Any], filename: Optional[str] = None) -> str:
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
    d = _logs_dir(portfolio)
    files = list(d.glob("*.json"))
    files.sort(by=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]

def load_training_log(path_or_portfolio: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if filename is None:
        path = Path(path_or_portfolio)
    else:
        path = _logs_dir(path_or_portfolio) / filename
    return _read_json(path)

# -----------------------------------------------------------------------------
# Optional index helpers
# -----------------------------------------------------------------------------

def list_index_cache() -> List[str]:
    d = _ensure_dir(get_data_dir() / "indexes")
    return sorted(p.name for p in d.glob("*.json"))

def save_index_members(index_key: str, payload: Dict[str, Any]) -> str:
    index_key = _clean_filename(index_key)
    d = _ensure_dir(get_data_dir() / "indexes")
    path = d / f"{index_key}.json"
    return _write_json_atomic(path, payload)

def load_index_members(index_key: str) -> Optional[Dict[str, Any]]:
    index_key = _clean_filename(index_key)
    d = _ensure_dir(get_data_dir() / "indexes")
    path = d / f"{index_key}.json"
    return _read_json(path)

# -----------------------------------------------------------------------------
# Legacy simulations (kept as-is)
# -----------------------------------------------------------------------------

def _sims_dir() -> Path:
    return _ensure_dir(get_data_dir() / "simulations")

def list_simulations() -> List[str]:
    d = _sims_dir()
    files = list(d.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]

def save_simulation(name: str, payload: Dict[str, Any]) -> str:
    name = _clean_filename(name)
    if not name.endswith(".json"):
        name += ".json"
    path = _sims_dir() / name
    meta = dict(payload.get("meta", {}))
    meta.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    payload["meta"] = meta
    return _write_json_atomic(path, payload)

def load_simulation(name_or_path: str) -> Optional[Dict[str, Any]]:
    p = Path(name_or_path)
    if p.suffix.lower() == ".json" and p.exists():
        return _read_json(p)
    name = _clean_filename(name_or_path)
    if not name.endswith(".json"):
        name += ".json"
    return _read_json(_sims_dir() / name)