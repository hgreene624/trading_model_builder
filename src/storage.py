# src/storage.py
from __future__ import annotations
import os, json, uuid, datetime
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORE_DIR = os.path.join(BASE_DIR, "storage")
STORE_PATH = os.path.join(STORE_DIR, "portfolios.json")


def _ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)
    if not os.path.exists(STORE_PATH):
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump({"portfolios": [], "simulations": [], "strategies": [], "param_bounds": []}, f, indent=2)
    else:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        changed = False
        for key in ["portfolios", "simulations", "strategies", "param_bounds"]:
            if key not in data:
                data[key] = []
                changed = True
        # Backfill is_default flag if missing on strategies/param_bounds
        for coll in ["strategies", "param_bounds"]:
            for rec in data.get(coll, []):
                if "is_default" not in rec:
                    rec["is_default"] = False
                    changed = True
        if changed:
            with open(STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)


def _load() -> Dict[str, Any]:
    _ensure_store()
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: Dict[str, Any]):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -------------------- Portfolios --------------------
def list_portfolios() -> List[Dict[str, Any]]:
    return _load().get("portfolios", [])


def get_portfolio(portfolio_id: str) -> Optional[Dict[str, Any]]:
    for p in list_portfolios():
        if p["id"] == portfolio_id:
            return p
    return None


def create_portfolio(name: str) -> Dict[str, Any]:
    data = _load()
    p = {"id": str(uuid.uuid4()), "name": name, "created_at": datetime.datetime.utcnow().isoformat(), "items": []}
    data["portfolios"].append(p)
    _save(data)
    return p


def delete_portfolio(portfolio_id: str) -> bool:
    data = _load()
    before = len(data["portfolios"])
    data["portfolios"] = [p for p in data["portfolios"] if p["id"] != portfolio_id]
    _save(data)
    return len(data["portfolios"]) < before


def add_item(portfolio_id: str, symbol: str, model: str, params: Dict[str, Any]) -> bool:
    data = _load()
    for p in data["portfolios"]:
        if p["id"] == portfolio_id:
            existing = next((it for it in p["items"] if it["symbol"].upper() == symbol.upper() and it["model"] == model), None)
            if existing:
                existing["params"] = params
                existing["updated_at"] = datetime.datetime.utcnow().isoformat()
            else:
                p["items"].append({"symbol": symbol.upper(), "model": model, "params": params, "added_at": datetime.datetime.utcnow().isoformat()})
            _save(data)
            return True
    return False


def remove_item(portfolio_id: str, symbol: str, model: str) -> bool:
    data = _load()
    for p in data["portfolios"]:
        if p["id"] == portfolio_id:
            before = len(p["items"])
            p["items"] = [it for it in p["items"] if not (it["symbol"].upper() == symbol.upper() and it["model"] == model)]
            _save(data)
            return len(p["items"]) < before
    return False


# -------------------- Simulations --------------------
def list_simulations(limit: int = 20) -> List[Dict[str, Any]]:
    sims = _load().get("simulations", [])
    sims.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return sims[:limit]


def add_simulation(record: Dict[str, Any]) -> Dict[str, Any]:
    data = _load()
    record = dict(record)
    record["id"] = record.get("id") or str(uuid.uuid4())
    record["created_at"] = record.get("created_at") or datetime.datetime.utcnow().isoformat()
    data.setdefault("simulations", []).append(record)
    _save(data)
    return record


# -------------------- Strategies (model params) --------------------
def save_strategy(symbol: str, model: str, params: Dict[str, Any], name: Optional[str] = None, is_default: bool = False) -> Dict[str, Any]:
    data = _load()
    sid = str(uuid.uuid4())
    short = sid.split("-")[0].upper()
    rec = {
        "id": sid,
        "name": (name.strip() if name else f"{symbol.upper()}-{short}"),
        "symbol": symbol.upper(),
        "model": model,
        "params": params,
        "is_default": bool(is_default),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    if is_default:
        # unset previous defaults for this symbol+model
        for s in data.setdefault("strategies", []):
            if s.get("symbol") == rec["symbol"] and s.get("model") == rec["model"]:
                s["is_default"] = False
    data.setdefault("strategies", []).append(rec)
    _save(data)
    return rec


def list_strategies(symbol: Optional[str] = None, model: Optional[str] = None) -> List[Dict[str, Any]]:
    data = _load()
    arr = data.get("strategies", [])
    if symbol:
        arr = [s for s in arr if s.get("symbol", "").upper() == symbol.upper()]
    if model:
        arr = [s for s in arr if s.get("model") == model]
    # sort default first, then by created_at desc
    arr.sort(key=lambda r: (not r.get("is_default", False), r.get("created_at", "")), reverse=False)
    return arr


def get_strategy(strategy_id: str) -> Optional[Dict[str, Any]]:
    data = _load()
    for s in data.get("strategies", []):
        if s.get("id") == strategy_id:
            return s
    return None


def delete_strategy(strategy_id: str) -> bool:
    data = _load()
    before = len(data.get("strategies", []))
    data["strategies"] = [s for s in data.get("strategies", []) if s.get("id") != strategy_id]
    _save(data)
    return len(data["strategies"]) < before


def set_default_strategy(symbol: str, model: str, strategy_id: str) -> bool:
    data = _load()
    changed = False
    for s in data.get("strategies", []):
        if s.get("symbol") == symbol.upper() and s.get("model") == model:
            is_def = (s.get("id") == strategy_id)
            if s.get("is_default", False) != is_def:
                s["is_default"] = is_def
                changed = True
    if changed:
        _save(data)
    return changed


def get_default_strategy(symbol: str, model: str) -> Optional[Dict[str, Any]]:
    arr = list_strategies(symbol, model)
    for s in arr:
        if s.get("is_default"):
            return s
    return None


# -------------------- Parameter Bounds Profiles (for tuner) --------------------
def save_param_bounds(symbol: str, model: str, profile: Dict[str, Any], name: Optional[str] = None, is_default: bool = False) -> Dict[str, Any]:
    """
    profile includes:
      start, end, starting_equity, pop_size, generations, crossover_rate, mutation_rate,
      bounds: {breakout_min, breakout_max, exit_min, exit_max, atr_min, atr_max,
               atr_multiple_min, atr_multiple_max, risk_per_trade_min, risk_per_trade_max}
    """
    data = _load()
    bid = str(uuid.uuid4())
    short = bid.split("-")[0].upper()
    rec = {
        "id": bid,
        "name": (name.strip() if name else f"{symbol.upper()}-BND-{short}"),
        "symbol": symbol.upper(),
        "model": model,
        "profile": profile,
        "is_default": bool(is_default),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    if is_default:
        for b in data.setdefault("param_bounds", []):
            if b.get("symbol") == rec["symbol"] and b.get("model") == rec["model"]:
                b["is_default"] = False
    data.setdefault("param_bounds", []).append(rec)
    _save(data)
    return rec


def list_param_bounds(symbol: Optional[str] = None, model: Optional[str] = None) -> List[Dict[str, Any]]:
    data = _load()
    arr = data.get("param_bounds", [])
    if symbol:
        arr = [b for b in arr if b.get("symbol", "").upper() == symbol.upper()]
    if model:
        arr = [b for b in arr if b.get("model") == model]
    # default first, then created_at desc
    arr.sort(key=lambda r: (not r.get("is_default", False), r.get("created_at", "")), reverse=False)
    return arr


def get_param_bounds(bounds_id: str) -> Optional[Dict[str, Any]]:
    data = _load()
    for b in data.get("param_bounds", []):
        if b.get("id") == bounds_id:
            return b
    return None


def delete_param_bounds(bounds_id: str) -> bool:
    data = _load()
    before = len(data.get("param_bounds", []))
    data["param_bounds"] = [b for b in data.get("param_bounds", []) if b.get("id") != bounds_id]
    _save(data)
    return len(data["param_bounds"]) < before


def set_default_param_bounds(symbol: str, model: str, bounds_id: str) -> bool:
    data = _load()
    changed = False
    for b in data.get("param_bounds", []):
        if b.get("symbol") == symbol.upper() and b.get("model") == model:
            is_def = (b.get("id") == bounds_id)
            if b.get("is_default", False) != is_def:
                b["is_default"] = is_def
                changed = True
    if changed:
        _save(data)
    return changed


def get_default_param_bounds(symbol: str, model: str) -> Optional[Dict[str, Any]]:
    arr = list_param_bounds(symbol, model)
    for b in arr:
        if b.get("is_default"):
            return b
    return None