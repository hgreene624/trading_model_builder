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
            json.dump({"portfolios": [], "simulations": [], "strategies": []}, f, indent=2)
    else:
        # Ensure new keys exist in older files
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        changed = False
        if "portfolios" not in data:
            data["portfolios"] = []
            changed = True
        if "simulations" not in data:
            data["simulations"] = []
            changed = True
        if "strategies" not in data:
            data["strategies"] = []
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


# -------------------- Strategies (NEW) --------------------
def save_strategy(symbol: str, model: str, params: Dict[str, Any], name: Optional[str] = None) -> Dict[str, Any]:
    """
    Save a strategy profile for a ticker. Returns the saved record.
    name default: {SYMBOL}-{short_id}
    """
    data = _load()
    sid = str(uuid.uuid4())
    short = sid.split("-")[0].upper()
    rec = {
        "id": sid,
        "name": name.strip() if name else f"{symbol.upper()}-{short}",
        "symbol": symbol.upper(),
        "model": model,
        "params": params,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
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
    arr.sort(key=lambda r: r.get("created_at", ""), reverse=True)
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