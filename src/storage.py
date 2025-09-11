# src/storage.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(".")
STOR = ROOT / "storage"
PORT_DIR = STOR / "portfolios"
BASE_DIR = STOR / "base_models"
REPORT_DIR = STOR / "reports"
CACHE_DIR = STOR / "cache"

for d in (STOR, PORT_DIR, BASE_DIR, REPORT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def read_json(path: Path) -> Any:
    return json.loads(path.read_text())

def portfolio_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
    return PORT_DIR / f"{safe}.json"

def list_portfolios() -> List[str]:
    return sorted(p.stem for p in PORT_DIR.glob("*.json"))

def save_portfolio(name: str, tickers: List[str], filters: Dict[str, Any]) -> Path:
    obj = {"name": name, "tickers": sorted(set([t.upper() for t in tickers])), "filters": filters}
    p = portfolio_path(name)
    write_json(p, obj)
    return p

def load_portfolio(name: str) -> Optional[Dict[str, Any]]:
    p = portfolio_path(name)
    return read_json(p) if p.exists() else None

def base_model_path(archetype: str, portfolio: str, tag: str) -> Path:
    safe_arch = "".join(c for c in archetype if c.isalnum() or c in ("-", "_"))
    safe_port = "".join(c for c in portfolio if c.isalnum() or c in ("-", "_"))
    safe_tag = "".join(c for c in tag if c.isalnum() or c in ("-", "_"))
    return BASE_DIR / f"{safe_arch}__{safe_port}__{safe_tag}.json"