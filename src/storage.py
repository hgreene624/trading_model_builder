# src/storage.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import pandas as pd
# ---- Root & directories -------------------------------------------------------

ROOT = Path(".")
STOR = ROOT / "storage"

PORT_DIR = STOR / "portfolios"
BASE_DIR = STOR / "base_models"
REPORT_DIR = STOR / "reports"
SIM_DIR = STOR / "simulations"
CACHE_DIR = STOR / "cache"
UNIVERSE_DIR = STOR / "universe"

for d in (STOR, PORT_DIR, BASE_DIR, REPORT_DIR, SIM_DIR, CACHE_DIR, UNIVERSE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---- JSON IO -----------------------------------------------------------------

def write_json(path: Path, obj: Any) -> None:
    """Pretty-write JSON with UTF-8, ensuring parent exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---- Portfolio helpers --------------------------------------------------------

# --- Base-model metrics persistence ------------------------------------------
from pathlib import Path
import json
from datetime import datetime

def _bm_metrics_dir() -> Path:
    p = Path("storage/base_models/metrics_cache")
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)

def save_base_metrics_ctx(ctx: dict) -> str:
    """
    Persist the computed metrics/prior context from Base-Model Lab.
    """
    win = ctx.get("windows", {})
    port = ctx.get("port", "unknown")
    key = f"{port}__{win.get('priors_start')}__{win.get('priors_end')}__{win.get('select_start')}__{win.get('select_end')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "meta": {
            "port": port,
            "created": ts,
            "windows": win,
            "tickers": ctx.get("tickers", []),
            "errors": ctx.get("errors", []),
        },
        "priors": ctx.get("priors", {}),
        "pri_df": ctx.get("pri_df", pd.DataFrame()).reset_index().to_dict(orient="records") if "pri_df" in ctx else [],
        "sel_df": ctx.get("sel_df", pd.DataFrame()).reset_index().to_dict(orient="records") if "sel_df" in ctx else [],
    }
    out_path = _bm_metrics_dir() / f"{key}__{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out_path.as_posix()

def list_base_metrics(port: str | None = None) -> list[str]:
    d = _bm_metrics_dir()
    allf = sorted([p.name for p in d.glob("*.json")])
    if port:
        allf = [f for f in allf if f.startswith(f"{port}__")]
    return allf

def load_base_metrics(file_name: str) -> dict | None:
    p = _bm_metrics_dir() / file_name
    if not p.exists():
        return None
    return read_json(p)

def load_latest_base_metrics(port: str) -> dict | None:
    files = list_base_metrics(port)
    if not files:
        return None
    latest = sorted(files)[-1]
    return load_base_metrics(latest)

def _safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip("_")


def portfolio_path(name: str) -> Path:
    return PORT_DIR / f"{_safe_name(name)}.json"


def list_portfolios() -> List[str]:
    """Return portfolio names (without extension)."""
    return sorted(p.stem for p in PORT_DIR.glob("*.json"))


def load_portfolio(name: str) -> Optional[Dict[str, Any]]:
    p = portfolio_path(name)
    return read_json(p) if p.exists() else None


def save_portfolio(name: str,
                   obj_or_tickers: Dict[str, Any] | Iterable[str] | None = None,
                   filters: Optional[Dict[str, Any]] = None) -> Path:
    """
    Flexible saver (backward compatible):

    - save_portfolio("core", {"name":"core","tickers":[...], ...})
    - save_portfolio("core", ["AAPL","MSFT"], filters={"min_price":5})

    Returns the path written.
    """
    p = portfolio_path(name)

    if isinstance(obj_or_tickers, dict):
        obj = dict(obj_or_tickers)  # shallow copy
        if "name" not in obj:
            obj["name"] = name
        if "tickers" in obj:
            obj["tickers"] = sorted({str(t).upper() for t in obj["tickers"]})
        obj.setdefault("updated", time.strftime("%Y-%m-%d"))
        write_json(p, obj)
        return p

    # Build from tickers + filters
    tickers: List[str] = []
    if obj_or_tickers is not None:
        tickers = sorted({str(t).upper() for t in obj_or_tickers})
    obj = {
        "name": name,
        "tickers": tickers,
        "filters": filters or {},
        "updated": time.strftime("%Y-%m-%d"),
    }
    write_json(p, obj)
    return p


# ---- Base model specs ---------------------------------------------------------

def base_model_path(archetype: str, portfolio: str, tag: str) -> Path:
    safe_arch = _safe_name(archetype)
    safe_port = _safe_name(portfolio)
    safe_tag = _safe_name(tag)
    return BASE_DIR / f"{safe_arch}__{safe_port}__{safe_tag}.json"


def save_base_spec(archetype: str, portfolio: str, tag: str, spec: Dict[str, Any]) -> Path:
    """Persist a base model spec JSON under storage/base_models."""
    path = base_model_path(archetype, portfolio, tag)
    payload = dict(spec)
    payload.setdefault("archetype", archetype)
    payload.setdefault("portfolio", portfolio)
    payload.setdefault("tag", tag)
    payload.setdefault("saved_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(path, payload)
    return path


def list_base_specs(filter_portfolio: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all saved base model specs (parsed JSON)."""
    out: List[Dict[str, Any]] = []
    for f in BASE_DIR.glob("*.json"):
        try:
            obj = read_json(f)
            if filter_portfolio and obj.get("portfolio") != filter_portfolio:
                continue
            obj["_path"] = str(f)
            out.append(obj)
        except Exception:
            continue
    # newest first (by file mtime)
    out.sort(key=lambda o: Path(o["_path"]).stat().st_mtime, reverse=True)
    return out


# ---- Simulations / reports listing (for Home.py) ------------------------------

def list_simulations(limit: int = 50,
                     roots: Tuple[str, ...] = ("storage/simulations", "storage/reports"),
                     extensions: Tuple[str, ...] = (".json",)) -> List[Dict[str, Any]]:
    """
    Scan simulation/report artifact folders for JSON summaries and normalize keys.

    Returns a list of dicts with:
      path, name, created_at, portfolio_name, start, end, starting_equity, final_equity
    Sorted by modified time (newest first).
    """
    items: List[Dict[str, Any]] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in extensions:
                continue
            try:
                try:
                    data = read_json(f)
                except Exception:
                    data = {}

                mtime = f.stat().st_mtime
                created = data.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))
                portfolio = data.get("portfolio_name") or data.get("portfolio") or data.get("meta", {}).get("portfolio", "")

                items.append({
                    "path": str(f),
                    "name": f.name,
                    "created_at": created,
                    "portfolio_name": portfolio,
                    "start": data.get("start") or data.get("date_start") or data.get("start_date") or data.get("meta", {}).get("start", ""),
                    "end": data.get("end") or data.get("date_end") or data.get("end_date") or data.get("meta", {}).get("end", ""),
                    "starting_equity": data.get("starting_equity", data.get("start_equity", None)),
                    "final_equity": data.get("final_equity", data.get("equity_final", None)),
                    "modified_ts": mtime,
                    "size": f.stat().st_size,
                })
            except Exception:
                continue

    items.sort(key=lambda r: r.get("modified_ts", 0), reverse=True)
    if limit and limit > 0:
        items = items[:limit]
    return items


# ---- Simple cache helpers (optional) -----------------------------------------

def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{_safe_name(name)}.json"


def cache_write(name: str, obj: Any) -> Path:
    p = cache_path(name)
    write_json(p, obj)
    return p


def cache_read(name: str) -> Optional[Any]:
    p = cache_path(name)
    return read_json(p) if p.exists() else None


__all__ = [
    # dirs
    "STOR", "PORT_DIR", "BASE_DIR", "REPORT_DIR", "SIM_DIR", "CACHE_DIR", "UNIVERSE_DIR",
    # json
    "write_json", "read_json",
    # portfolios
    "portfolio_path", "list_portfolios", "save_portfolio", "load_portfolio",
    # base models
    "base_model_path", "save_base_spec", "list_base_specs",
    # sims
    "list_simulations",
    # cache
    "cache_path", "cache_write", "cache_read",
]