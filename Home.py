# ---- Simulation listing helper (for Home.py) ---------------------------------
from pathlib import Path
from typing import List, Dict, Tuple
import json
import time

def list_simulations(limit: int = 50,
                     roots: Tuple[str, ...] = ("storage/simulations", "storage/reports"),
                     exts: Tuple[str, ...] = (".json",)) -> List[Dict]:
    """
    Return recent simulation artifacts so Home.py can render a recents table.

    We look for JSON artifacts under storage/simulations and storage/reports.
    We then normalize keys to what Home.py expects:
      created_at, portfolio_name, start, end, starting_equity, final_equity

    If a JSON file is missing some fields, we fill sensible defaults and use
    the file's mtime as created_at.

    NOTE: We intentionally avoid reading Parquet here to keep it lightweight.
    """
    items: List[Dict] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            try:
                data = {}
                try:
                    with f.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    # Not a valid JSON we understand; still show a shell row
                    data = {}

                st_mtime = f.stat().st_mtime
                created = data.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st_mtime))

                # normalize fields with several common aliases
                portfolio = (
                    data.get("portfolio_name")
                    or data.get("portfolio")
                    or data.get("meta", {}).get("portfolio", "")
                )
                start = (
                    data.get("start")
                    or data.get("date_start")
                    or data.get("start_date")
                    or data.get("meta", {}).get("start", "")
                )
                end = (
                    data.get("end")
                    or data.get("date_end")
                    or data.get("end_date")
                    or data.get("meta", {}).get("end", "")
                )
                start_eq = data.get("starting_equity", data.get("start_equity", 0))
                final_eq = data.get("final_equity", data.get("equity_final", 0))

                items.append({
                    "path": str(f),
                    "name": f.name,
                    "created_at": created,
                    "portfolio_name": portfolio,
                    "start": start,
                    "end": end,
                    "starting_equity": start_eq,
                    "final_equity": final_eq,
                    "modified_ts": st_mtime,
                })
            except Exception:
                # Skip unreadable files
                continue

    # newest first
    items.sort(key=lambda r: r.get("modified_ts", 0), reverse=True)
    if limit and limit > 0:
        items = items[:limit]
    return items

# Ensure the symbol is exported even if __all__ exists.
try:
    _all = globals().get("__all__", None)
    if isinstance(_all, list):
        if "list_simulations" not in _all:
            _all.append("list_simulations")
    elif isinstance(_all, tuple):
        if "list_simulations" not in _all:
            __all__ = list(_all) + ["list_simulations"]
    else:
        __all__ = ["list_simulations"]
except Exception:
    # Fallback if __all__ is not defined or is immutable
    try:
        __all__ = list(__all__) + ["list_simulations"]  # type: ignore[name-defined]
    except Exception:
        __all__ = ["list_simulations"]

import os
import sys
import json
import time
import inspect
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ---- Page config
st.set_page_config(page_title="Trading Research Dashboard", layout="wide")

# ---- Env
load_dotenv()

# ---- Sidebar debug + cache clear
st.sidebar.caption("**Debug**")
st.sidebar.write("Python:", sys.executable)
st.sidebar.write("CWD:", os.getcwd())

try:
    import src
    st.sidebar.write("src path:", inspect.getfile(src))
except Exception as _e:
    st.sidebar.write("src import issue:", str(_e))

if st.sidebar.button("Clear Streamlit caches"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Caches cleared")

# ---- Storage imports (with fallback shims)
try:
    from src.storage import list_portfolios, list_simulations
except Exception as e:
    st.warning(f"storage import issue: {e}")

    def list_portfolios(root: str | Path = "storage/portfolios"):
        p = Path(root)
        if not p.exists():
            return []
        return [f.stem for f in p.glob("*.json")]

    def list_simulations(limit: int = 50,
                         roots=("storage/simulations", "storage/reports")):
        items = []
        for root in roots:
            p = Path(root)
            if not p.exists():
                continue
            for f in p.rglob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                ts = f.stat().st_mtime
                created = data.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
                items.append({
                    "name": f.name,
                    "path": str(f),
                    "created_at": created,
                    "portfolio_name": data.get("portfolio_name") or data.get("portfolio") or data.get("meta", {}).get("portfolio", ""),
                    "start": data.get("start") or data.get("start_date") or data.get("meta", {}).get("start", ""),
                    "end": data.get("end") or data.get("end_date") or data.get("meta", {}).get("end", ""),
                    "starting_equity": data.get("starting_equity", data.get("start_equity", None)),
                    "final_equity": data.get("final_equity", data.get("equity_final", None)),
                })
        items.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        if limit and limit > 0:
            items = items[:limit]
        return items

# ---- Title
st.title("📊 Trading Research Dashboard")

# ---- High-level metrics
try:
    portfolios = list_portfolios() if callable(list_portfolios) else []
except Exception:
    portfolios = []

try:
    recent_sims = list_simulations(limit=10)
except Exception:
    recent_sims = []

base_model_dir = Path("storage/base_models")
base_specs = list(base_model_dir.glob("*.json")) if base_model_dir.exists() else []

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Portfolios", len(portfolios))
with m2:
    st.metric("Recent Sims", len(recent_sims))
with m3:
    st.metric("Base Model Specs", len(base_specs))

st.divider()

# ---- Lists
c1, c2 = st.columns(2)

with c1:
    st.subheader("🗂️ Portfolios")
    if portfolios:
        st.dataframe({"portfolio": portfolios}, use_container_width=True, height=300)
    else:
        st.info("No portfolios saved yet. Create one in the **Portfolios** page.")

with c2:
    st.subheader("🧪 Recent simulations")
    if recent_sims:
        import pandas as pd
        cols = ["name", "portfolio_name", "start", "end", "starting_equity", "final_equity", "created_at", "path"]
        try:
            df = pd.DataFrame(recent_sims)[cols]
        except Exception:
            df = pd.DataFrame(recent_sims)
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.info("No simulation artifacts found under storage/simulations or storage/reports.")

st.divider()

st.caption("Shortcuts")
st.write("- Open **Portfolios** from the left sidebar.")
st.write("- Use **Base Model Lab** to train priors, then **Tuning** to optimize, then **Simulate Portfolio**.")