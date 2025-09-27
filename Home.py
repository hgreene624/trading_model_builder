from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# ---- Page config
def render_home():
    st.set_page_config(page_title="Trading Research Dashboard", layout="wide")

    # ---- Env
    load_dotenv()

    # ---- Alpaca connection status -------------------------------------------------

    def _alpaca_connection_status():
        """Return (title, message, level) for Alpaca Trading API status."""
        # Prefer Streamlit secrets, fall back to environment
        api_key = st.secrets.get("ALPACA_API_KEY", None) if hasattr(st, "secrets") else None
        api_secret = st.secrets.get("ALPACA_SECRET_KEY", None) if hasattr(st, "secrets") else None
        base_url = st.secrets.get("ALPACA_BASE_URL", None) if hasattr(st, "secrets") else None

        import os as _os
        api_key = api_key or _os.getenv("ALPACA_API_KEY")
        api_secret = api_secret or _os.getenv("ALPACA_SECRET_KEY")
        base_url = base_url or _os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        paper = True if (base_url and "paper" in base_url) else True  # default to paper

        if not api_key or not api_secret:
            return (
                "⚠️ Alpaca: Missing credentials",
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env or Streamlit secrets.",
                "warning",
            )

        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(api_key, api_secret, paper=paper)
            acct = client.get_account()
            msg = f"Status: {acct.status} | Equity: {acct.equity} | Buying Power: {acct.buying_power}"
            return ("✅ Alpaca: Connected", msg, "success")
        except Exception as e:
            return ("❌ Alpaca: Error", f"{type(e).__name__}: {e}", "error")

    # ---- Storage imports (with fallback shims)
    try:
        from src.storage import list_portfolios, list_simulations  # use storage if available
    except Exception:
        # silent fallback – avoid noisy toasts on Home
        def list_portfolios(root: str | Path = "storage/portfolios"):
            p = Path(root)
            if not p.exists():
                return []
            return [f.stem for f in p.glob("*.json")]

        def list_simulations(limit: int = 10,
                             roots: tuple[str, ...] = ("storage/simulations", "storage/reports"),
                             exts: tuple[str, ...] = (".json",)):
            items = []
            for root in roots:
                p = Path(root)
                if not p.exists():
                    continue
                for f in p.rglob("*.json"):
                    try:
                        import json
                        with f.open("r", encoding="utf-8") as fh:
                            data = json.load(fh)
                    except Exception:
                        data = {}
                    items.append({
                        "name": f.name,
                        "portfolio_name": data.get("portfolio_name", data.get("portfolio", "")),
                        "start": data.get("start", ""),
                        "end": data.get("end", ""),
                        "starting_equity": data.get("starting_equity", 0),
                        "final_equity": data.get("final_equity", 0),
                        "created_at": data.get("created_at", ""),
                        "path": str(f),
                        "_mtime": f.stat().st_mtime,
                    })
            items.sort(key=lambda r: r.get("_mtime", 0), reverse=True)
            return items[:limit] if limit else items

    # ---- Title
    st.title("📊 Trading Research Dashboard")

    # Connection status banner
    _title, _msg, _level = _alpaca_connection_status()
    if _level == "success":
        st.success(f"**{_title}** — {_msg}")
    elif _level == "warning":
        st.warning(f"**{_title}** — {_msg}")
    else:
        st.error(f"**{_title}** — {_msg}")

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
            st.dataframe({"portfolio": portfolios},  width="stretch", height=300)
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
            st.dataframe(df,  width="stretch", height=300)
        else:
            st.info("No simulation artifacts found under storage/simulations or storage/reports.")

    st.divider()

    st.caption("Shortcuts")
    st.write("- Open **Portfolios** from the left sidebar.")
    st.write("- Use **Base Model Lab** to train priors, then **Tuning** to optimize, then **Simulate Portfolio**.")


import multiprocessing as _mp
from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx

def _has_ctx():
    try:
        return _get_ctx() is not None
    except Exception:
        return False

# Only render UI in the Streamlit main process with an active ScriptRunContext
if _mp.current_process().name == "MainProcess" and _has_ctx():
    render_home()