# tests/test_data_pipeline.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ))  # ensure 'src' is importable when run as module

SECRETS_FILE = PROJ / ".streamlit" / "secrets.toml"


def _load_secrets_to_env():
    # Try streamlit first (if available), else parse toml manually.
    loaded = False
    keys = {}
    try:
        import streamlit as st  # noqa
        try:
            from streamlit.web.cli import _main_run_clExplicit
        except Exception:
            pass
        try:
            from streamlit.runtime.secrets import _file as st_secrets_file
            # st.secrets only works inside a Streamlit runtime; fallback:
        except Exception:
            pass
    except Exception:
        pass

    # Manual TOML load
    try:
        import tomllib  # py3.11+
    except Exception:
        tomllib = None

    if SECRETS_FILE.exists() and tomllib is not None:
        try:
            with open(SECRETS_FILE, "rb") as f:
                data = tomllib.load(f)
            # Accept either ALPACA_* or APCA_* naming
            def pick(*names):
                for n in names:
                    if n in data:
                        return data[n]
                return None

            k = pick("APCA_API_KEY_ID", "ALPACA_API_KEY")
            s = pick("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY")
            b = pick("APCA_API_BASE_URL", "ALPACA_BASE_URL")
            d = pick("ALPACA_DATA_URL")  # optional

            if k: os.environ["APCA_API_KEY_ID"] = k
            if s: os.environ["APCA_API_SECRET_KEY"] = s
            if b: os.environ["APCA_API_BASE_URL"] = b
            if d: os.environ["ALPACA_DATA_URL"] = d

            keys = list(data.keys())
            loaded = True
        except Exception as e:
            print("WARN: Failed to parse secrets.toml:", e)
    else:
        keys = []

    print("\nsecrets.toml present:", SECRETS_FILE.exists(), "keys:", keys)
    return loaded


def _print_env():
    print("\n========================================================================")
    print("Environment")
    print("========================================================================")
    print("Python: ", sys.version.split()[0], " (", sys.executable, ")")
    print("PWD:    ", os.getcwd())
    print("APCA_API_KEY_ID     :", os.environ.get("APCA_API_KEY_ID", "<EMPTY>" if "APCA_API_KEY_ID" not in os.environ else "************" + os.environ["APCA_API_KEY_ID"][-4:]))
    print("APCA_API_SECRET_KEY :", "<EMPTY>" if "APCA_API_SECRET_KEY" not in os.environ else "*"*36 + os.environ["APCA_API_SECRET_KEY"][-4:])
    print("APCA_API_BASE_URL   :", os.environ.get("APCA_API_BASE_URL", "<EMPTY>"))
    print("ALPACA_DATA_URL     :", os.environ.get("ALPACA_DATA_URL", "<EMPTY>"))
    print("ALPACA_FEED         :", os.environ.get("ALPACA_FEED", "<EMPTY>"), "(set to 'iex' for paper/basic)")  # important


def _show_df(df: pd.DataFrame, label: str):
    if df is None or df.empty:
        print(f"{label}: EMPTY")
        return
    cols = list(df.columns)
    idx = type(df.index).__name__
    print(f"{label}: rows={len(df)} cols={cols} index={idx}")
    try:
        print("  head:", df.index[0], "tail:", df.index[-1])
    except Exception:
        pass


def _leg(title, fn, *args, **kwargs):
    print("\n------------------------------------------------------------------------")
    print(title)
    print("------------------------------------------------------------------------")
    try:
        df = fn(*args, **kwargs)
        _show_df(df, "OK")
        print("RESULT: PASS")
    except Exception as e:
        print("RESULT: FAIL")
        print("ERROR :", repr(e))


def main():
    _load_secrets_to_env()
    _print_env()

    # Date range: 1y back to avoid recent-SIP windows; adjust as needed.
    end = datetime.utcnow()
    start = end - timedelta(days=365)

    symbol = os.environ.get("TEST_SYMBOL", "AAPL")

    print("\n========================================================================")
    print("Module paths")
    print("========================================================================")
    import importlib
    L = importlib.import_module("src.data.loader")
    A = importlib.import_module("src.data.alpaca_data")
    Y = importlib.import_module("src.data.yf")
    print("loader file :", getattr(L, "__file__", "<??>"))
    print("alpaca file :", getattr(A, "__file__", "<??>"))
    print("yf file     :", getattr(Y, "__file__", "<??>"))
    if not hasattr(Y, "load_ohlcv"):
        print("FATAL: src.data.yf.load_ohlcv is missing!")
    else:
        print("yf.load_ohlcv: OK (callable)")

    # Force feed=iex for Alpaca leg unless user overrides
    feed = os.environ.get("ALPACA_FEED", "iex")

    # 1) Direct Alpaca leg (will fail if no SDK in workers or wrong feed)
    _leg(
        f"LEG 1: Direct Alpaca (feed={feed})",
        getattr(A, "load_ohlcv"),
        symbol,
        start,
        end,
        "1Day",
        feed=feed,
    )

    # 2) Direct Yahoo leg
    _leg(
        "LEG 2: Direct Yahoo Finance",
        getattr(Y, "load_ohlcv"),
        symbol,
        start,
        end,
        "1d",
        auto_adjust=True,
    )

    # 3) Unified loader (tries Alpaca then YF)
    _leg(
        "LEG 3: Unified loader (get_ohlcv → Alpaca fallback to YF)",
        getattr(L, "get_ohlcv"),
        symbol,
        start,
        end,
    )


if __name__ == "__main__":
    main()