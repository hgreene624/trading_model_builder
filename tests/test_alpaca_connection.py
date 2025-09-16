# tests/test_alpaca_connection.py
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import datetime, timedelta, UTC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def banner(t): print("\n" + "="*72 + f"\n{t}\n" + "="*72)

def load_streamlit_secrets_into_env():
    p = ROOT / ".streamlit" / "secrets.toml"
    if not p.exists():
        return "none"
    try:
        import tomllib
    except Exception:
        import tomli as tomllib
    data = tomllib.loads(p.read_text())
    # prefer ALPACA_* but accept APCA_*
    os.environ.setdefault("ALPACA_API_KEY", data.get("ALPACA_API_KEY", data.get("APCA_API_KEY_ID","")))
    os.environ.setdefault("ALPACA_SECRET_KEY", data.get("ALPACA_SECRET_KEY", data.get("APCA_API_SECRET_KEY","")))
    os.environ.setdefault("ALPACA_BASE_URL", data.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    os.environ.setdefault("ALPACA_DATA_URL", data.get("ALPACA_DATA_URL", "https://data.alpaca.markets"))
    return "file"

def main():
    src = load_streamlit_secrets_into_env()
    banner("Environment")
    print("Python:", sys.executable)
    print("Secrets source:", src)
    print("ALPACA_API_KEY:", (os.environ.get("ALPACA_API_KEY") or "<EMPTY>")[:4] + "..." )
    print("ALPACA_SECRET_KEY:", ("*"*36 + (os.environ.get("ALPACA_SECRET_KEY","")[-4:])))
    print("ALPACA_BASE_URL:", os.environ.get("ALPACA_BASE_URL"))

    banner("Trading Client (alpaca-py)")
    from alpaca.trading.client import TradingClient
    t = TradingClient(api_key=os.environ["ALPACA_API_KEY"], secret_key=os.environ["ALPACA_SECRET_KEY"], paper=True)
    acct = t.get_account()
    print("status:", acct.status, " cash:", acct.cash, " pdt:", acct.pattern_day_trader)

    try:
        clock = t.get_clock()
        print("market is_open:", clock.is_open, " next_open:", clock.next_open, " next_close:", clock.next_close)
    except Exception as e:
        print("Clock error:", repr(e))

    banner("Historical Data Client (alpaca-py, IEX)")
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.models.bars import Adjustment
    from alpaca.data.enums import DataFeed

    h = StockHistoricalDataClient(api_key=os.environ["ALPACA_API_KEY"], secret_key=os.environ["ALPACA_SECRET_KEY"])

    end = datetime.now(UTC)
    start = end - timedelta(days=400)
    req = StockBarsRequest(
        symbol_or_symbols="AAPL",
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.RAW,
        feed=DataFeed.IEX,
    )
    bars = h.get_stock_bars(req)
    df = bars.df
    # reduce to single symbol index if multi
    try:
        if not df.empty and getattr(df.index, "names", [None])[0] == "symbol":
            df = df.xs("AAPL", level="symbol")
    except Exception:
        pass
    print(df.head(5).to_string() if df is not None and not df.empty else "No bars returned.")

if __name__ == "__main__":
    main()