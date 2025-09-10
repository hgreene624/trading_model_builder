# quick_check_alpaca.py
import os
from datetime import date, timedelta
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"[dotenv] loaded: {env_path}")
else:
    print("[dotenv] no .env found via find_dotenv(); continuing without it")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not SECRET_KEY:
    raise SystemExit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. Check your .env.")

# 1) Trading API: verify account (paper)
from alpaca.trading.client import TradingClient
trading = TradingClient(API_KEY, SECRET_KEY, paper=True)
acct = trading.get_account()
print("[Trading] account status:", acct.status, "| equity:", acct.equity)

# 2) Market Data API: fetch daily bars (Alpaca-only; no yfinance fallback)
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

symbol = os.getenv("SYMBOL", "AAPL")
end = date.today()
start = end - timedelta(days=30)

req = StockBarsRequest(
    symbol_or_symbols=[symbol],
    timeframe=TimeFrame.Day,
    feed=DataFeed.IEX,  # <- use free IEX feed
    start=pd.Timestamp(start).to_pydatetime(),
    end=(pd.Timestamp(end) + pd.Timedelta(days=1)).to_pydatetime(),  # inclusive end
)

bars = data_client.get_stock_bars(req)

# Try the modern .df attribute first
if hasattr(bars, "df") and not bars.df.empty:
    df = bars.df
    if df.index.nlevels == 2:
        df = df.xs(symbol, level=0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.index = pd.to_datetime(out.index.date)
    out.index.name = "date"
    print(f"[Data] {symbol} rows:", len(out))
    print(out.tail().to_string())
else:
    # Fallback for older SDK response shapes
    rows = []
    try:
        seq = bars[symbol]
    except Exception:
        seq = getattr(bars, symbol, [])
    for b in seq:
        ts = pd.Timestamp(getattr(b, "timestamp", None)).tz_localize(None)
        rows.append({"date": ts.date(), "open": float(b.open), "high": float(b.high), "low": float(b.low),
                     "close": float(b.close), "volume": int(b.volume)})
    print(f"[Data] {symbol} rows:", len(rows))
    for r in rows[-5:]:
        print(r)