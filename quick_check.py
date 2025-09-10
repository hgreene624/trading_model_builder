# quick_check.py  (place at the project root)
import os
from datetime import date, timedelta
import pandas as pd

def main():
    # Change ticker if you like
    symbol = os.getenv("SYMBOL", "AAPL")
    end = date.today()
    start = end - timedelta(days=365)

    try:
        from src.data.loader import get_ohlcv
    except ImportError:
        # If you skipped loader.py, call Alpaca directly:
        from src.data.alpaca_data import load_ohlcv as get_ohlcv

    df = get_ohlcv(symbol, start.isoformat(), end.isoformat())
    print(f"Fetched {symbol} rows: {len(df)}")
    print(df.head(5).to_string())

if __name__ == "__main__":
    main()