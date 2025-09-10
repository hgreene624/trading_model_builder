# Starter Implementation (v1)

This is a minimal working version of the Streamlit trading research app you scaffolded.

**Pages**
- Home (dashboard)
- Ticker Selector & Tuning (single-ticker backtest and save to portfolio)
- Portfolios (manage portfolios & items)
- Simulate Portfolio (run a portfolio backtest; chart + per-ticker table)

**Data**
- Pulls daily OHLCV from Alpaca (paper) if API key/secret are provided via `.env` or Streamlit secrets.
- Falls back to yfinance if Alpaca is unavailable.

## Setup
1) Put your Alpaca paper keys in either **.env** or **.streamlit/secrets.toml**:
```
# .env
ALPACA_API_KEY=YOUR_KEY
ALPACA_SECRET_KEY=YOUR_SECRET
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
```
or
```
# .streamlit/secrets.toml
ALPACA_API_KEY = "YOUR_KEY"
ALPACA_SECRET_KEY = "YOUR_SECRET"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"
```

2) Install & run:
```
pip install -r requirements.txt
streamlit run Home.py
```
