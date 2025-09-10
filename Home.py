import streamlit as st
import pandas as pd
from src.storage import list_portfolios, list_simulations

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()  # ensure .env is read when launching Streamlit

#####Helpers
def get_alpaca_status():
    try:
        api = os.getenv("ALPACA_API_KEY")
        sec = os.getenv("ALPACA_SECRET_KEY")
        if not api or not sec:
            return {"ok": False, "error": "Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in .env or Streamlit secrets."}
        client = TradingClient(api, sec, paper=True)
        acct = client.get_account()
        return {"ok": True, "status": str(acct.status), "equity": float(acct.equity)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


st.set_page_config(page_title="Trading Dashboard", layout="wide", page_icon="📊")

st.title("📊 Trading Research Dashboard")

with st.sidebar:
    st.subheader("Alpaca")
    s = get_alpaca_status()
    if s.get("ok"):
        st.success(f"Connected ({s['status']})")
        st.write(f"Equity: ${s['equity']:,.2f}")
        st.caption("Feed: IEX")
    else:
        st.error("Not connected")
        st.caption(s.get("error", ""))

portfolios = list_portfolios()
sims = list_simulations(limit=10)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Portfolios", len(portfolios))
with col2:
    total_items = sum(len(p.get("items", [])) for p in portfolios)
    st.metric("Saved Tickers", total_items)
with col3:
    st.metric("Saved Simulations", len(sims))

st.subheader("Recent Simulations")
if not sims:
    st.info("No simulations yet. Go to **Simulate Portfolio** to run one.")
else:
    rows = []
    for r in sims:
        rows.append({
            "when": r.get("created_at","")[:19].replace("T"," "),
            "portfolio": r.get("portfolio_name",""),
            "range": f"{r.get('start','?')} → {r.get('end','?')}",
            "start_eq": r.get("starting_equity", 0),
            "final_eq": r.get("final_equity", 0),
            "total_return_%": round(100 * (r.get("final_equity",0)/max(r.get("starting_equity",1),1)-1), 2)
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()
st.markdown("""
**Workflow**
1. **Ticker Selector & Tuning**: try a symbol and parameters.
2. Save the best combo to a **Portfolio** (create one if needed).
3. **Simulate Portfolio**: pick a portfolio + date range + starting equity and run.
4. Iterate here.
""")
