# pages/1_Live_KPIs.py
import json
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Live KPIs", layout="wide")

STORE = Path("storage/live_kpis.json")
STORE.parent.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "equity": 1000.0,
    "max_dd_pct": 0.0,          # e.g. -12.3 for -12.3%
    "sharpe_live": 0.0,
    "sharpe_oos": 0.8,
    "trades_total": 0,
    "exec_slip_bps": 0.0,
    "cost_share": 0.0,          # (comm+slip)/gross PnL
    "signal_compliance": 100.0, # %
}

def load():
    if STORE.exists():
        try:
            return json.loads(STORE.read_text())
        except Exception:
            return dict(DEFAULTS)
    return dict(DEFAULTS)

def save(obj):
    STORE.write_text(json.dumps(obj, indent=2))

def color(val):  # helper for status lights
    return f":green_circle:" if val == "GREEN" else (":yellow_circle:" if val == "YELLOW" else ":red_circle:")

def grade_max_dd(dd_pct):
    # dd_pct should be negative; treat absolute size
    dd = abs(dd_pct)
    if dd <= 12: return "GREEN"
    if dd <= 18: return "YELLOW"
    return "RED"

def grade_sharpe(s):
    if s >= 0.9: return "GREEN"
    if s >= 0.6: return "YELLOW"
    return "RED"

def grade_drift(live, oos):
    if oos is None: return "YELLOW"
    if abs(live - oos) <= 0.2: return "GREEN"
    if abs(live - oos) <= 0.35: return "YELLOW"
    return "RED"

def grade_slippage(bps):
    if bps <= 20: return "GREEN"
    if bps <= 50: return "YELLOW"
    return "RED"

def grade_costshare(x):
    if x <= 0.25: return "GREEN"
    if x <= 0.35: return "YELLOW"
    return "RED"

def grade_compliance(pct):
    if pct >= 98: return "GREEN"
    if pct >= 95: return "YELLOW"
    return "RED"

st.title("Live KPIs (Green / Yellow / Red)")

state = load()
with st.form("edit"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state["equity"] = st.number_input("Account equity ($)", value=float(state["equity"]), min_value=0.0, step=50.0)
        state["max_dd_pct"] = st.number_input("Max drawdown (%)", value=float(state["max_dd_pct"]), step=0.1, help="Use negative for losses, e.g., -12.0")
    with c2:
        state["sharpe_live"] = st.number_input("Live Sharpe", value=float(state["sharpe_live"]), step=0.05)
        state["sharpe_oos"] = st.number_input("OOS Sharpe (target)", value=float(state["sharpe_oos"]), step=0.05)
    with c3:
        state["trades_total"] = st.number_input("Total trades (period)", value=int(state["trades_total"]), step=10)
        state["exec_slip_bps"] = st.number_input("Avg execution slippage (bps)", value=float(state["exec_slip_bps"]), step=1.0)
    with c4:
        state["cost_share"] = st.number_input("Cost share of gross P&L", value=float(state["cost_share"]), step=0.05, help="(commissions+slippage)/gross P&L")
        state["signal_compliance"] = st.number_input("Signal compliance (%)", value=float(state["signal_compliance"]), step=0.5)

    left, right = st.columns([1,1])
    with left:
        save_btn = st.form_submit_button("💾 Save KPIs")
    with right:
        reset = st.form_submit_button("↩️ Reset to defaults")
        if reset:
            state = dict(DEFAULTS)
            save(state)

if save_btn:
    save(state)
    st.success("Saved.")

# --- grading
grades = {
    "Max DD": grade_max_dd(state["max_dd_pct"]),
    "Sharpe (live)": grade_sharpe(state["sharpe_live"]),
    "Live vs OOS drift": grade_drift(state["sharpe_live"], state.get("sharpe_oos")),
    "Execution slippage": grade_slippage(state["exec_slip_bps"]),
    "Cost share": grade_costshare(state["cost_share"]),
    "Signal compliance": grade_compliance(state["signal_compliance"]),
}

gc = list(grades.values()).count("GREEN")
yc = list(grades.values()).count("YELLOW")
rc = list(grades.values()).count("RED")

st.subheader(f"Status: {gc} {color('GREEN')}  |  {yc} {color('YELLOW')}  |  {rc} {color('RED')}")
st.divider()

cols = st.columns(3)
items = list(grades.items())
for i, (name, g) in enumerate(items):
    with cols[i % 3]:
        st.metric(name, grades[name], help=None)
        st.write(color(g), name)

st.caption("Thresholds come from your `live_checklist.md`. Adjust if your risk policy changes.")