# pages/4_Simulate_Portfolio.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Dynamic allocation helpers (beta) ---
def _wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _entry_exit_levels(df: pd.DataFrame, breakout_n: int, exit_n: int) -> pd.DataFrame:
    out = df.copy()
    out["breakout_high"] = out["high"].rolling(window=breakout_n, min_periods=breakout_n).max().shift(1)
    out["exit_low"] = out["low"].rolling(window=exit_n, min_periods=exit_n).min().shift(1)
    out["next_open"] = out["open"].shift(-1)
    return out

def _prepare_frame(symbol: str, start_iso: str, end_iso: str, params: dict) -> pd.DataFrame:
    df = get_ohlcv_cached(symbol, start_iso, end_iso).copy()
    if df.empty:
        return df
    df["atr"] = _wilder_atr(df, int(params.get("atr_n", 14)))
    df = _entry_exit_levels(df, int(params.get("breakout_n", 55)), int(params.get("exit_n", 20)))
    # Signals defined on bar t for execution at t+1 open
    df["entry_sig"] = (df["close"] > df["breakout_high"])
    df["exit_sig"] = (df["close"] < df["exit_low"])
    # Scores
    metric = st.session_state.get("_dyn_score_metric", "Breakout distance / ATR")
    if metric == "20d momentum / ATR":
        mom = df["close"].pct_change(20)
        atr_pct = (df["atr"] / df["close"]).replace([0, np.inf, -np.inf], np.nan)
        score = (mom / atr_pct).replace([np.inf, -np.inf], np.nan)
    else:
        # default: Breakout distance / ATR
        score = (df["close"] - df["breakout_high"]) / df["atr"]
    df["score"] = score.clip(lower=0).fillna(0.0)
    return df

def _simulate_dynamic(items, strategy_selections, start, end, starting_equity,
                      deploy_pct: int, max_concurrent: int, min_alloc: float, score_metric: str):
    """
    Simple multi-asset daily scheduler:
      - On day t, we EXECUTE exits and entries that were SIGNED on day t-1 at today's OPEN.
      - Entry candidates share a budget = deploy_pct% of equity available; capital split ∝ score.
      - Max concurrent positions enforced.
      - Exit when exit_sig triggered (Donchian-like), executed next morning open.
    """
    st.session_state["_dyn_score_metric"] = score_metric  # used in _prepare_frame
    start_iso, end_iso = start.isoformat(), end.isoformat()
    # Prepare frames and params per symbol
    sym_params = {}
    frames = {}
    for it in items:
        symbol = it["symbol"]
        chosen = strategy_selections.get(symbol, "PORTFOLIO_DEFAULT")
        if chosen != "PORTFOLIO_DEFAULT":
            srec = get_strategy(chosen)
            params = srec["params"] if srec else it["params"]
        else:
            params = it["params"]
        sym_params[symbol] = params
        frames[symbol] = _prepare_frame(symbol, start_iso, end_iso, params)

    # Build joint calendar
    all_idx = sorted(set().union(*[set(df.index) for df in frames.values() if not df.empty]))
    if not all_idx:
        return {"equity": pd.Series(dtype=float, name="equity"), "per_ticker": []}

    cash = float(starting_equity)
    positions = {s: {"shares": 0.0, "entry_px": None} for s in frames.keys()}
    equity_series = []
    dates = []

    pending_buys = {}   # symbol -> size hint (score at t-1)
    pending_sells = set()

    for i, dt in enumerate(all_idx):
        # EXECUTE scheduled orders at today's open
        # Sells first
        for s in list(pending_sells):
            df = frames[s]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            if positions[s]["shares"] > 0 and not np.isnan(row["open"]):
                px = float(row["open"])
                cash += positions[s]["shares"] * px
                positions[s] = {"shares": 0.0, "entry_px": None}
        pending_sells.clear()

        # Then buys (allocate from today's cash)
        if pending_buys:
            # Available budget
            deploy_cash = cash * (deploy_pct / 100.0)
            # Filter to symbols still valid (today has data)
            valid = {s: sc for s, sc in pending_buys.items() if dt in frames[s].index}
            pending_buys.clear()
            if valid and deploy_cash > 0:
                # Enforce max concurrent
                open_count = sum(1 for s in positions if positions[s]["shares"] > 0)
                slots = max_concurrent - open_count
                if slots > 0:
                    # Rank by score desc and take top 'slots'
                    ranked = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:slots]
                    total_score = sum(sc for _, sc in ranked) or 1.0
                    for s, sc in ranked:
                        row = frames[s].loc[dt]
                        px = float(row["open"])
                        alloc = max(min_alloc, deploy_cash * (sc / total_score))
                        if alloc <= cash and px > 0:
                            sh = alloc / px
                            cash -= sh * px
                            positions[s] = {"shares": positions[s]["shares"] + sh, "entry_px": px}

        # SCHEDULE next-day orders based on today's signals
        for s, df in frames.items():
            if dt not in df.index:
                continue
            row = df.loc[dt]
            # schedule sell if in position and exit_sig today
            if positions[s]["shares"] > 0 and bool(row.get("exit_sig", False)):
                pending_sells.add(s)
            # schedule buy if flat and entry_sig today
            if positions[s]["shares"] == 0 and bool(row.get("entry_sig", False)):
                pending_buys[s] = float(row.get("score", 0.0))

        # Mark-to-market equity at today's close
        mtm = cash
        for s, pos in positions.items():
            if pos["shares"] > 0:
                df = frames[s]
                if dt in df.index:
                    mtm += pos["shares"] * float(df.loc[dt, "close"])
        dates.append(pd.Timestamp(dt))
        equity_series.append(float(mtm))

    equity = pd.Series(equity_series, index=pd.to_datetime(dates), name="equity")
    # Basic per-ticker summaries (final value per open positions counted at last close)
    per_ticker = []
    for s, df in frames.items():
        # Approximate ticker return as contribution vs equal split of start (rough)
        # For display only; full trade logs would need additional bookkeeping
        per_ticker.append({"symbol": s, "total_return": np.nan, "sharpe": np.nan,
                           "max_drawdown": np.nan, "final_equity": np.nan,
                           "start": start_iso, "end": end_iso})
    return {"equity": equity, "per_ticker": per_ticker}

from src.storage import (
    list_portfolios,
    add_simulation,
    list_strategies,
    get_strategy,
    get_default_strategy,
)
from src.models.atr_breakout import backtest_single
from src.utils.plotting import equity_chart
from src.data.cache import get_ohlcv_cached

st.set_page_config(page_title="Simulate Portfolio", page_icon="🧮")

st.title("🧮 Simulate Portfolio")

# --- Load portfolios ---
portfolios = list_portfolios()
if not portfolios:
    st.info("Create a portfolio first (use the Tuning page → Save to Portfolio).")
    st.stop()

names = [p["name"] for p in portfolios]
choice = st.selectbox("Portfolio", names, index=0)
p = next((x for x in portfolios if x["name"] == choice), None)
items = p.get("items", [])

if not items:
    st.warning("This portfolio has no items yet.")
    st.stop()

# --- Global sim inputs ---
colA, colB, colC = st.columns(3)
with colA:
    start = st.date_input("Start Date", value=date.today() - timedelta(days=365 * 3))
with colB:
    end = st.date_input("End Date", value=date.today())
with colC:
    starting_equity = st.number_input("Starting Equity ($)", min_value=1000, value=50_000, step=1000)

use_risk_parity = st.checkbox("Use risk-parity weighting (1/vol)", value=True)

# --- Allocation mode UI ---
alloc_mode = st.radio("Allocation mode", ["Static weights (equal / 1/vol)", "Dynamic: opportunity-weighted (beta)"], index=0)
if alloc_mode.startswith("Dynamic"):
    colD1, colD2, colD3 = st.columns(3)
    with colD1:
        deploy_pct = st.slider("Daily deploy cap (% of equity)", 5, 100, 30, 5)
    with colD2:
        max_concurrent = st.slider("Max concurrent positions", 1, 20, 5, 1)
    with colD3:
        min_alloc = st.number_input("Min allocation per trade ($)", min_value=0, value=1000, step=500)
    score_metric = st.selectbox("Signal score", ["Breakout distance / ATR", "20d momentum / ATR"], index=0)

st.subheader(f"Tickers in “{p['name']}”")

# --- Per-ticker strategy selection (default strategy first) ---
strategy_selections = {}
for it in items:
    symbol = it["symbol"]
    model = it["model"]

    strategies = list_strategies(symbol=symbol, model=model)
    default_strat = get_default_strategy(symbol, model)

    opts = []
    ids = []

    if default_strat:
        opts.append(f"★ Default: {default_strat['name']}  ({default_strat['id'][:8]})")
        ids.append(default_strat["id"])

    for s in strategies:
        if not default_strat or s["id"] != default_strat["id"]:
            opts.append(f"{s['name']}  ({s['id'][:8]})")
            ids.append(s["id"])

    # Always include the portfolio-saved params option
    opts.append("Use portfolio params")
    ids.append("PORTFOLIO_DEFAULT")

    default_idx = 0  # show default strategy first if exists
    sel = st.selectbox(f"{symbol} — strategy", opts, index=default_idx, key=f"strategy_{p['id']}_{symbol}")
    strategy_selections[symbol] = ids[opts.index(sel)]

st.write("---")

# --- Run Simulation ---
if st.button("Run Simulation", type="primary"):
    with st.spinner("Simulating..."):
        try:
            # Choose simulation path
            if alloc_mode.startswith("Dynamic"):
                result = _simulate_dynamic(items, strategy_selections, start, end, starting_equity,
                                           deploy_pct, max_concurrent, float(min_alloc), score_metric)
                total_eq = result["equity"]
                per_ticker = result["per_ticker"]
            else:
                per_ticker = []
                curves = []

                # ------- Compute portfolio weights -------
                symbols = [it["symbol"] for it in items]

                if use_risk_parity:
                    vols = {}
                    for s in symbols:
                        try:
                            dfp = get_ohlcv_cached(s, start.isoformat(), end.isoformat())
                            ret = dfp["close"].pct_change().dropna()
                            vol = float(ret.std())  # daily stdev; relative only
                            if not np.isfinite(vol) or vol <= 0:
                                vol = np.nan
                        except Exception:
                            vol = np.nan
                        vols[s] = vol

                    inv_vol = {s: (1.0 / v) if (v and np.isfinite(v) and v > 0) else np.nan for s, v in vols.items()}
                    if all(np.isnan(x) for x in inv_vol.values()):
                        # fallback: equal weights
                        weights = {s: 1.0 / max(len(symbols), 1) for s in symbols}
                    else:
                        valid = {s: x for s, x in inv_vol.items() if np.isfinite(x)}
                        total = sum(valid.values())
                        weights = {s: (valid.get(s, 0.0) / total) for s in symbols}
                else:
                    weights = {s: 1.0 / max(len(symbols), 1) for s in symbols}

                # ------- Per-ticker backtests (weighted capital) -------
                curves = []
                for it in items:
                    symbol = it["symbol"]
                    alloc = float(starting_equity) * float(weights.get(symbol, 0.0))

                    chosen = strategy_selections.get(symbol, "PORTFOLIO_DEFAULT")
                    if chosen != "PORTFOLIO_DEFAULT":
                        srec = get_strategy(chosen)
                        params = srec["params"] if srec else it["params"]
                    else:
                        params = it["params"]

                    res = backtest_single(
                        symbol,
                        start.isoformat(),
                        end.isoformat(),
                        breakout_n=int(params.get("breakout_n", 55)),
                        exit_n=int(params.get("exit_n", 20)),
                        atr_n=int(params.get("atr_n", 14)),
                        starting_equity=alloc,  # weighted capital
                        atr_multiple=float(params.get("atr_multiple", 3.0)),
                        risk_per_trade=float(params.get("risk_per_trade", 0.01)),
                        allow_fractional=bool(params.get("allow_fractional", True)),
                    )
                    eq = res["equity"].rename(symbol)
                    curves.append(eq)
                    per_ticker.append({"symbol": symbol, **res["metrics"]})

                # Align and sum
                aligned = pd.concat(curves, axis=1).fillna(method="ffill").dropna(how="all")
                total_eq = aligned.sum(axis=1)

            # ------- Aggregate & display -------
            if total_eq is None or total_eq.empty:
                st.error("No equity curves produced.")
                st.stop()

            st.plotly_chart(
                equity_chart(total_eq, title=f"Total Equity — {p['name']}"),
                use_container_width=True,
            )

            out = pd.DataFrame(per_ticker).sort_values("total_return", ascending=False)
            out["total_return_%"] = (out["total_return"] * 100).round(2)
            st.subheader("Per-Ticker Performance")
            st.dataframe(
                out[["symbol", "total_return_%", "sharpe", "max_drawdown", "final_equity", "start", "end"]],
                use_container_width=True,
            )

            # Save a summary record of this simulation
            add_simulation({
                "portfolio_id": p["id"],
                "portfolio_name": p["name"],
                "start": start.isoformat(),
                "end": end.isoformat(),
                "starting_equity": float(starting_equity),
                "final_equity": float(total_eq.iloc[-1]),
                "use_risk_parity": bool(use_risk_parity),
            })

            st.success("Simulation saved.")
        except Exception as e:
            st.error(f"Error: {e}")