import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.backtest.metrics import summarize_costs

# --- Dynamic allocation helpers (beta) ---
def _wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = (-low.diff())
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx

def _recent_sharpe(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    r = df["close"].pct_change()
    m = r.rolling(lookback).mean()
    s = r.rolling(lookback).std().replace(0, np.nan)
    sh = (m / s).fillna(0.0)
    return sh

def _entry_exit_levels(df: pd.DataFrame, breakout_n: int, exit_n: int) -> pd.DataFrame:
    out = df.copy()
    out["breakout_high"] = out["high"].rolling(window=breakout_n, min_periods=breakout_n).max().shift(1)
    out["exit_low"] = out["low"].rolling(window=exit_n, min_periods=exit_n).min().shift(1)
    out["next_open"] = out["open"].shift(-1)
    return out

def _compute_trend_cols(df: pd.DataFrame, sma_fast: int, sma_slow: int, sma_long: int, long_slope_len: int) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(sma_fast, min_periods=sma_fast).mean()
    out["sma_slow"] = out["close"].rolling(sma_slow, min_periods=sma_slow).mean()
    out["sma_long"] = out["close"].rolling(sma_long, min_periods=sma_long).mean()
    out["sma_long_prev"] = out["sma_long"].shift(long_slope_len)
    out["long_slope_up"] = (out["sma_long"] - out["sma_long_prev"]) > 0
    out["fast_above_slow"] = out["sma_fast"] > out["sma_slow"]
    out["trend_ok"] = out["fast_above_slow"] & out["long_slope_up"]
    return out

def _prepare_frame(symbol: str, start_iso: str, end_iso: str, params: dict) -> pd.DataFrame:
    df = get_ohlcv_cached(symbol, start_iso, end_iso).copy()
    if df.empty:
        return df
    atr_n = int(params.get("atr_n", 14))
    breakout_n = int(params.get("breakout_n", 55))
    exit_n = int(params.get("exit_n", 20))
    k_atr_buffer = float(params.get("k_atr_buffer", 0.0) or 0.0)
    if k_atr_buffer < 0.0:
        k_atr_buffer = 0.0
    persist_n = int(params.get("persist_n", 1) or 1)
    if persist_n < 1:
        persist_n = 1
    use_trend = bool(params.get("use_trend_filter", False))
    sma_fast = int(params.get("sma_fast", 30))
    sma_slow = int(params.get("sma_slow", 50))
    sma_long = int(params.get("sma_long", 150))
    long_slope_len = int(params.get("long_slope_len", 15))

    df["atr"] = _wilder_atr(df, atr_n)
    df["adx14"] = _adx(df, 14)
    df["sharpe60"] = _recent_sharpe(df, 60)
    df = _entry_exit_levels(df, breakout_n, exit_n)
    if use_trend:
        df = _compute_trend_cols(df, sma_fast, sma_slow, sma_long, long_slope_len)
    else:
        df["trend_ok"] = True

    # Signals on bar t for execution at t+1 open
    raw_entry = (df["close"] > df["breakout_high"]) & df["trend_ok"]
    if k_atr_buffer > 0.0:
        raw_entry = raw_entry & (df["close"] > (df["breakout_high"] + k_atr_buffer * df["atr"]))
    raw_entry = raw_entry.fillna(False)
    if persist_n > 1:
        entry_sig = (
            raw_entry.astype(int)
            .rolling(window=persist_n, min_periods=persist_n)
            .sum()
            .eq(persist_n)
        ).fillna(False)
    else:
        entry_sig = raw_entry
    df["entry_sig"] = entry_sig.astype(bool)
    df["exit_sig"] = (df["close"] < df["exit_low"])

    # Base score
    metric = st.session_state.get("_dyn_score_metric", "Breakout distance / ATR")
    if metric == "20d momentum / ATR":
        mom = df["close"].pct_change(20)
        atr_pct = (df["atr"] / df["close"]).replace([0, np.inf, -np.inf], np.nan)
        base = (mom / atr_pct).replace([np.inf, -np.inf], np.nan)
    else:
        breakout_ref = df["breakout_high"]
        if k_atr_buffer > 0.0:
            breakout_ref = breakout_ref + k_atr_buffer * df["atr"]
        base = (df["close"] - breakout_ref) / df["atr"]
    df["score_base"] = base.clip(lower=0).fillna(0.0)
    return df

def _simulate_dynamic(items, strategy_selections, start, end, starting_equity,
                      deploy_pct: int, max_concurrent: int, min_alloc: float,
                      score_metric: str, secondary_metric: str):
    """
    Multi-asset daily scheduler (next-open execution):
      - EXECUTE exits & entries signed on t-1 at t OPEN.
      - Entry candidates share budget = deploy_pct% of cash; split ∝ score.
      - Enforces max concurrent positions.
      - Exits: channel exit, ATR stop, TP, holding limit.
    """
    st.session_state["_dyn_score_metric"] = score_metric
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

    # Joint calendar
    all_idx = sorted(set().union(*[set(df.index) for df in frames.values() if not df.empty]))
    if not all_idx:
        return {"equity": pd.Series(dtype=float, name="equity"), "per_ticker": []}

    cash = float(starting_equity)
    positions = {s: {"shares": 0.0, "entry_px": None, "entry_dt": None, "stop_px": None, "tp_px": None, "bars": 0}
                 for s in frames.keys()}
    equity_series, dates = [], []
    pending_buys = {}   # symbol -> (score, tie)
    pending_sells = set()
    trades_by_symbol = {s: [] for s in frames.keys()}

    def _tie_value(row):
        if secondary_metric == "ADX(14)":
            return float(row.get("adx14", 0.0) or 0.0)
        elif secondary_metric == "Recent Sharpe(60d)":
            return float(row.get("sharpe60", 0.0) or 0.0)
        else:
            return 0.0

    for dt in all_idx:
        # Execute scheduled sells at today's open
        for s in list(pending_sells):
            df = frames[s]
            if dt not in df.index:
                continue
            row = df.loc[dt]
            if positions[s]["shares"] > 0 and not np.isnan(row["open"]):
                px = float(row["open"])
                sh = positions[s]["shares"]
                entry_px = positions[s]["entry_px"] or px
                entry_dt = positions[s]["entry_dt"] or pd.Timestamp(dt)
                pnl = (px - entry_px) * sh
                ret_pct = (px / entry_px) - 1.0 if entry_px > 0 else 0.0
                trades_by_symbol[s].append({
                    "symbol": s,
                    "entry_date": entry_dt.date(),
                    "entry_price": float(entry_px),
                    "exit_date": pd.Timestamp(dt).date(),
                    "exit_price": float(px),
                    "shares": float(sh),
                    "pnl": float(pnl),
                    "return_pct": float(ret_pct),
                    "holding_days": int((pd.Timestamp(dt) - entry_dt).days),
                    "reason": "scheduled_exit",
                })
                cash += sh * px
                positions[s] = {"shares": 0.0, "entry_px": None, "entry_dt": None, "stop_px": None, "tp_px": None, "bars": 0}
        pending_sells.clear()

        # Execute scheduled buys at today's open
        if pending_buys:
            deploy_cash = cash * (deploy_pct / 100.0)
            valid_today = {s: v for s, v in pending_buys.items() if dt in frames[s].index}
            pending_buys.clear()
            if valid_today and deploy_cash > 0:
                # Respect max slots
                open_count = sum(1 for s in positions if positions[s]["shares"] > 0)
                slots = max_concurrent - open_count
                if slots > 0:
                    # Rank by score then tie
                    ranked = sorted(valid_today.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)[:slots]
                    total_score = sum(sc for (sc, tie) in [v for _, v in ranked]) or 1.0
                    for s, (sc, tie) in ranked:
                        row = frames[s].loc[dt]
                        px = float(row["open"])
                        alloc = max(min_alloc, deploy_cash * (sc / total_score))
                        alloc = min(alloc, cash)
                        if alloc > 0 and px > 0:
                            sh = alloc / px
                            cash -= sh * px
                            atr_val = float(row.get("atr", np.nan))
                            tp_mult = float(sym_params[s].get("tp_multiple", 0.0) or 0.0)
                            atr_mult = float(sym_params[s].get("atr_multiple", 3.0))
                            tp_px = (px + tp_mult * atr_val) if (tp_mult and atr_val == atr_val) else None
                            stop_px = (px - atr_mult * atr_val) if (atr_mult and atr_val == atr_val) else None
                            positions[s] = {"shares": positions[s]["shares"] + sh, "entry_px": px, "entry_dt": pd.Timestamp(dt),
                                            "stop_px": stop_px, "tp_px": tp_px, "bars": 0}

        # Schedule next-day orders based on today's bar
        for s, df in frames.items():
            if dt not in df.index:
                continue
            row = df.loc[dt]
            pos = positions[s]
            # schedule sell: exit rules
            if pos["shares"] > 0:
                exit_flag = False
                # channel exit
                if bool(row.get("exit_sig", False)):
                    exit_flag = True
                # ATR stop (intraday low breach)
                if pos["stop_px"] is not None and float(row["low"]) <= float(pos["stop_px"]):
                    exit_flag = True
                # TP (intraday high breach)
                if pos["tp_px"] is not None and float(row["high"]) >= float(pos["tp_px"]):
                    exit_flag = True
                # holding limit
                hpl = int(sym_params[s].get("holding_period_limit", 0) or 0)
                if hpl > 0 and (pos["bars"] + 1) >= hpl:
                    exit_flag = True
                if exit_flag:
                    pending_sells.add(s)
            else:
                # schedule buy if flat and entry signal
                if bool(row.get("entry_sig", False)):
                    base = float(row.get("score_base", 0.0) or 0.0)
                    if base > 0:
                        tie = 0.0
                        if secondary_metric == "ADX(14)":
                            tie = float(row.get("adx14", 0.0) or 0.0)
                        elif secondary_metric == "Recent Sharpe(60d)":
                            tie = float(row.get("sharpe60", 0.0) or 0.0)
                        pending_buys[s] = (base, tie)

        # Mark-to-market equity at today's close & age positions
        mtm = cash
        for s, pos in positions.items():
            if pos["shares"] > 0 and dt in frames[s].index:
                mtm += pos["shares"] * float(frames[s].loc[dt, "close"])
                positions[s]["bars"] += 1
        dates.append(pd.Timestamp(dt))
        equity_series.append(float(mtm))

    equity = pd.Series(equity_series, index=pd.to_datetime(dates), name="equity")

    # Per-ticker stats from trade logs
    per_ticker = []
    for s, logs in trades_by_symbol.items():
        if not logs:
            per_ticker.append({"symbol": s, "trades": 0, "win_rate": 0.0, "avg_return": 0.0,
                               "total_pnl": 0.0, "invested": 0.0, "total_return": 0.0})
            continue
        dftr = pd.DataFrame(logs)
        wins = float((dftr["pnl"] > 0).mean())
        avg_ret = float(dftr["return_pct"].mean())
        total_pnl = float(dftr["pnl"].sum())
        invested = float((dftr["entry_price"] * dftr["shares"]).sum())
        tot_ret = float(total_pnl / invested) if invested > 0 else 0.0
        per_ticker.append({
            "symbol": s,
            "trades": int(len(dftr)),
            "win_rate": wins,
            "avg_return": avg_ret,
            "total_pnl": total_pnl,
            "invested": invested,
            "total_return": tot_ret,
        })

    return {"equity": equity, "per_ticker": per_ticker, "trades_by_symbol": trades_by_symbol, "frames": frames}


def _calc_cagr(series: pd.Series) -> float:
    if series is None or len(series) < 2:
        return 0.0
    start_val = float(series.iloc[0]) if len(series) else 0.0
    end_val = float(series.iloc[-1]) if len(series) else 0.0
    if not (start_val > 0 and end_val > 0):
        return 0.0
    try:
        delta_days = (series.index[-1] - series.index[0]).days
    except Exception:
        return 0.0
    if delta_days <= 0:
        return 0.0
    years = delta_days / 365.25
    if years <= 0:
        return 0.0
    return float((end_val / start_val) ** (1 / years) - 1.0)


def _aggregate_cost_summary(results: dict[str, dict]) -> dict:
    if not results:
        return {}

    frames_pre: list[pd.Series] = []
    frames_post: list[pd.Series] = []
    trade_tables: list[pd.DataFrame] = []
    start_equity_total = 0.0

    for symbol, res in results.items():
        eq_post = res.get("equity")
        eq_pre = res.get("equity_pre_cost")
        if isinstance(eq_pre, pd.Series) and not eq_pre.empty:
            frames_pre.append(eq_pre.rename(symbol))
        if isinstance(eq_post, pd.Series) and not eq_post.empty:
            frames_post.append(eq_post.rename(symbol))
            start_equity_total += float(eq_post.iloc[0]) if len(eq_post) else 0.0
        trades_df = res.get("trades_df")
        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
            trade_tables.append(trades_df.assign(symbol=symbol))

    df_pre = pd.concat(frames_pre, axis=1).sort_index() if frames_pre else pd.DataFrame()
    df_post = pd.concat(frames_post, axis=1).sort_index() if frames_post else pd.DataFrame()
    if not df_pre.empty:
        df_pre = df_pre.fillna(method="ffill")
        portfolio_pre = df_pre.sum(axis=1, skipna=True)
    else:
        portfolio_pre = pd.Series(dtype=float)
    if not df_post.empty:
        df_post = df_post.fillna(method="ffill")
        portfolio_post = df_post.sum(axis=1, skipna=True)
    else:
        portfolio_post = pd.Series(dtype=float)

    agg_trades = pd.concat(trade_tables, ignore_index=True) if trade_tables else pd.DataFrame()

    summary = summarize_costs(
        agg_trades if not agg_trades.empty else None,
        portfolio_pre if not portfolio_pre.empty else None,
        portfolio_post if not portfolio_post.empty else None,
    )

    total_slip_cost = float(
        agg_trades.get("entry_slippage_cost", pd.Series(dtype=float)).sum()
        + agg_trades.get("exit_slippage_cost", pd.Series(dtype=float)).sum()
    ) if not agg_trades.empty else 0.0
    fee_cols = [
        agg_trades.get("entry_fee_cost", pd.Series(dtype=float)).sum(),
        agg_trades.get("exit_fee_cost", pd.Series(dtype=float)).sum(),
        agg_trades.get("entry_fixed_cost", pd.Series(dtype=float)).sum(),
        agg_trades.get("exit_fixed_cost", pd.Series(dtype=float)).sum(),
    ] if not agg_trades.empty else [0.0]
    total_fee_cost = float(sum(fee_cols)) if fee_cols else 0.0

    turnover_gross = float(summary.get("turnover_gross", 0.0))
    start_equity_total = start_equity_total or 0.0
    turnover_ratio = float(summary.get("turnover_ratio", 0.0) or 0.0)
    if turnover_ratio == 0.0 and start_equity_total:
        turnover_ratio = (turnover_gross / start_equity_total) if start_equity_total else 0.0

    turnover_multiple = float(summary.get("turnover_multiple", 0.0) or 0.0)
    if turnover_multiple == 0.0 and start_equity_total:
        turnover_multiple = turnover_gross / start_equity_total if start_equity_total else 0.0

    out = {k: float(v) for k, v in summary.items()}
    out["turnover_ratio"] = float(turnover_ratio)
    out["turnover"] = float(turnover_ratio)
    out["turnover_multiple"] = float(turnover_multiple)
    out["weighted_slippage_bps"] = float(out.get("slippage_bps_weighted", 0.0))
    out["weighted_fees_bps"] = float(out.get("fees_bps_weighted", 0.0))
    if "annualized_drag_bps" in out:
        out["annualized_drag"] = float(out["annualized_drag_bps"] / 10_000.0)
    out["total_slippage_cost"] = float(total_slip_cost)
    out["total_fee_cost"] = float(total_fee_cost)
    out["total_cost"] = float(total_slip_cost + total_fee_cost)

    alpha_retention = None
    sharpe_gross = out.get("sharpe_gross", summary.get("sharpe_gross"))
    sharpe_net = out.get("sharpe_net", summary.get("sharpe_net"))
    try:
        if sharpe_gross is not None and sharpe_net is not None and abs(float(sharpe_gross)) > 1e-12:
            alpha_retention = float(sharpe_net) / float(sharpe_gross)
    except (TypeError, ValueError):
        alpha_retention = None
    if alpha_retention is None:
        cagr_gross = out.get("cagr_gross", out.get("pre_cost_cagr"))
        cagr_net = out.get("cagr_net", out.get("post_cost_cagr"))
        try:
            if cagr_gross is not None and cagr_net is not None and abs(float(cagr_gross)) > 1e-12:
                alpha_retention = float(cagr_net) / float(cagr_gross)
        except (TypeError, ValueError):
            alpha_retention = None
    out["alpha_retention_ratio"] = float(alpha_retention) if alpha_retention is not None else None

    return out


def _format_cost_table(summary: dict) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame(columns=["Metric", "Value"])
    rows = [
        ("Turnover (gross $)", f"${summary.get('turnover_gross', 0.0):,.2f}"),
        ("Turnover (avg daily $)", f"${summary.get('turnover_avg_daily', 0.0):,.2f}"),
        ("Turnover (x start)", f"{summary.get('turnover_multiple', summary.get('turnover_ratio', summary.get('turnover', 0.0))):.2f}"),
        (
            "Turnover (×/yr)",
            f"{summary.get('turnover_ratio', summary.get('turnover', 0.0)):.2f}",
        ),
        (
            "Alpha Retention %",
            f"{(summary.get('alpha_retention_ratio') or 0.0) * 100:.0f}%"
            if summary.get("alpha_retention_ratio") is not None
            else "—",
        ),
        ("Weighted Slippage (bps)", f"{summary.get('weighted_slippage_bps', 0.0):.2f}"),
        ("Weighted Fees (bps)", f"{summary.get('weighted_fees_bps', 0.0):.2f}"),
        ("Pre-Cost CAGR", f"{summary.get('pre_cost_cagr', 0.0):.2%}"),
        ("Post-Cost CAGR", f"{summary.get('post_cost_cagr', 0.0):.2%}"),
        ("Annualized Drag (bps)", f"{summary.get('annualized_drag_bps', summary.get('annualized_drag', 0.0) * 10_000):.1f}"),
        (
            "Cost per Turnover (bps/1×)",
            f"{summary.get('cost_per_turnover_bps', 0.0):.1f}"
            if summary.get("cost_per_turnover_bps") is not None
            else "—",
        ),
        ("Total Slippage Cost", f"${summary.get('total_slippage_cost', 0.0):,.2f}"),
        ("Total Fee Cost", f"${summary.get('total_fee_cost', 0.0):,.2f}"),
        ("Total Cost", f"${summary.get('total_cost', 0.0):,.2f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])

def _symbol_price_equity_chart(symbol: str, price: pd.Series, total_eq: pd.Series, trades: list[dict] | None = None, title: str | None = None):
    """Dual-axis chart: left=total equity, right=symbol price; markers = Buy/Sell on the price line.

    We align trade timestamps to the price index and place markers at the price value on that date
    so they sit directly on the price curve. If trade price is present, it's ignored for plotting
    to keep markers on the line (still informative via tooltip).
    """
    if title is None:
        title = f"{symbol}: Price & Portfolio Equity"

    # Ensure Series and aligned index
    price = price.copy()
    total_eq = total_eq.copy()
    df = pd.concat({"equity": total_eq, "price": price}, axis=1).dropna(how="all")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Equity on left y
    fig.add_trace(go.Scatter(x=df.index, y=df["equity"], name="Total Equity", mode="lines"), secondary_y=False)
    # Price on right y
    fig.add_trace(go.Scatter(x=df.index, y=df["price"], name=f"{symbol} Price", mode="lines"), secondary_y=True)

    def _first_present(d: dict, keys: list[str]):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def _align_ts(ts_like, idx: pd.DatetimeIndex):
        if ts_like is None:
            return None
        try:
            t = pd.to_datetime(ts_like)
        except Exception:
            return None
        # If exact match
        if t in idx:
            return t
        # Otherwise snap forward to the next available index (or last if beyond)
        pos = idx.searchsorted(t)
        if pos < len(idx):
            return idx[pos]
        return idx[-1] if len(idx) else None

    # Trade markers on price axis (place on the price curve)
    if trades:
        buys_x, buys_y, sells_x, sells_y, buy_text, sell_text = [], [], [], [], [], []
        idx = price.index
        for t in trades:
            # Support multiple schema variants
            bx_raw = _first_present(t, ["entry_date", "entry_dt", "entry", "buy_date"])  # date-like
            sx_raw = _first_present(t, ["exit_date", "exit_dt", "exit", "sell_date"])    # date-like
            bx = _align_ts(bx_raw, idx)
            sx = _align_ts(sx_raw, idx)
            if bx is not None and bx in price.index:
                buys_x.append(bx)
                buys_y.append(float(price.loc[bx]))
                # Tooltip text (show the trade price if present)
                ep = _first_present(t, ["entry_price", "entry_px", "buy_price"]) or ""
                buy_text.append(f"Entry @ {ep}")
            if sx is not None and sx in price.index:
                sells_x.append(sx)
                sells_y.append(float(price.loc[sx]))
                xp = _first_present(t, ["exit_price", "exit_px", "sell_price"]) or ""
                sell_text.append(f"Exit @ {xp}")
        if buys_x:
            fig.add_trace(
                go.Scatter(
                    x=buys_x, y=buys_y, mode="markers", name="Buy",
                    marker_symbol="triangle-up", marker_size=10,
                    marker_color="green", marker_line_color="green", marker_line_width=1,
                    text=buy_text, hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
                ),
                secondary_y=True,
            )
        if sells_x:
            fig.add_trace(
                go.Scatter(
                    x=sells_x, y=sells_y, mode="markers", name="Sell",
                    marker_symbol="triangle-down", marker_size=10,
                    marker_color="red", marker_line_color="red", marker_line_width=1,
                    text=sell_text, hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
                ),
                secondary_y=True,
            )

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text=None, title=title)
    fig.update_yaxes(title_text="Equity", secondary_y=False)
    fig.update_yaxes(title_text=f"{symbol} Price", secondary_y=True)
    return fig

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
    secondary_metric = st.selectbox("Secondary ranking (tie-breaker)", ["None", "ADX(14)", "Recent Sharpe(60d)"], index=0)

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
                                           deploy_pct, max_concurrent, float(min_alloc), score_metric, secondary_metric)
                total_eq = result["equity"]
                per_ticker = result["per_ticker"]
                trades_map = result.get("trades_by_symbol", {})
                frames = result.get("frames", {})
            else:
                per_ticker = []
                curves = []
                res_map = {}

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

                    persist_val = int(params.get("persist_n", 1) or 1)
                    if persist_val < 1:
                        persist_val = 1
                    k_buffer = float(params.get("k_atr_buffer", 0.0) or 0.0)
                    if k_buffer < 0.0:
                        k_buffer = 0.0

                    res = backtest_single(
                        symbol,
                        start.isoformat(),
                        end.isoformat(),
                        breakout_n=int(params.get("breakout_n", 55)),
                        exit_n=int(params.get("exit_n", 20)),
                        atr_n=int(params.get("atr_n", 14)),
                        starting_equity=alloc,  # weighted capital
                        atr_multiple=float(params.get("atr_multiple", 3.0)),
                        k_atr_buffer=k_buffer,
                        persist_n=persist_val,
                        risk_per_trade=float(params.get("risk_per_trade", 0.01)),
                        allow_fractional=bool(params.get("allow_fractional", True)),
                    )
                    res_map[symbol] = res
                    eq = res["equity"].rename(symbol)
                    curves.append(eq)
                    per_ticker.append({"symbol": symbol, **res["metrics"]})

                # Align and sum
                aligned = pd.concat(curves, axis=1).fillna(method="ffill").dropna(how="all")
                total_eq = aligned.sum(axis=1)

            cost_summary = {}
            if not alloc_mode.startswith("Dynamic"):
                try:
                    cost_summary = _aggregate_cost_summary(res_map)
                except Exception:
                    cost_summary = {}

            # ------- Aggregate & display -------
            if total_eq is None or total_eq.empty:
                st.error("No equity curves produced.")
                st.stop()

            st.plotly_chart(
                equity_chart(total_eq, title=f"Total Equity — {p['name']}"),
                width="stretch",
            )

            with st.expander("Cost Attribution (post-cost)", expanded=False):
                if cost_summary:
                    st.table(_format_cost_table(cost_summary))
                else:
                    st.info("No cost attribution available for this run.")

            out = pd.DataFrame(per_ticker)
            st.subheader("Per-Ticker Performance")
            if alloc_mode.startswith("Dynamic"):
                if not out.empty:
                    out = out.sort_values("total_pnl", ascending=False)
                    out["win_rate_%"] = (out["win_rate"] * 100).round(1)
                    out["avg_return_%"] = (out["avg_return"] * 100).round(2)
                    out["total_return_%"] = (out["total_return"] * 100).round(2)
                    st.dataframe(
                        out[["symbol", "trades", "win_rate_%", "avg_return_%", "total_pnl", "invested", "total_return_%"]],
                        width="stretch",
                    )
                else:
                    st.write("No trades executed.")
            else:
                if not out.empty:
                    out = out.sort_values("total_return", ascending=False)
                    out["total_return_%"] = (out["total_return"] * 100).round(2)
                    st.dataframe(
                        out[["symbol", "total_return_%", "sharpe", "max_drawdown", "final_equity", "start", "end"]],
                        width="stretch",
                    )
                else:
                    st.write("No trades executed.")

            # --- Per-Ticker Charts (dual-axis with trade markers) ---
            st.write("---")
            st.subheader("Per-Ticker Charts")
            try:
                for it in items:
                    symbol = it["symbol"]

                    # Determine params used for this symbol in the run
                    chosen = strategy_selections.get(symbol, "PORTFOLIO_DEFAULT") if 'strategy_selections' in locals() else "PORTFOLIO_DEFAULT"
                    if chosen != "PORTFOLIO_DEFAULT":
                        srec = get_strategy(chosen)
                        params = srec["params"] if srec else it["params"]
                    else:
                        params = it["params"]

                    # Price series for right axis
                    df_sym = get_ohlcv_cached(symbol, start.isoformat(), end.isoformat())
                    if df_sym is None or df_sym.empty:
                        continue
                    price_series = df_sym["close"].astype(float)

                    # --- Get trade markers without calling backtest_single ---
                    trades_list: list[dict] = []
                    # Dynamic mode: use trades emitted by the dynamic simulator
                    if 'alloc_mode' in locals() and alloc_mode.startswith("Dynamic"):
                        if 'trades_map' in locals() and isinstance(trades_map, dict):
                            trades_list = trades_map.get(symbol, []) or []
                    else:
                        # Static mode: use trades captured when we ran backtest_single earlier
                        if 'res_map' in locals() and isinstance(res_map, dict) and symbol in res_map:
                            maybe = res_map[symbol]
                            if isinstance(maybe, dict):
                                trades_list = maybe.get("trades", []) or []

                    fig_sym = _symbol_price_equity_chart(symbol, price_series, total_eq, trades_list)
                    st.plotly_chart(fig_sym,  width="stretch")
            except Exception as _e:
                st.warning(f"Per-ticker chart rendering skipped: {_e}")

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

# QA Checklist: in the app, expand "Cost Attribution (post-cost)" after running a simulation with costs enabled.
