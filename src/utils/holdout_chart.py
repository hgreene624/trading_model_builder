from typing import Any, Callable, Dict, List, Tuple
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


# ---- Internal state keys ------------------------------------------------------

_SS_BASE = "hc"  # namespace prefix in st.session_state to avoid collisions

def _ss_key(name: str) -> str:
    return f"{_SS_BASE}:{name}"


# ---- Rendering ----------------------------------------------------------------

def _render(curves: List[Dict[str, Any]], placeholder: st.delta_generator.DeltaGenerator) -> None:
    """Render blank axes when no curves; otherwise grey old lines (no legend) and show legend for latest only."""
    fig = go.Figure()

    if not curves:
        fig.update_layout(
            title="Holdout equity (awaiting Gen 0)",
            margin=dict(l=10, r=10, t=35, b=10),
            showlegend=False,
            xaxis_title="Date",
            yaxis_title="Equity",
        )
        # Add empty trace so axes render
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False))
        _plot(fig, placeholder)
        return

    # Old curves (all except last) -> grey, no legend
    if len(curves) > 1:
        for run in curves[:-1]:
            fig.add_trace(
                go.Scatter(
                    x=run.get("x", []),
                    y=run.get("y", []),
                    mode="lines",
                    line=dict(width=1.0, color="rgba(150,150,150,0.55)"),
                    name="",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Current best = last item -> normal color, legend shown
    best = curves[-1]
    best_gen = best.get("gen", None)
    best_label = best.get("label") or (f"Gen {best_gen}" if best_gen is not None else "Current best")
    best_score = best.get("score", None)
    legend_name = best_label if best_score is None else f"{best_label} (score={best_score:.3f})"

    fig.add_trace(
        go.Scatter(
            x=best.get("x", []),
            y=best.get("y", []),
            mode="lines",
            line=dict(width=2.4),
            name=legend_name,
            showlegend=True,
        )
    )

    # Title reflects current best generation
    title = best_label if best_gen is not None else "Holdout equity (current best)"
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Equity",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )

    _plot(fig, placeholder)


def _plot(fig: "go.Figure", placeholder: st.delta_generator.DeltaGenerator) -> None:
    """Plot with a unique key per render to avoid duplicate element ID collisions."""
    seq_key = _ss_key("render_seq")
    seq = st.session_state.get(seq_key, 0) + 1
    st.session_state[seq_key] = seq
    render_key = f"holdout_equity_chart_{seq}"
    placeholder.plotly_chart(fig, use_container_width=True, key=render_key)


# ---- Holdout simulation -------------------------------------------------------

def _simulate_holdout_equity(
    params: Dict[str, Any],
    loader_fn: Callable[..., Dict[str, pd.DataFrame]],
    engine_fn: Callable[..., pd.DataFrame],
    symbols: List[str],
    holdout_start: Any,
    holdout_end: Any,
    starting_equity: float,
) -> Tuple[List[Any], List[float]]:
    """Runs an out-of-sample (holdout) backtest for the given params and returns (x, y)."""
    if not symbols:
        return [], []

    data = loader_fn(symbols=symbols, start=holdout_start, end=holdout_end)
    equity_df = engine_fn(params=params, data=data, starting_equity=starting_equity)

    if not isinstance(equity_df, pd.DataFrame) or "equity" not in equity_df.columns:
        return [], []

    x = equity_df.index.tolist()
    y = equity_df["equity"].astype(float).tolist()
    return x, y


# ---- Public API ---------------------------------------------------------------

def init_chart(
    placeholder: st.delta_generator.DeltaGenerator,
    starting_equity: float,
    holdout_start: Any,
    holdout_end: Any,
    loader_fn: Callable[..., Dict[str, pd.DataFrame]],
    engine_fn: Callable[..., pd.DataFrame],
    symbols: List[str],
    max_curves: int = 8,
) -> None:
    """Initialize/Reset chart and state; renders a blank chart immediately."""
    st.session_state[_ss_key("placeholder")] = placeholder
    st.session_state[_ss_key("render_seq")] = 0
    st.session_state[_ss_key("history")] = []
    st.session_state[_ss_key("best_score")] = None
    st.session_state[_ss_key("cfg")] = {
        "starting_equity": float(starting_equity),
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "loader_fn": loader_fn,
        "engine_fn": engine_fn,
        "symbols": list(symbols),
        "max_curves": int(max_curves),
    }
    _render([], placeholder)


def set_config(
    starting_equity: float | None = None,
    holdout_start: Any | None = None,
    holdout_end: Any | None = None,
    loader_fn: Callable[..., Dict[str, pd.DataFrame]] | None = None,
    engine_fn: Callable[..., pd.DataFrame] | None = None,
    symbols: List[str] | None = None,
    max_curves: int | None = None,
) -> None:
    """Update chart config during a run if UI changes occur (e.g., symbols or window)."""
    cfg = st.session_state.get(_ss_key("cfg"), {})
    if starting_equity is not None:
        cfg["starting_equity"] = float(starting_equity)
    if holdout_start is not None:
        cfg["holdout_start"] = holdout_start
    if holdout_end is not None:
        cfg["holdout_end"] = holdout_end
    if loader_fn is not None:
        cfg["loader_fn"] = loader_fn
    if engine_fn is not None:
        cfg["engine_fn"] = engine_fn
    if symbols is not None:
        cfg["symbols"] = list(symbols)
    if max_curves is not None:
        cfg["max_curves"] = int(max_curves)
    st.session_state[_ss_key("cfg")] = cfg


def on_generation_end(gen_idx: int, best_score: float, best_params: Dict[str, Any]) -> None:
    """EA callback: append and render a new curve ONLY when the score improves."""
    prev_best = st.session_state.get(_ss_key("best_score"), None)
    if (prev_best is not None) and not (best_score > prev_best):
        return  # no improvement; do nothing

    st.session_state[_ss_key("best_score")] = float(best_score)

    cfg = st.session_state.get(_ss_key("cfg"), {})
    placeholder = st.session_state.get(_ss_key("placeholder"))
    if not cfg or placeholder is None:
        return

    x, y = _simulate_holdout_equity(
        params=best_params,
        loader_fn=cfg.get("loader_fn"),
        engine_fn=cfg.get("engine_fn"),
        symbols=cfg.get("symbols", []),
        holdout_start=cfg.get("holdout_start"),
        holdout_end=cfg.get("holdout_end"),
        starting_equity=cfg.get("starting_equity", 100_000.0),
    )
    if not (x and y):
        return

    history = st.session_state.get(_ss_key("history"), [])
    history.append({"gen": gen_idx, "x": x, "y": y, "label": f"Gen {gen_idx}", "score": best_score})
    max_curves = int(cfg.get("max_curves", 8))
    if len(history) > max_curves:
        history = history[-max_curves:]
    st.session_state[_ss_key("history")] = history

    _render(history, placeholder)
