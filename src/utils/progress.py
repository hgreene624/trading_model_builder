# src/utils/progress.py
from __future__ import annotations
from typing import Callable, Dict, Any

# Public type alias: a function taking (event, payload)
ProgressCallback = Callable[[str, Dict[str, Any]], None]


def console_progress(event: str, payload: Dict[str, Any]) -> None:
    """Lightweight progress sink for non-UI contexts (safe in subprocesses)."""
    try:
        key_bits = {
            k: payload.get(k)
            for k in ("gen", "idx", "score", "best_score", "avg_score", "avg_trades")
            if k in payload
        }
        print(f"[{event}] {key_bits}")
    except Exception:
        # Never let progress crash the caller
        pass


def streamlit_progress(status_box=None, indiv_box=None, gen_box=None) -> ProgressCallback:
    """Return a callback that updates Streamlit widgets *only when a session exists*.

    Performs a LAZY import of `streamlit` inside the callback so importing this
    module in worker processes won't emit the ScriptRunContext warning.
    """
    def cb(event: str, payload: Dict[str, Any]) -> None:
        try:
            import streamlit as st  # lazy import in UI process
            if not getattr(st.runtime, "exists", lambda: False)():
                return  # no Streamlit session (e.g., in workers)
        except Exception:
            return

        try:
            if event == "generation_start":
                gen = payload.get("gen")
                pop = payload.get("pop_size")
                if status_box is not None:
                    status_box.info(f"Generation {gen} starting … (population={pop})")

            elif event == "individual_evaluated":
                if indiv_box is not None:
                    m = payload.get("metrics", {}) or {}
                    row = {
                        "gen": payload.get("gen"),
                        "idx": payload.get("idx"),
                        "score": payload.get("score"),
                        "trades": int(m.get("trades", 0) or 0),
                        "avg_hold_days": float(m.get("avg_holding_days", 0.0) or 0.0),
                        "cagr": float(m.get("cagr", 0.0) or 0.0),
                        "calmar": float(m.get("calmar", 0.0) or 0.0),
                        "sharpe": float(m.get("sharpe", 0.0) or 0.0),
                    }
                    import pandas as pd  # defer import
                    if not hasattr(cb, "_rows"):
                        cb._rows = []  # type: ignore[attr-defined]
                    cb._rows.append(row)  # type: ignore[attr-defined]
                    cb._rows = cb._rows[-100:]  # keep last 100
                    indiv_box.dataframe(pd.DataFrame(cb._rows), width="stretch", height=260)

            elif event == "generation_end":
                if gen_box is not None:
                    import pandas as pd
                    df = pd.DataFrame([{
                        "generation": payload.get("gen"),
                        "best_score": payload.get("best_score"),
                        "avg_score": payload.get("avg_score"),
                        "avg_trades": payload.get("avg_trades"),
                        "no_trades_%": 100.0 * (payload.get("pct_no_trades", 0.0) or 0.0),
                        "elite_n": payload.get("elite_n"),
                        "breed_n": payload.get("breed_n"),
                        "inject_n": payload.get("inject_n"),
                    }])
                    gen_box.dataframe(df, width="stretch", height=88)

            elif event == "done":
                if status_box is not None:
                    secs = payload.get("elapsed_sec", 0.0)
                    status_box.success(f"EA completed in {secs:.1f}s")
        except Exception:
            # Never let UI updates crash the training loop
            pass

    return cb