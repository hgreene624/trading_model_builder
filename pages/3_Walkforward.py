# pages/3_Walkforward.py
from __future__ import annotations
import json
from datetime import date, timedelta
from typing import List, Optional
import streamlit as st

# Storage + WF
from src.storage import list_portfolios, load_portfolio, load_strategy_params
# We import inside a wrapper to allow signature-flexible calling

st.set_page_config(page_title="Walk-Forward Validation", layout="wide")


def _load_portfolio_symbols(port_name: str) -> List[str]:
    obj = load_portfolio(port_name)
    if isinstance(obj, dict):
        raw = obj.get("tickers") or obj.get("symbols") or obj.get("items") or obj.get("data") or []
    else:
        raw = obj
    # Normalize
    return [s.strip().upper().replace("/", "-") for s in raw if isinstance(s, str) and s.strip()]


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _call_walk_forward_adaptive(*, tickers: List[str], strategy: str, params: dict,
                                splits: int, train_m: int, test_m: int,
                                start: Optional[str], end: Optional[str],
                                use_ea_inside_each_split: bool):
    from src.optimization.walkforward import walk_forward
    import inspect

    # Candidate kwargs mapped to common names; we’ll filter by the real signature
    candidates = {
        "symbols": tickers,
        "tickers": tickers,
        "portfolio": {"name": st.session_state.get("wf_port_name", ""), "tickers": tickers},
        "strategy": strategy,
        "strategy_name": strategy,
        "params": params,
        "base_params": params,
        "ea_params": params,
        "ea_kwargs": {"base_params": params},
        "splits": splits,
        "n_splits": splits,
        "wf_splits": splits,
        "train_months": train_m,
        "wf_train_months": train_m,
        "test_months": test_m,
        "wf_test_months": test_m,
        "start": start,
        "window_start": start,
        "end": end,
        "window_end": end,
        "use_ea_inside_each_split": use_ea_inside_each_split,
        "reoptimize_each_split": use_ea_inside_each_split,
    }

    sig = inspect.signature(walk_forward)
    call_kwargs = {k: v for k, v in candidates.items() if k in sig.parameters and v is not None}
    return walk_forward(**call_kwargs)


def main():
    st.title("Walk-Forward Validation")

    # --- Portfolio selection ---
    try:
        portfolios = sorted(list_portfolios())
    except Exception as e:
        st.error(f"Could not list portfolios: {e}")
        st.stop()

    if not portfolios:
        st.error("No saved portfolios found. Create one on the Portfolios page first.")
        st.stop()

    default_idx = portfolios.index("Default") if "Default" in portfolios else 0
    port_name = st.selectbox("Portfolio", options=portfolios, index=default_idx)
    st.session_state["wf_port_name"] = port_name

    try:
        tickers = _load_portfolio_symbols(port_name)
    except Exception as e:
        st.error(f"Failed to load portfolio '{port_name}': {e}")
        st.stop()

    if not tickers:
        st.warning("Selected portfolio has no symbols.")
        st.stop()

    st.info(f"Loaded **{len(tickers)}** symbols.")

    # --- Strategy + EA params (loaded from Step 2 save) ---
    # For now we assume ATR breakout; adjust if you support multiple strategies in UI.
    strategy_dotted = "src.models.atr_breakout"

    saved = load_strategy_params(portfolio=port_name, strategy=strategy_dotted, scope="ea")
    saved_params = (saved or {}).get("params") or {}

    with st.expander("EA parameters (loaded)", expanded=bool(saved_params)):
        if saved_params:
            st.json(saved_params)
        else:
            st.info("No saved EA params for this portfolio/strategy yet.")

    override_json = st.text_area(
        "Optional override (JSON dict). Leave blank to use loaded EA params.",
        value="",
        height=140,
        help='Example: {"atr_len": 14, "risk_mult": 2.5, "breakout_lookback": 20}',
    )
    override = _safe_json_loads(override_json) if override_json.strip() else None
    run_params = override if override is not None else saved_params

    if not run_params:
        st.warning("No parameters to run. Save EA best params on the Strategy Adapter page or paste overrides here.")
        st.stop()

    # --- WF settings ---
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        splits = st.number_input("Splits", min_value=2, max_value=24, value=3, step=1)
    with col_b:
        train_m = st.number_input("Train months", min_value=1, max_value=60, value=9, step=1)
    with col_c:
        test_m = st.number_input("Test months", min_value=1, max_value=24, value=3, step=1)
    with col_d:
        reopt = st.checkbox("Use EA inside each split", value=True)

    # Date bounds (optional)
    today = date.today()
    default_start = today - timedelta(days=365 * 3)
    col_s, col_e = st.columns(2)
    with col_s:
        start_d = st.date_input("Start (optional)", value=default_start)
    with col_e:
        end_d = st.date_input("End (optional)", value=today)

    start_s = start_d.isoformat() if start_d else None
    end_s = end_d.isoformat() if end_d else None

    # --- Run WF ---
    if st.button("Run Walk-Forward", type="primary", use_container_width=True):
        with st.spinner("Running walk-forward..."):
            try:
                result = _call_walk_forward_adaptive(
                    tickers=tickers,
                    strategy=strategy_dotted,
                    params=run_params,
                    splits=int(splits),
                    train_m=int(train_m),
                    test_m=int(test_m),
                    start=start_s,
                    end=end_s,
                    use_ea_inside_each_split=bool(reopt),
                )
            except Exception as e:
                st.exception(e)
                st.stop()

        st.success("Walk-forward completed.")
        st.subheader("Result")
        st.json(result if isinstance(result, dict) else {"result": str(result)})


if __name__ == "__main__":
    main()