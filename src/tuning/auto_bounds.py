# src/tuning/auto_bounds.py
from __future__ import annotations

def write_suggested_bounds(bounds: dict[str, tuple[float, float]]) -> None:
    """
    Best-effort: write suggested bounds into st.session_state using your current widget keys.

    If called outside of a Streamlit session (e.g., worker process), this function
    safely no-ops instead of raising ScriptRunContext warnings.
    """
    try:
        from src.utils.st_safe import _has_ctx
        if not _has_ctx():
            # No active Streamlit context → silently skip
            return

        import streamlit as st
        ss = st.session_state

        for key, (low, high) in bounds.items():
            if key not in ss:
                continue
            # Update widget state safely
            val = ss[key]
            if isinstance(val, (int, float)):
                # Clamp existing values to new suggested bounds
                if val < low:
                    ss[key] = low
                elif val > high:
                    ss[key] = high
            # You could extend here to handle list/range inputs if needed

        st.info("Suggested parameter bounds have been applied.")
    except Exception as e:
        print(f"WARNING: auto_bounds.write_suggested_bounds failed: {e}")