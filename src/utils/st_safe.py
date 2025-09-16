# src/utils/st_safe.py
from __future__ import annotations

def _has_ctx() -> bool:
    try:
        import streamlit as st  # lazy
        return bool(getattr(st.runtime, "exists", lambda: False)())
    except Exception:
        return False

def info(msg: str) -> None:
    if _has_ctx():
        import streamlit as st
        st.info(msg)
    else:
        print(msg)

def warn(msg: str) -> None:
    if _has_ctx():
        import streamlit as st
        st.warning(msg)
    else:
        print(f"WARNING: {msg}")

def error(msg: str) -> None:
    if _has_ctx():
        import streamlit as st
        st.error(msg)
    else:
        print(f"ERROR: {msg}")