from __future__ import annotations
import pandas as pd
import streamlit as st
from .alpaca_data import load_ohlcv

@st.cache_data(show_spinner=False, ttl=3600)
def get_ohlcv_cached(symbol: str, start: str, end: str) -> pd.DataFrame:
    return load_ohlcv(symbol, start, end)