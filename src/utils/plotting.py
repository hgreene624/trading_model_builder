from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go

def equity_chart(equity: pd.Series, title: str = "Equity Curve"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Equity ($)", template="plotly_dark", height=420)
    return fig
