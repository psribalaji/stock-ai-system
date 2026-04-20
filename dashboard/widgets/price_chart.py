"""
dashboard/widgets/price_chart.py — Reusable interactive price chart widget.

Usage from any Streamlit page:
    from dashboard.widgets.price_chart import render_price_chart
    render_price_chart("MRVL")
    render_price_chart("NVDA", period="3mo", chart_type="candlestick")
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_PERIODS = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV: try local Parquet first, then Alpaca, then yfinance."""
    # Try local
    try:
        from src.ingestion.storage import ParquetStore
        from src.config import get_config
        store = ParquetStore(get_config().data.storage_path)
        df = store.load_ohlcv(ticker)
        if not df.empty and len(df) > 50:
            return df
    except Exception:
        pass

    # Try Alpaca
    try:
        from src.ingestion.alpaca_client import AlpacaClient
        days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 90)
        df = AlpacaClient().get_recent_bars(ticker, days=days)
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: yfinance (free, no key)
    try:
        import yfinance as yf
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        data.columns = [c.lower() for c in data.columns]
        data = data.rename(columns={"date": "timestamp"})
        return data
    except Exception:
        return pd.DataFrame()


def render_price_chart(
    ticker: str,
    period: str = "3mo",
    chart_type: str = "candlestick",
    show_controls: bool = True,
    show_volume: bool = True,
    show_ma: bool = True,
    height: int = 450,
    key_prefix: str = "",
) -> None:
    """
    Render an interactive price chart for a ticker.

    Args:
        ticker:        Stock ticker symbol
        period:        Default period ("1mo", "3mo", "6mo", "1y", "2y")
        chart_type:    "candlestick" or "line"
        show_controls: Show period/type selectors
        show_volume:   Show volume subplot
        show_ma:       Overlay SMA50/SMA200
        height:        Chart height in pixels
        key_prefix:    Unique key prefix (for multiple charts on same page)
    """
    # Controls
    if show_controls:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{ticker}**")
        with col2:
            selected_period = st.selectbox(
                "Period", list(_PERIODS.keys()),
                index=list(_PERIODS.keys()).index("3M"),
                key=f"{key_prefix}period_{ticker}",
                label_visibility="collapsed",
            )
            period = _PERIODS[selected_period]
        with col3:
            chart_type = st.selectbox(
                "Type", ["candlestick", "line"],
                key=f"{key_prefix}type_{ticker}",
                label_visibility="collapsed",
            )

    # Fetch data
    df = _fetch_ohlcv(ticker, period)
    if df.empty:
        st.warning(f"No price data available for {ticker}")
        return

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        x = df["timestamp"]
    else:
        x = df.index

    # Current price + change
    current = df["close"].iloc[-1]
    prev = df["close"].iloc[-2] if len(df) > 1 else current
    change_pct = (current - prev) / prev
    period_start = df["close"].iloc[0]
    period_change = (current - period_start) / period_start

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${current:.2f}", f"{change_pct:+.2%} today")
    c2.metric("Period Change", f"{period_change:+.2%}")
    c3.metric("Bars", len(df))

    # Build chart
    row_heights = [0.7, 0.3] if show_volume else [1.0]
    rows = 2 if show_volume else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03,
    )

    # Price
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=x, y=df["close"], mode="lines", name="Close",
            line=dict(color="#2196F3", width=2),
        ), row=1, col=1)

    # Moving averages
    if show_ma and len(df) >= 50:
        sma50 = df["close"].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=x, y=sma50, mode="lines", name="SMA50",
            line=dict(color="#FF9800", width=1, dash="dot"),
        ), row=1, col=1)

        if len(df) >= 200:
            sma200 = df["close"].rolling(200).mean()
            fig.add_trace(go.Scatter(
                x=x, y=sma200, mode="lines", name="SMA200",
                line=dict(color="#9C27B0", width=1, dash="dot"),
            ), row=1, col=1)

    # Volume
    if show_volume and "volume" in df.columns:
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=x, y=df["volume"], name="Volume",
            marker_color=colors, opacity=0.5,
        ), row=2, col=1)

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Vol", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}chart_{ticker}")
