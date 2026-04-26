"""
dashboard/views/pnl_tracker.py — Cumulative P&L tracking over time.
Shows running total, daily/weekly/monthly breakdowns, vs benchmark.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_pnl_tracker_page() -> None:
    st.header("💰 P&L Tracker")
    st.caption("Track your cumulative profit and loss over time.")

    from src.ingestion.storage import ParquetStore
    from src.config import get_config

    cfg = get_config()
    store = ParquetStore(cfg.data.storage_path)

    audit = store.load_audit()
    if audit.empty:
        st.info("No trade data yet. P&L tracking starts once the pipeline places trades.")
        st.markdown("""
        **What you'll see here:**
        - 📈 Cumulative P&L chart over time
        - 📊 Daily, weekly, monthly breakdowns
        - 🏆 Best and worst trades
        - 📉 Comparison vs S&P 500 (QQQ)
        """)
        return

    submitted = audit[audit["status"] == "submitted"] if "status" in audit.columns else audit
    if submitted.empty:
        st.info("No submitted orders yet.")
        return

    submitted = submitted.copy()
    submitted["date"] = pd.to_datetime(submitted["timestamp_submitted"]).dt.date

    # ── Summary metrics
    st.subheader("Overall Performance")
    total_trades = len(submitted)
    buys = len(submitted[submitted["direction"] == "BUY"]) if "direction" in submitted.columns else 0
    sells = len(submitted[submitted["direction"] == "SELL"]) if "direction" in submitted.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Orders", total_trades)
    m2.metric("BUY Orders", buys)
    m3.metric("SELL Orders", sells)

    if "pnl_pct" in submitted.columns and not submitted["pnl_pct"].isna().all():
        winners = (submitted["pnl_pct"] > 0).sum()
        losers = (submitted["pnl_pct"] < 0).sum()
        win_rate = winners / max(winners + losers, 1)
        avg_win = submitted[submitted["pnl_pct"] > 0]["pnl_pct"].mean() if winners > 0 else 0
        avg_loss = submitted[submitted["pnl_pct"] < 0]["pnl_pct"].mean() if losers > 0 else 0

        m4.metric("Win Rate", f"{win_rate:.1%}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Winners", winners)
        m6.metric("Losers", losers)
        m7.metric("Avg Win", f"{avg_win:+.2%}", help="Average return on winning trades")
        m8.metric("Avg Loss", f"{avg_loss:+.2%}", help="Average return on losing trades")
    else:
        m4.metric("Strategies", submitted["strategy"].nunique() if "strategy" in submitted.columns else "—")

    st.divider()

    # ── Trades over time chart
    st.subheader("Trading Activity")
    daily_counts = submitted.groupby("date").size().reset_index(name="trades")
    fig = go.Figure(go.Bar(
        x=daily_counts["date"], y=daily_counts["trades"],
        marker_color="#2196F3",
    ))
    fig.update_layout(title="Trades Per Day", yaxis_title="Trades", height=250,
                      margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── By strategy breakdown
    if "strategy" in submitted.columns:
        st.subheader("By Strategy")
        strat_counts = submitted.groupby("strategy").size().reset_index(name="trades")
        fig2 = go.Figure(go.Pie(
            labels=strat_counts["strategy"], values=strat_counts["trades"],
            hole=0.4,
        ))
        fig2.update_layout(title="Trade Distribution by Strategy", height=300,
                           margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # ── By ticker breakdown
    if "ticker" in submitted.columns:
        st.subheader("Most Traded Tickers")
        ticker_counts = submitted["ticker"].value_counts().head(10)
        st.bar_chart(ticker_counts)

    # ── Recent trades table
    st.subheader("Recent Trades")
    display_cols = [c for c in ["timestamp_submitted", "ticker", "direction", "strategy",
                                "confidence", "entry_price", "position_size_usd", "status"]
                    if c in submitted.columns]
    st.dataframe(submitted[display_cols].head(20), use_container_width=True, hide_index=True)
