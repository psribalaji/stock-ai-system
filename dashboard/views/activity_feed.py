"""
dashboard/views/activity_feed.py — Real-time activity feed page.
Shows chronological notifications: trades, stops, discoveries, alerts.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_activity_feed_page() -> None:
    st.header("🔔 Activity Feed")
    st.caption("Real-time log of everything the system does — trades, stops, discoveries, alerts.")

    from src.notifications import Notifier

    notifier = Notifier()
    feed = notifier.get_feed(limit=100)

    if feed.empty:
        st.info("No activity yet. Events will appear here when the pipeline runs — "
                "trades placed, stops triggered, stocks discovered, etc.")
        st.markdown("""
        **Events you'll see here:**
        - 🟢 Trade placed (BUY)
        - 🔴 Trade placed (SELL)
        - 🎯 Take-profit hit
        - 🛑 Trailing stop triggered
        - 🔔 New stock discovered
        - ⏳ Stock awaiting approval
        - ⚠️ Circuit breaker / drift warning
        - ☠️ Kill switch activated
        - 📊 Daily summary
        """)
        return

    # ── Filters
    col1, col2 = st.columns(2)
    with col1:
        levels = ["All"] + sorted(feed["level"].unique().tolist())
        level_filter = st.selectbox("Filter by type", levels)
    with col2:
        tickers = ["All"] + sorted([t for t in feed["ticker"].unique() if t])
        ticker_filter = st.selectbox("Filter by ticker", tickers)

    filtered = feed.copy()
    if level_filter != "All":
        filtered = filtered[filtered["level"] == level_filter]
    if ticker_filter != "All":
        filtered = filtered[filtered["ticker"] == ticker_filter]

    st.caption(f"{len(filtered)} events")

    # ── Feed display
    for _, row in filtered.iterrows():
        ts = pd.to_datetime(row["timestamp"]).strftime("%b %d %H:%M")
        icon = row.get("icon", "📌")
        msg = row.get("message", "")
        ticker = row.get("ticker", "")
        level = row.get("level", "info")

        # Color based on level
        if level in ("critical", "warning"):
            st.warning(f"**{ts}** — {icon} {msg}")
        elif level in ("trade", "tp"):
            st.success(f"**{ts}** — {icon} {msg}")
        elif level in ("sell", "stop"):
            st.error(f"**{ts}** — {icon} {msg}")
        else:
            st.info(f"**{ts}** — {icon} {msg}")
