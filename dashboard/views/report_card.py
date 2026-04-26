"""
dashboard/views/report_card.py — Paper trading performance report card.
Summary of how the system performed over a period.
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


def render_report_card_page() -> None:
    st.header("📝 Report Card")
    st.caption("How is the system performing? A summary of paper trading results.")

    from src.ingestion.storage import ParquetStore
    from src.config import get_config

    cfg = get_config()
    store = ParquetStore(cfg.data.storage_path)

    period = st.selectbox("Period", ["Last 7 days", "Last 30 days", "Last 90 days", "All time"])
    days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": 3650}
    days = days_map[period]

    from datetime import date, timedelta
    end = date.today()
    start = None if days >= 3650 else end - timedelta(days=days)
    audit = store.load_audit(start=start, end=end)
    signals = store.load_signals()

    if audit.empty and signals.empty:
        st.info("No trading data yet. Run the pipeline to start generating your report card.")
        st.markdown("""
        **What you'll see here after paper trading:**
        - 🏆 Overall grade (A through F)
        - 📊 Key metrics: return, Sharpe, win rate, drawdown
        - 🥇 Best and worst trades
        - 📈 Strategy rankings
        - 📉 Comparison vs holding QQQ
        """)
        return

    submitted = audit[audit["status"] == "submitted"] if "status" in audit.columns else audit

    # ── Overall Grade
    st.subheader("Overall Grade")

    total_trades = len(submitted)
    has_pnl = "pnl_pct" in submitted.columns and not submitted["pnl_pct"].isna().all()

    if has_pnl and total_trades >= 10:
        win_rate = (submitted["pnl_pct"] > 0).sum() / total_trades
        avg_return = submitted["pnl_pct"].mean()

        # Simple grading
        if win_rate >= 0.55 and avg_return > 0.01:
            grade, color = "A", "🟢"
        elif win_rate >= 0.45 and avg_return > 0:
            grade, color = "B", "🟢"
        elif win_rate >= 0.40:
            grade, color = "C", "🟡"
        elif win_rate >= 0.35:
            grade, color = "D", "🟠"
        else:
            grade, color = "F", "🔴"

        st.markdown(f"## {color} Grade: **{grade}**")
        st.write(f"Based on {total_trades} trades over {period.lower()}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Win Rate", f"{win_rate:.1%}")
        m2.metric("Avg Return/Trade", f"{avg_return:+.2%}")
        m3.metric("Total Trades", total_trades)
        m4.metric("Best Trade", f"{submitted['pnl_pct'].max():+.2%}" if has_pnl else "—")
    else:
        st.info(f"Need at least 10 trades with P&L data for grading. "
                f"Currently: {total_trades} trades.")

    st.divider()

    # ── Strategy Rankings
    if "strategy" in submitted.columns and total_trades > 0:
        st.subheader("🏆 Strategy Rankings")
        strat_stats = []
        for strat in submitted["strategy"].unique():
            subset = submitted[submitted["strategy"] == strat]
            row = {"Strategy": strat, "Trades": len(subset)}
            if has_pnl:
                row["Win Rate"] = f"{(subset['pnl_pct'] > 0).sum() / len(subset):.1%}" if len(subset) > 0 else "—"
                row["Avg Return"] = f"{subset['pnl_pct'].mean():+.2%}"
            strat_stats.append(row)

        st.dataframe(pd.DataFrame(strat_stats), use_container_width=True, hide_index=True)

    # ── Signal efficiency
    if not signals.empty and "approved" in signals.columns:
        st.divider()
        st.subheader("📊 Signal Efficiency")
        total_signals = len(signals)
        approved = signals["approved"].sum() if "approved" in signals.columns else 0
        blocked = total_signals - approved

        e1, e2, e3 = st.columns(3)
        e1.metric("Total Signals", total_signals)
        e2.metric("Approved", f"{approved} ({approved/max(total_signals,1):.0%})")
        e3.metric("Blocked", f"{blocked} ({blocked/max(total_signals,1):.0%})")

        if "block_reason" in signals.columns:
            blocked_signals = signals[signals["approved"] == False]
            if not blocked_signals.empty:
                st.write("**Top block reasons:**")
                reasons = blocked_signals["block_reason"].value_counts().head(5)
                for reason, count in reasons.items():
                    st.write(f"  • {reason}: {count} times")

    # ── Recommendations
    st.divider()
    st.subheader("💡 Recommendations")
    recs = []
    if total_trades < 30:
        recs.append("📈 **Keep running** — need 30+ trades for statistically meaningful results")
    if has_pnl:
        win_rate = (submitted["pnl_pct"] > 0).sum() / max(total_trades, 1)
        if win_rate < 0.40:
            recs.append("⚠️ **Win rate below 40%** — consider raising min_confidence threshold")
        if win_rate > 0.60:
            recs.append("🎯 **Win rate above 60%** — system is performing well, consider lowering min_confidence slightly to capture more trades")
    if not signals.empty and "approved" in signals.columns:
        approval_rate = signals["approved"].sum() / max(len(signals), 1)
        if approval_rate < 0.10:
            recs.append("🔒 **Very few signals approved (<10%)** — thresholds may be too strict")
        if approval_rate > 0.50:
            recs.append("⚠️ **Many signals approved (>50%)** — thresholds may be too loose")

    if recs:
        for r in recs:
            st.markdown(r)
    else:
        st.success("✅ No issues detected — keep paper trading to build more data!")
