"""
dashboard/views/watchlist_alerts.py — Set price/indicator alerts for watchlist stocks.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_ALERTS_FILE = _ROOT / "data" / "watchlist_alerts.json"


def _load_alerts() -> list[dict]:
    if _ALERTS_FILE.exists():
        return json.loads(_ALERTS_FILE.read_text())
    return []


def _save_alerts(alerts: list[dict]) -> None:
    _ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ALERTS_FILE.write_text(json.dumps(alerts, indent=2))


def render_watchlist_alerts_page() -> None:
    st.header("🔔 Watchlist Alerts")
    st.caption("Set price and indicator alerts for any stock. "
               "Get notified via Activity Feed and Telegram when conditions are met.")

    from src.config import get_config
    cfg = get_config()

    alerts = _load_alerts()

    # ── Create new alert
    st.subheader("Create Alert")
    all_tickers = sorted(set(cfg.assets.all_symbols + cfg.assets.watchlist))

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.selectbox("Ticker", all_tickers + ["Custom..."], key="alert_ticker")
        if ticker == "Custom...":
            ticker = st.text_input("Enter ticker", key="custom_ticker").upper()
    with col2:
        condition = st.selectbox("Condition", [
            "Price drops below",
            "Price rises above",
            "RSI drops below",
            "RSI rises above",
            "Golden cross detected",
            "Death cross detected",
        ], help="What triggers the alert")
    with col3:
        if "detected" not in condition:
            threshold = st.number_input("Threshold", value=100.0, step=1.0,
                                        help="The value that triggers the alert")
        else:
            threshold = 0.0
            st.write("No threshold needed")

    if st.button("➕ Add Alert", type="primary"):
        if ticker:
            new_alert = {
                "ticker": ticker,
                "condition": condition,
                "threshold": threshold,
                "created": pd.Timestamp.now().isoformat(),
                "triggered": False,
            }
            alerts.append(new_alert)
            _save_alerts(alerts)
            st.success(f"Alert added: {ticker} — {condition} {threshold}")
            st.rerun()

    st.divider()

    # ── Active alerts
    st.subheader("Active Alerts")
    active = [a for a in alerts if not a.get("triggered", False)]

    if not active:
        st.info("No active alerts. Create one above to get notified when conditions are met.")
    else:
        for i, alert in enumerate(active):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                thresh_str = f" {alert['threshold']}" if alert['threshold'] else ""
                st.write(f"**{alert['ticker']}** — {alert['condition']}{thresh_str}")
                st.caption(f"Created: {alert['created'][:16]}")
            with col2:
                # Check current status
                try:
                    from src.ingestion.storage import ParquetStore
                    store = ParquetStore(cfg.data.storage_path)
                    df = store.load_ohlcv(alert["ticker"])
                    if not df.empty:
                        price = df["close"].iloc[-1]
                        st.metric("Current", f"${price:,.2f}")
                except Exception:
                    st.write("—")
            with col3:
                if st.button("🗑️ Remove", key=f"remove_alert_{i}"):
                    alerts.remove(alert)
                    _save_alerts(alerts)
                    st.rerun()

    # ── Triggered alerts
    triggered = [a for a in alerts if a.get("triggered", False)]
    if triggered:
        st.divider()
        st.subheader("Triggered Alerts")
        for alert in triggered:
            st.success(f"✅ {alert['ticker']} — {alert['condition']} {alert.get('threshold', '')} "
                       f"(triggered: {alert.get('triggered_at', '?')})")
