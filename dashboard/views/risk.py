"""
dashboard/pages/risk.py — Risk Dashboard page.
Shows portfolio risk limits with guided tooltips, correlation heatmap, and circuit breaker status.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_risk_page() -> None:
    st.header("Risk Dashboard")
    st.caption("Monitor all risk limits, circuit breakers, and portfolio correlation. "
               "Hover ℹ️ icons for explanations.")

    from src.config import get_config
    from src.ingestion.storage import ParquetStore
    from src.risk.risk_manager import RiskManager

    cfg = get_config()
    rm = RiskManager()
    store = ParquetStore(cfg.data.storage_path)

    # ── Circuit breaker / kill switch status
    st.subheader("System Status")
    c1, c2 = st.columns(2)
    if rm.is_killed:
        c1.error("🔴 KILL SWITCH ACTIVE — all trading halted")
    elif rm.is_paused:
        c1.warning("⚠️ CIRCUIT BREAKER — trading paused for the day")
    else:
        c1.success("✅ All systems normal")
    c2.metric("Trading Mode", cfg.trading.mode.upper(),
              help="'PAPER' = simulated trades, no real money. 'LIVE' = real money at risk.")

    # ── Risk limits with tooltips
    st.subheader("Risk Limits")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Position Size", f"{cfg.risk.max_position_pct:.0%}",
                help="Maximum % of portfolio in a single trade. "
                     "5% means each trade risks $5,000 on a $100K portfolio. "
                     "Lower = safer but smaller gains. Recommended: 3–7%.")
    col2.metric("Max Open Positions", cfg.risk.max_open_positions,
                help="Maximum simultaneous trades. "
                     "5 positions × 5% each = 25% of portfolio at risk. "
                     "Lower = less exposure. Recommended: 3–8.")
    col3.metric("Daily Loss Limit", f"{cfg.risk.daily_loss_limit:.0%}",
                help="Circuit breaker: if portfolio drops this much in one day, "
                     "ALL trading pauses until tomorrow. Prevents panic spirals. "
                     "Recommended: 1.5–3%.")
    col4.metric("Max Drawdown", f"{cfg.risk.max_drawdown_pct:.0%}",
                help="Kill switch: if portfolio drops this much from its peak, "
                     "ALL positions are closed and trading halts until manual reset. "
                     "This is your last line of defense. Recommended: 10–20%.")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Max Crypto", f"{cfg.risk.max_crypto_pct:.0%}",
                help="Maximum total portfolio exposure to crypto (BTC, SOL). "
                     "Crypto is more volatile than stocks — this caps your risk. "
                     "Recommended: 5–15%.")
    col6.metric("Max Correlation", f"{cfg.risk.max_portfolio_corr:.0%}",
                help="Blocks new trades if average correlation with held positions exceeds this. "
                     "Prevents holding 5 stocks that all move together (e.g. all semiconductors). "
                     "Lower = more diversified. Recommended: 0.60–0.80.")
    col7.metric("Trailing Stop", f"{cfg.risk.trailing_stop_atr_mult:.1f}× ATR",
                help="Trailing stop distance = this multiplier × ATR (Average True Range). "
                     "For NVDA with ATR=$4.50: stop sits $9 below the highest price since entry. "
                     "Lower = tighter stops (protect more, exit sooner). "
                     "Higher = wider stops (ride trends longer). Recommended: 1.5–3.0.")
    col8.metric("Take-Profit", f"{cfg.risk.reward_risk_ratio:.1f}:1",
                help="Sell when profit = this × your risk. "
                     "2:1 means if you risk $9 (stop loss), you take profit at $18 gain. "
                     "Higher = bigger wins but fewer of them. Recommended: 1.5–3.0.")

    st.divider()

    # ── Signal thresholds
    st.subheader("Signal Thresholds")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min Confidence", f"{cfg.risk.min_confidence:.0%}",
                help="Signals below this confidence score are blocked. "
                     "Confidence = base_win_rate × regime × volume × sentiment. "
                     "Lower = more trades but lower quality. Recommended: 0.55–0.70.")
    col2.metric("Signal Interval", f"{cfg.schedule.signal_interval_min} min",
                help="How often the system scans for new trade signals. "
                     "30 min is optimal for daily indicators. "
                     "Shorter = more compute + API costs, same signals.")
    col3.metric("Position Check", f"{cfg.schedule.position_check_interval_min} min",
                help="How often trailing stops and take-profits are checked. "
                     "5 min ensures you don't miss an exit. "
                     "This is cheap — just price lookups for held positions.")
    col4.metric("Simulation Mode", "ON" if cfg.schedule.force_market_hours else "OFF",
                help="When ON, the system runs 24/7 regardless of market hours. "
                     "Use for paper trading testing only — signals outside market hours "
                     "use stale prices.")

    st.divider()

    # ── Correlation heatmap
    st.subheader("Portfolio Correlation Heatmap")
    st.caption("Shows how much your stocks move together. Red = high correlation (risky). "
               "If you hold 5 highly correlated stocks, one bad day hits them all.")

    tickers = cfg.assets.stocks[:15]
    price_data = {}
    for t in tickers:
        df = store.load_ohlcv(t)
        if not df.empty and "close" in df.columns:
            price_data[t] = df["close"].tail(60)

    if len(price_data) >= 2:
        returns_df = pd.DataFrame(price_data).pct_change().dropna()
        corr = returns_df.corr()

        st.dataframe(corr.round(2), use_container_width=True)

        high_corr_pairs = []
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                val = corr.iloc[i, j]
                if val >= cfg.risk.max_portfolio_corr:
                    high_corr_pairs.append((corr.index[i], corr.columns[j], val))

        if high_corr_pairs:
            st.warning(f"⚠️ {len(high_corr_pairs)} pair(s) above {cfg.risk.max_portfolio_corr:.0%} limit:")
            for t1, t2, val in high_corr_pairs:
                st.write(f"  • {t1} ↔ {t2}: **{val:.2f}**")
        else:
            st.success("All correlations within limits")
    else:
        st.info("Need OHLCV data for at least 2 tickers to show correlation heatmap")
