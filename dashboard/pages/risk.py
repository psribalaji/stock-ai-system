"""
dashboard/pages/risk.py — Risk Dashboard page.
Shows portfolio risk limits, correlation heatmap, and circuit breaker status.
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
    c2.metric("Trading Mode", cfg.trading.mode.upper())

    # ── Risk limits gauge
    st.subheader("Risk Limits")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Position Size", f"{cfg.risk.max_position_pct:.0%}")
    col2.metric("Max Open Positions", cfg.risk.max_open_positions)
    col3.metric("Daily Loss Limit", f"{cfg.risk.daily_loss_limit:.0%}")
    col4.metric("Max Drawdown", f"{cfg.risk.max_drawdown_pct:.0%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Max Crypto", f"{cfg.risk.max_crypto_pct:.0%}")
    col6.metric("Max Correlation", f"{cfg.risk.max_portfolio_corr:.0%}")
    col7.metric("Trailing Stop ATR Mult", f"{cfg.risk.trailing_stop_atr_mult:.1f}x")
    col8.metric("Reward:Risk Ratio", f"{cfg.risk.reward_risk_ratio:.1f}:1")

    # ── Correlation heatmap
    st.subheader("Portfolio Correlation Heatmap")
    tickers = cfg.assets.stocks[:15]  # top 15 for readability
    price_data = {}
    for t in tickers:
        df = store.load_ohlcv(t)
        if not df.empty and "close" in df.columns:
            price_data[t] = df["close"].tail(60)

    if len(price_data) >= 2:
        returns_df = pd.DataFrame(price_data).pct_change().dropna()
        corr = returns_df.corr()

        # Color-code: red = high correlation, green = low
        st.dataframe(
            corr.style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=True,
        )

        # Flag high correlations
        high_corr_pairs = []
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                val = corr.iloc[i, j]
                if val >= cfg.risk.max_portfolio_corr:
                    high_corr_pairs.append((corr.index[i], corr.columns[j], val))

        if high_corr_pairs:
            st.warning(f"⚠️ {len(high_corr_pairs)} pair(s) above {cfg.risk.max_portfolio_corr:.0%} correlation limit:")
            for t1, t2, val in high_corr_pairs:
                st.write(f"  • {t1} ↔ {t2}: **{val:.2f}**")
        else:
            st.success("All correlations within limits")
    else:
        st.info("Need OHLCV data for at least 2 tickers to show correlation heatmap")
