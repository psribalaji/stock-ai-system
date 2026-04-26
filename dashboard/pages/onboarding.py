"""
dashboard/pages/onboarding.py — How It Works / Onboarding page.
Plain-English explanation of the system for new users.
"""
from __future__ import annotations
import streamlit as st


def render_onboarding_page() -> None:
    st.header("🎓 How It Works")
    st.caption("A plain-English guide to how StockAI finds, evaluates, and trades stocks.")

    # ── The Big Picture
    st.subheader("The Big Picture")
    st.markdown("""
    StockAI is an automated stock trading system that:
    1. **Watches** 35+ stocks and discovers new trending ones
    2. **Analyzes** each stock using 30+ technical indicators
    3. **Decides** whether to buy or sell based on 5 different strategies
    4. **Checks** every decision against strict risk rules before trading
    5. **Manages** open positions with trailing stops and profit targets

    No trade happens without passing through all 5 steps.
    """)

    # ── Pipeline Flow
    st.subheader("How a Trade Happens")
    st.code("""
    📊 Price Data (OHLCV)
         │
         ▼
    🔬 Feature Engine — computes 30+ indicators
         │               (RSI, MACD, Bollinger Bands, moving averages...)
         ▼
    🧠 5 Strategies — each votes BUY / SELL / HOLD
         │  • Momentum: "Is the stock accelerating?"
         │  • Trend Following: "Is there a strong uptrend?"
         │  • Volatility Breakout: "Is it breaking out of a tight range?"
         │  • Mean Reversion: "Has it dropped too far and will bounce back?"
         │  • ML Ensemble: "What does the trained model predict?" (when enabled)
         ▼
    📏 Confidence Score — how likely is this trade to win?
         │  = historical win rate × market regime × volume × news sentiment
         │  Must be ≥ 60% to proceed
         ▼
    🛡️ Risk Manager — is this trade safe?
         │  ✓ Position size ≤ 5% of portfolio
         │  ✓ Not too correlated with existing positions
         │  ✓ Under 5 open positions
         │  ✓ No circuit breaker active
         ▼
    ✅ APPROVED → Order placed
         │
         ├── Trailing stop registered (follows price up, never down)
         └── Take-profit target set (auto-sell at 2:1 reward)
    """, language=None)

    # ── Key Concepts
    st.subheader("Key Concepts Explained")

    with st.expander("📈 What is RSI?"):
        st.markdown("""
        **Relative Strength Index** — measures how fast a stock is rising vs falling over 14 days.
        - **Below 35** = oversold (dropped too fast, might bounce back)
        - **Above 70** = overbought (rose too fast, might pull back)
        - **35–70** = neutral

        The system uses RSI to detect momentum shifts.
        """)

    with st.expander("📊 What is MACD?"):
        st.markdown("""
        **Moving Average Convergence Divergence** — shows when momentum is shifting.
        - **MACD crosses above signal line** = momentum turning bullish → potential BUY
        - **MACD crosses below signal line** = momentum turning bearish → potential SELL

        Think of it as an early warning system for trend changes.
        """)

    with st.expander("📉 What are Bollinger Bands?"):
        st.markdown("""
        **Bollinger Bands** — a channel around the stock's average price.
        - **Price at upper band** = stock is expensive relative to recent history
        - **Price at lower band** = stock is cheap relative to recent history
        - **Bands squeezing tight** = low volatility, breakout coming soon

        The volatility breakout strategy watches for squeezes.
        """)

    with st.expander("🔀 What is a Golden Cross / Death Cross?"):
        st.markdown("""
        - **Golden Cross** = 50-day average crosses ABOVE 200-day average → bullish signal
        - **Death Cross** = 50-day average crosses BELOW 200-day average → bearish signal

        These are slow but reliable trend signals. The trend-following strategy uses them.
        """)

    with st.expander("📏 What is ATR?"):
        st.markdown("""
        **Average True Range** — how much a stock typically moves per day.
        - NVDA with ATR of $4.50 means it moves about $4.50 up or down daily
        - JNJ with ATR of $2.00 is much calmer

        The trailing stop uses ATR to set a smart distance: volatile stocks get wider stops,
        calm stocks get tighter stops. This prevents getting stopped out on normal daily swings.
        """)

    with st.expander("🎯 What is Reward-to-Risk Ratio?"):
        st.markdown("""
        If you risk $9 per share (distance to stop loss), a 2:1 ratio means you target $18 profit.

        Why this matters: even with only 45% winning trades, 2:1 ratio is profitable:
        - 45 wins × $18 = $810
        - 55 losses × $9 = $495
        - **Net profit = $315 per 100 trades**
        """)

    with st.expander("🔗 What is Correlation?"):
        st.markdown("""
        How much two stocks move together. Scale: -1.0 to +1.0.
        - **0.85** (NVDA ↔ AMD) = move almost identically → risky to hold both
        - **0.15** (NVDA ↔ JNJ) = move independently → good diversification
        - **-0.30** = move in opposite directions → great hedge

        The system blocks new trades if they'd make your portfolio too correlated (>70%).
        """)

    # ── What to Check
    st.subheader("What to Check and When")
    st.markdown("""
    | When | What to check | Page |
    |---|---|---|
    | **Every morning** | Daily digest — what happened overnight? | Signals Today |
    | **During market hours** | Active trailing stops and take-profits | Stops & TPs |
    | **End of day** | Portfolio P&L, any circuit breakers triggered? | Portfolio, Risk |
    | **Weekly** | Strategy performance — which ones are working? | Strategy Performance |
    | **Weekly** | Discovery — any new trending stocks to approve? | Discovery |
    | **Monthly** | Audit log — review trade history and patterns | Audit Log |
    | **Quarterly** | ML ensemble — enough data to train yet? | ML Ensemble |
    """)

    # ── Safety
    st.subheader("Safety Layers")
    st.markdown("""
    Your money is protected by 5 independent safety layers:

    1. **Confidence threshold (60%)** — blocks low-quality signals
    2. **Position sizing (5% max)** — limits damage from any single trade
    3. **Correlation check (70% max)** — prevents concentrated bets
    4. **Daily circuit breaker (-2%)** — pauses trading after a bad day
    5. **Kill switch (-15%)** — closes everything after severe drawdown

    Each layer works independently. Even if one fails, the others protect you.
    """)
