"""
dashboard/pages/config_editor.py — Interactive config editor with impact preview.
Preview-only: shows what would change, requires explicit save to apply.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import yaml

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CONFIG_PATH = _ROOT / "config.yaml"


def render_config_editor_page() -> None:
    st.header("⚙️ Config Editor")
    st.caption("Adjust trading parameters with live impact preview. "
               "Changes are preview-only until you click Save.")

    from src.config import get_config
    cfg = get_config()

    # Track changes
    changes = {}

    # ── Risk Parameters ───────────────────────────────────────────
    st.subheader("Risk Parameters")

    col1, col2 = st.columns(2)
    with col1:
        new_min_conf = st.slider(
            "Min Confidence",
            min_value=0.40, max_value=0.85, value=cfg.risk.min_confidence, step=0.05,
            help="Signals below this are blocked. Lower = more trades, higher risk.",
        )
        if new_min_conf != cfg.risk.min_confidence:
            direction = "more" if new_min_conf < cfg.risk.min_confidence else "fewer"
            st.info(f"📊 **Impact:** {direction} signals will pass. "
                    f"{'Riskier trades may slip through.' if direction == 'more' else 'Only high-conviction trades.'}")
            changes["risk.min_confidence"] = new_min_conf

        new_max_pos_pct = st.slider(
            "Max Position Size (%)",
            min_value=1, max_value=15, value=int(cfg.risk.max_position_pct * 100), step=1,
            help="Max % of portfolio per trade.",
        )
        if new_max_pos_pct != int(cfg.risk.max_position_pct * 100):
            dollar = new_max_pos_pct * 1000  # assuming $100K portfolio
            st.info(f"💰 **Impact:** Each trade risks up to ${dollar:,} on a $100K portfolio.")
            changes["risk.max_position_pct"] = new_max_pos_pct / 100

        new_max_positions = st.slider(
            "Max Open Positions",
            min_value=1, max_value=15, value=cfg.risk.max_open_positions, step=1,
            help="Maximum simultaneous trades.",
        )
        if new_max_positions != cfg.risk.max_open_positions:
            total_exposure = new_max_positions * (new_max_pos_pct if "risk.max_position_pct" in changes else cfg.risk.max_position_pct * 100)
            st.info(f"📊 **Impact:** Max portfolio exposure = {total_exposure:.0f}% "
                    f"({new_max_positions} × {new_max_pos_pct}%)")
            changes["risk.max_open_positions"] = new_max_positions

    with col2:
        new_stop_loss = st.slider(
            "Hard Stop Loss (%)",
            min_value=3, max_value=15, value=int(cfg.risk.stop_loss_pct * 100), step=1,
            help="Maximum loss per trade before forced exit.",
        )
        if new_stop_loss != int(cfg.risk.stop_loss_pct * 100):
            st.info(f"🛑 **Impact:** Buy at $100 → hard stop at ${100 - new_stop_loss:.0f}. "
                    f"{'Tighter protection.' if new_stop_loss < cfg.risk.stop_loss_pct * 100 else 'More room to breathe but bigger potential loss.'}")
            changes["risk.stop_loss_pct"] = new_stop_loss / 100

        new_trail_mult = st.slider(
            "Trailing Stop (ATR multiplier)",
            min_value=1.0, max_value=4.0, value=cfg.risk.trailing_stop_atr_mult, step=0.5,
            help="Trailing stop = this × ATR below highest price since entry.",
        )
        if new_trail_mult != cfg.risk.trailing_stop_atr_mult:
            example_atr = 4.50  # typical for NVDA
            stop_dist = new_trail_mult * example_atr
            st.info(f"📏 **Impact (NVDA example):** ATR=$4.50 → stop trails ${stop_dist:.1f} below peak. "
                    f"{'Tighter — exits sooner, protects more.' if new_trail_mult < cfg.risk.trailing_stop_atr_mult else 'Wider — rides trends longer, risks more.'}")
            changes["risk.trailing_stop_atr_mult"] = new_trail_mult

        new_rr_ratio = st.slider(
            "Take-Profit Ratio (reward:risk)",
            min_value=1.0, max_value=5.0, value=cfg.risk.reward_risk_ratio, step=0.5,
            help="Sell when profit = this × your risk amount.",
        )
        if new_rr_ratio != cfg.risk.reward_risk_ratio:
            example_risk = 9.0
            tp = new_rr_ratio * example_risk
            st.info(f"🎯 **Impact:** Risk $9 per share → take profit at +${tp:.0f}. "
                    f"{'Smaller wins but more frequent.' if new_rr_ratio < cfg.risk.reward_risk_ratio else 'Bigger wins but fewer of them.'}")
            changes["risk.reward_risk_ratio"] = new_rr_ratio

    st.divider()

    # ── Circuit Breakers ──────────────────────────────────────────
    st.subheader("Circuit Breakers")
    col1, col2 = st.columns(2)
    with col1:
        new_daily_limit = st.slider(
            "Daily Loss Limit (%)",
            min_value=1, max_value=5, value=int(cfg.risk.daily_loss_limit * 100), step=1,
            help="Pause all trading if portfolio drops this much in one day.",
        )
        if new_daily_limit != int(cfg.risk.daily_loss_limit * 100):
            dollar = new_daily_limit * 1000
            st.info(f"⚡ **Impact:** Trading pauses after -${dollar:,} on a $100K portfolio in one day.")
            changes["risk.daily_loss_limit"] = new_daily_limit / 100

    with col2:
        new_max_dd = st.slider(
            "Max Drawdown Kill Switch (%)",
            min_value=5, max_value=30, value=int(cfg.risk.max_drawdown_pct * 100), step=5,
            help="Close ALL positions if portfolio drops this much from peak. Requires manual reset.",
        )
        if new_max_dd != int(cfg.risk.max_drawdown_pct * 100):
            dollar = new_max_dd * 1000
            st.warning(f"☠️ **Impact:** All positions liquidated after -${dollar:,} from peak. "
                       f"{'Very aggressive protection.' if new_max_dd < cfg.risk.max_drawdown_pct * 100 else '⚠️ More room but bigger potential loss.'}")
            changes["risk.max_drawdown_pct"] = new_max_dd / 100

    st.divider()

    # ── Correlation & Diversification ─────────────────────────────
    st.subheader("Diversification")
    new_max_corr = st.slider(
        "Max Portfolio Correlation",
        min_value=0.40, max_value=0.90, value=cfg.risk.max_portfolio_corr, step=0.05,
        help="Block trades if average correlation with held positions exceeds this.",
    )
    if new_max_corr != cfg.risk.max_portfolio_corr:
        st.info(f"🔗 **Impact:** "
                f"{'Stricter — forces more diversification, may block good trades in same sector.' if new_max_corr < cfg.risk.max_portfolio_corr else 'Looser — allows more concentrated bets, higher sector risk.'}")
        changes["risk.max_portfolio_corr"] = new_max_corr

    new_max_crypto = st.slider(
        "Max Crypto Exposure (%)",
        min_value=0, max_value=30, value=int(cfg.risk.max_crypto_pct * 100), step=5,
        help="Maximum total portfolio in crypto assets.",
    )
    if new_max_crypto != int(cfg.risk.max_crypto_pct * 100):
        changes["risk.max_crypto_pct"] = new_max_crypto / 100

    st.divider()

    # ── Schedule ──────────────────────────────────────────────────
    st.subheader("Schedule")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_signal_int = st.selectbox(
            "Signal Interval (min)",
            options=[5, 10, 15, 30, 60],
            index=[5, 10, 15, 30, 60].index(cfg.schedule.signal_interval_min)
                if cfg.schedule.signal_interval_min in [5, 10, 15, 30, 60] else 3,
            help="How often to scan for new trades. 30 min recommended for daily indicators.",
        )
        if new_signal_int != cfg.schedule.signal_interval_min:
            evals = (390 // new_signal_int) * 35 * 4
            st.info(f"📊 **Impact:** ~{evals:,} strategy evaluations/day")
            changes["schedule.signal_interval_min"] = new_signal_int

    with col2:
        new_pos_int = st.selectbox(
            "Position Check Interval (min)",
            options=[1, 2, 5, 10, 15],
            index=[1, 2, 5, 10, 15].index(cfg.schedule.position_check_interval_min)
                if cfg.schedule.position_check_interval_min in [1, 2, 5, 10, 15] else 2,
            help="How often to check trailing stops and take-profits.",
        )
        if new_pos_int != cfg.schedule.position_check_interval_min:
            changes["schedule.position_check_interval_min"] = new_pos_int

    with col3:
        new_sim = st.toggle(
            "Simulation Mode (24/7)",
            value=cfg.schedule.force_market_hours,
            help="Run signals outside market hours for testing. Uses stale prices.",
        )
        if new_sim != cfg.schedule.force_market_hours:
            changes["schedule.force_market_hours"] = new_sim

    st.divider()

    # ── Strategy Toggles ──────────────────────────────────────────
    st.subheader("Strategy Toggles")
    st.caption("Disable individual strategies without stopping the whole system.")

    _STRATEGIES_FILE = Path(get_config().data.storage_path) / "disabled_strategies.json"

    import json
    disabled = set()
    if _STRATEGIES_FILE.exists():
        disabled = set(json.loads(_STRATEGIES_FILE.read_text()))

    all_strategies = ["momentum", "trend_following", "volatility_breakout", "mean_reversion"]
    if cfg.ml_ensemble.enabled:
        all_strategies.append("ml_ensemble")

    cols = st.columns(len(all_strategies))
    strategy_changes = False
    for i, strat in enumerate(all_strategies):
        with cols[i]:
            enabled = strat not in disabled
            new_val = st.toggle(strat.replace("_", " ").title(), value=enabled, key=f"strat_{strat}",
                                help=f"{'Active' if enabled else 'Disabled'} — "
                                     f"{'disable to stop this strategy from generating signals' if enabled else 'enable to resume signal generation'}")
            if new_val != enabled:
                strategy_changes = True
                if new_val:
                    disabled.discard(strat)
                else:
                    disabled.add(strat)

    if strategy_changes:
        if st.button("💾 Save Strategy Toggles"):
            _STRATEGIES_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STRATEGIES_FILE.write_text(json.dumps(list(disabled)))
            st.success(f"Saved! Disabled strategies: {list(disabled) or 'none'}")

    st.divider()

    # ── Changes summary + Save ────────────────────────────────────
    if changes:
        st.subheader(f"📝 {len(changes)} Pending Change(s)")
        for key, val in changes.items():
            section, param = key.split(".")
            old_val = _get_nested(cfg, section, param)
            st.write(f"  • `{key}`: **{old_val}** → **{val}**")

        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("💾 Save to config.yaml", type="primary"):
                _apply_changes(changes)
                st.success("Config saved! Restart the pipeline for changes to take effect.")
                st.cache_data.clear()
                st.cache_resource.clear()
        with col_cancel:
            if st.button("Cancel"):
                st.rerun()
    else:
        st.success("✅ No changes — all values match current config.")


def _get_nested(cfg, section: str, param: str):
    """Get a nested config value like cfg.risk.min_confidence."""
    return getattr(getattr(cfg, section), param)


def _apply_changes(changes: dict) -> None:
    """Write changes to config.yaml."""
    with open(_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    for key, val in changes.items():
        section, param = key.split(".")
        if section not in raw:
            raw[section] = {}
        raw[section][param] = val

    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
