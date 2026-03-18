"""
risk/risk_manager.py — Risk validation and position sizing.

Enforces all hardcoded risk rules before a signal reaches DecisionEngine.
Rules are defined in config.yaml but treated as immutable at runtime.

Risk rules:
  - Max position size: 5% of portfolio per trade
  - Max crypto exposure: 10% of portfolio total
  - Max open positions: 5 simultaneously
  - Daily loss circuit breaker: -2% triggers pause
  - Max drawdown kill switch: -15% closes all positions
  - Minimum confidence to trade: 0.60
  - Stop loss on every trade: 7% below entry price
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.config import get_config
from src.signals.confidence_scorer import ScoredSignal


@dataclass
class RiskDecision:
    """Output of RiskManager.validate()."""
    approved: bool
    block_reason: str           # Empty string if approved
    position_size_pct: float    # Fraction of portfolio (e.g. 0.05 = 5%)
    position_size_usd: float    # Dollar amount
    stop_loss_price: float      # Hard stop price
    stop_loss_pct: float        # e.g. 0.07 = 7%
    risk_notes: list[str]       # Non-blocking observations


@dataclass
class PortfolioState:
    """
    Snapshot of current portfolio for risk checks.
    Provided by the caller (e.g. DecisionEngine pulling from Alpaca).
    """
    total_value_usd: float
    cash_usd: float
    open_positions: int
    crypto_exposure_usd: float
    daily_pnl_pct: float        # e.g. -0.015 = -1.5% today
    peak_value_usd: float       # All-time peak for drawdown calc


class RiskManager:
    """
    Validates signals against all portfolio risk rules.

    Usage:
        rm = RiskManager()
        state = PortfolioState(...)
        decision = rm.validate(scored_signal, entry_price, state, ticker)
    """

    def __init__(self) -> None:
        self.config = get_config()
        self._paused = False        # Circuit breaker flag (daily loss)
        self._killed = False        # Kill switch flag (max drawdown)

    # ── Main entry point ─────────────────────────────────────────────

    def validate(
        self,
        signal: ScoredSignal,
        entry_price: float,
        portfolio: PortfolioState,
        ticker: str,
    ) -> RiskDecision:
        """
        Validate a scored signal against all risk rules.

        Args:
            signal:       ScoredSignal from ConfidenceScorer
            entry_price:  Current/expected fill price
            portfolio:    Current portfolio state
            ticker:       Ticker symbol

        Returns:
            RiskDecision with approved status, position size, and stop loss.
        """
        cfg = self.config.risk
        notes: list[str] = []

        # ── 1. Kill switch check ─────────────────────────────────────
        if self._killed:
            return self._block("Kill switch active: max drawdown exceeded")

        drawdown = self._calc_drawdown(portfolio)
        if drawdown >= cfg.max_drawdown_pct:
            self._killed = True
            logger.critical(
                f"[RiskManager] KILL SWITCH: drawdown={drawdown:.2%} >= {cfg.max_drawdown_pct:.2%}. "
                f"All positions should be closed."
            )
            return self._block(
                f"Kill switch triggered: drawdown {drawdown:.2%} >= limit {cfg.max_drawdown_pct:.2%}"
            )

        # ── 2. Circuit breaker check ─────────────────────────────────
        if self._paused:
            return self._block("Circuit breaker active: daily loss limit reached")

        if portfolio.daily_pnl_pct <= -cfg.daily_loss_limit:
            self._paused = True
            logger.warning(
                f"[RiskManager] CIRCUIT BREAKER: daily P&L={portfolio.daily_pnl_pct:.2%}. "
                f"Trading paused for the day."
            )
            return self._block(
                f"Circuit breaker: daily P&L {portfolio.daily_pnl_pct:.2%} "
                f"<= -{cfg.daily_loss_limit:.2%}"
            )

        # ── 3. Confidence check ───────────────────────────────────────
        if signal.confidence < cfg.min_confidence:
            return self._block(
                f"Confidence {signal.confidence:.3f} < minimum {cfg.min_confidence}"
            )

        # ── 4. Max open positions ─────────────────────────────────────
        if portfolio.open_positions >= cfg.max_open_positions:
            return self._block(
                f"Max open positions reached ({portfolio.open_positions}/{cfg.max_open_positions})"
            )

        # ── 5. Crypto exposure ────────────────────────────────────────
        is_crypto = ticker in self.config.assets.crypto
        if is_crypto:
            crypto_pct = portfolio.crypto_exposure_usd / portfolio.total_value_usd
            if crypto_pct >= cfg.max_crypto_pct:
                return self._block(
                    f"Max crypto exposure reached: {crypto_pct:.2%} >= {cfg.max_crypto_pct:.2%}"
                )
            if crypto_pct > cfg.max_crypto_pct * 0.8:
                notes.append(
                    f"Crypto exposure {crypto_pct:.2%} approaching limit {cfg.max_crypto_pct:.2%}"
                )

        # ── 6. Position sizing ────────────────────────────────────────
        position_pct = cfg.max_position_pct
        position_usd = portfolio.total_value_usd * position_pct

        # Cap at available cash
        if position_usd > portfolio.cash_usd:
            position_usd = portfolio.cash_usd
            position_pct = position_usd / portfolio.total_value_usd
            notes.append(
                f"Position size capped at available cash: ${position_usd:,.2f}"
            )

        if position_usd < 1.0:
            return self._block(f"Insufficient cash: ${portfolio.cash_usd:.2f}")

        # ── 7. Stop loss ──────────────────────────────────────────────
        stop_loss_price = entry_price * (1 - cfg.stop_loss_pct)

        # ── Approved ──────────────────────────────────────────────────
        logger.info(
            f"[RiskManager] APPROVED {ticker} {signal.direction}: "
            f"size=${position_usd:,.2f} ({position_pct:.2%}), "
            f"stop=${stop_loss_price:.2f} (-{cfg.stop_loss_pct:.2%})"
        )

        return RiskDecision(
            approved=True,
            block_reason="",
            position_size_pct=position_pct,
            position_size_usd=position_usd,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=cfg.stop_loss_pct,
            risk_notes=notes,
        )

    # ── Circuit breaker reset ─────────────────────────────────────────

    def reset_daily(self) -> None:
        """Reset daily circuit breaker. Call at start of each trading day."""
        self._paused = False
        logger.info("[RiskManager] Daily circuit breaker reset")

    def reset_kill_switch(self) -> None:
        """
        Reset kill switch. Should only be called after manual review.
        This is a safeguard — do not automate this.
        """
        self._killed = False
        logger.warning("[RiskManager] Kill switch manually reset — ensure drawdown is resolved")

    @property
    def is_paused(self) -> bool:
        """True if circuit breaker is active."""
        return self._paused

    @property
    def is_killed(self) -> bool:
        """True if kill switch is active."""
        return self._killed

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _calc_drawdown(portfolio: PortfolioState) -> float:
        """
        Compute current drawdown from peak.

        Args:
            portfolio: Current portfolio state

        Returns:
            Drawdown as positive fraction (e.g. 0.12 = 12% drawdown)
        """
        if portfolio.peak_value_usd <= 0:
            return 0.0
        dd = (portfolio.peak_value_usd - portfolio.total_value_usd) / portfolio.peak_value_usd
        return max(0.0, dd)

    @staticmethod
    def _block(reason: str) -> RiskDecision:
        """Return a blocked RiskDecision with the given reason."""
        logger.warning(f"[RiskManager] BLOCKED: {reason}")
        return RiskDecision(
            approved=False,
            block_reason=reason,
            position_size_pct=0.0,
            position_size_usd=0.0,
            stop_loss_price=0.0,
            stop_loss_pct=0.0,
            risk_notes=[],
        )

    def get_status(self) -> dict:
        """Return current risk manager status for monitoring."""
        return {
            "paused":  self._paused,
            "killed":  self._killed,
            "max_position_pct":    self.config.risk.max_position_pct,
            "max_open_positions":  self.config.risk.max_open_positions,
            "daily_loss_limit":    self.config.risk.daily_loss_limit,
            "max_drawdown_pct":    self.config.risk.max_drawdown_pct,
        }
