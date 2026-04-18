"""
execution/decision_engine.py — Final trade decision orchestrator.

Combines:
  1. SignalDetector   → raw signals from all strategies
  2. ConfidenceScorer → statistical confidence per signal
  3. LLMAnalysisService → reasoning text (optional, enrichment only)
  4. RiskManager      → position sizing + rule validation

Produces a list of TradeDecision objects — one per approved signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.config import get_config
from src.signals.signal_detector import SignalDetector
from src.signals.confidence_scorer import ConfidenceScorer, ScoredSignal
from src.risk.risk_manager import RiskManager, PortfolioState, RiskDecision


@dataclass
class TradeDecision:
    """Final output of the decision pipeline for a single signal."""
    timestamp: datetime
    ticker: str
    direction: str              # "BUY" | "SELL"
    strategy: str
    pattern: str
    confidence: float
    position_size_pct: float
    position_size_usd: float
    entry_price: float
    stop_loss_price: float
    stop_loss_pct: float
    trailing_stop_price: float  # ATR-based trailing stop (same as stop_loss if ATR unavailable)
    trailing_stop_atr: float    # ATR value used — 0.0 if ATR unavailable
    take_profit_price: float    # 2:1 reward-to-risk target
    signal_reason: str
    llm_reasoning: str          # From LLMAnalysisService (empty if unavailable)
    llm_summary: str
    approved: bool
    block_reason: str           # Empty if approved
    risk_notes: list[str] = field(default_factory=list)


class DecisionEngine:
    """
    Orchestrates the full signal → decision pipeline for a single ticker.

    Usage:
        engine = DecisionEngine()
        portfolio = PortfolioState(...)
        decisions = engine.decide("NVDA", ohlcv_df, entry_price=485.50, portfolio=portfolio)
        # decisions is a list[TradeDecision] of APPROVED trades only
    """

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        llm_service=None,
    ) -> None:
        """
        Args:
            risk_manager: Optional shared RiskManager (allows circuit breaker state to persist)
            llm_service:  Optional LLMAnalysisService (skipped if None — no API key needed in tests)
        """
        self.config   = get_config()
        self.detector = SignalDetector()
        self.scorer   = ConfidenceScorer()
        self.risk     = risk_manager or RiskManager()
        self.llm      = llm_service  # None is fine — LLM is enrichment only

    def decide(
        self,
        ticker: str,
        df: pd.DataFrame,
        entry_price: float,
        portfolio: PortfolioState,
        news_summary: Optional[str] = None,
    ) -> List[TradeDecision]:
        """
        Run full pipeline for a single ticker and return approved decisions.

        Args:
            ticker:       Ticker symbol (e.g. "NVDA")
            df:           OHLCV DataFrame (features computed internally if missing)
            entry_price:  Current market price for position sizing + stop loss
            portfolio:    Current portfolio state for risk checks
            news_summary: Optional news text for LLM enrichment

        Returns:
            List of approved TradeDecision objects (may be empty).
        """
        logger.info(f"[DecisionEngine] Processing {ticker} @ ${entry_price:.2f}")

        # ── Step 1: Detect signals ────────────────────────────────────
        raw_signals = self.detector.detect_actionable(ticker, df)
        if not raw_signals:
            logger.debug(f"[DecisionEngine] No actionable signals for {ticker}")
            return []

        # ── Step 2: Score confidence ──────────────────────────────────
        scored_signals = self.scorer.score_all(raw_signals, ticker)
        passed = [s for s in scored_signals if not s.blocked]
        if not passed:
            logger.info(f"[DecisionEngine] All signals blocked (low confidence) for {ticker}")
            return []

        # ── Step 3: Risk validation ───────────────────────────────────
        decisions: List[TradeDecision] = []
        for signal in passed:
            decision = self._process_signal(
                signal, ticker, entry_price, portfolio, news_summary
            )
            if decision.approved:
                decisions.append(decision)

        logger.info(
            f"[DecisionEngine] {ticker}: {len(decisions)} approved decision(s) "
            f"from {len(raw_signals)} raw signal(s)"
        )
        return decisions

    def decide_all(
        self,
        tickers: List[str],
        data_map: dict,             # {ticker: pd.DataFrame}
        price_map: dict,            # {ticker: float}
        portfolio: PortfolioState,
        news_map: Optional[dict] = None,   # {ticker: str}
    ) -> List[TradeDecision]:
        """
        Run the pipeline for all tickers in the asset universe.

        Args:
            tickers:   List of ticker symbols
            data_map:  Dict mapping ticker → OHLCV DataFrame
            price_map: Dict mapping ticker → current price
            portfolio: Shared portfolio state
            news_map:  Optional dict mapping ticker → news summary

        Returns:
            Combined list of approved TradeDecision objects.
        """
        all_decisions: List[TradeDecision] = []

        for ticker in tickers:
            df = data_map.get(ticker, pd.DataFrame())
            price = price_map.get(ticker, 0.0)
            news = (news_map or {}).get(ticker)

            if df.empty or price <= 0:
                logger.warning(f"[DecisionEngine] Skipping {ticker}: no data or invalid price")
                continue

            try:
                decisions = self.decide(ticker, df, price, portfolio, news)
                all_decisions.extend(decisions)
            except Exception as exc:
                logger.error(f"[DecisionEngine] Failed for {ticker}: {exc}")

        return all_decisions

    # ── Internal helpers ──────────────────────────────────────────────

    def _process_signal(
        self,
        signal: ScoredSignal,
        ticker: str,
        entry_price: float,
        portfolio: PortfolioState,
        news_summary: Optional[str],
    ) -> TradeDecision:
        """Run risk check + LLM enrichment for a single scored signal."""

        # Risk validation
        risk_decision: RiskDecision = self.risk.validate(
            signal, entry_price, portfolio, ticker
        )

        # LLM enrichment (only if approved and LLM is available)
        llm_reasoning = ""
        llm_summary = signal.reason
        if risk_decision.approved and self.llm is not None:
            try:
                enrichment = self.llm.enrich(signal, news_summary)
                llm_reasoning = enrichment.get("reasoning", "")
                llm_summary   = enrichment.get("summary", signal.reason)
            except Exception as exc:
                logger.warning(f"[DecisionEngine] LLM enrichment skipped for {ticker}: {exc}")

        return TradeDecision(
            timestamp=datetime.now(timezone.utc),
            ticker=ticker,
            direction=signal.direction,
            strategy=signal.strategy,
            pattern=signal.pattern,
            confidence=signal.confidence,
            position_size_pct=risk_decision.position_size_pct,
            position_size_usd=risk_decision.position_size_usd,
            entry_price=entry_price,
            stop_loss_price=risk_decision.stop_loss_price,
            stop_loss_pct=risk_decision.stop_loss_pct,
            trailing_stop_price=risk_decision.trailing_stop_price,
            trailing_stop_atr=risk_decision.trailing_stop_atr,
            take_profit_price=risk_decision.take_profit_price,
            signal_reason=signal.reason,
            llm_reasoning=llm_reasoning,
            llm_summary=llm_summary,
            approved=risk_decision.approved,
            block_reason=risk_decision.block_reason,
            risk_notes=risk_decision.risk_notes,
        )

    @staticmethod
    def decisions_to_dataframe(decisions: List[TradeDecision]) -> pd.DataFrame:
        """
        Convert approved decisions to a DataFrame for Parquet storage.

        Args:
            decisions: List of TradeDecision

        Returns:
            DataFrame with one row per decision.
        """
        if not decisions:
            return pd.DataFrame()

        rows = []
        for d in decisions:
            rows.append({
                "timestamp":         d.timestamp,
                "ticker":            d.ticker,
                "direction":         d.direction,
                "strategy":          d.strategy,
                "pattern":           d.pattern,
                "confidence":        d.confidence,
                "position_size_pct": d.position_size_pct,
                "position_size_usd": d.position_size_usd,
                "entry_price":       d.entry_price,
                "stop_loss_price":   d.stop_loss_price,
                "stop_loss_pct":     d.stop_loss_pct,
                "signal_reason":     d.signal_reason,
                "llm_summary":       d.llm_summary,
                "approved":          d.approved,
                "block_reason":      d.block_reason,
            })
        return pd.DataFrame(rows)
