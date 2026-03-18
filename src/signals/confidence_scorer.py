"""
signals/confidence_scorer.py — Statistical confidence scoring for raw signals.

Confidence is computed STATISTICALLY — NOT by an LLM. Formula:
    base_confidence   = historical_win_rate(pattern, strategy, lookback=60)
    regime_mult       = 1.0 if bull_regime else 0.85
    volume_mult       = 1.05 if high_volume else 0.95
    final_confidence  = clamp(base * regime_mult * volume_mult, 0.0, 1.0)

Signals below config.risk.min_confidence (0.60) are blocked here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.config import get_config
from src.signals.strategies.momentum import RawSignal


@dataclass
class ScoredSignal:
    """RawSignal enriched with a statistical confidence score."""
    ticker: str
    direction: str
    strategy: str
    pattern: str
    strength: float
    confidence: float           # 0.0–1.0 final score
    base_confidence: float      # historical win rate
    regime_multiplier: float
    volume_multiplier: float
    reason: str
    blocked: bool               # True if confidence < min_confidence
    block_reason: str
    features_snapshot: dict


class ConfidenceScorer:
    """
    Scores RawSignal objects using historical win rates and regime adjustments.

    Historical win rates are computed from stored signal outcomes in Parquet.
    If no historical data exists for a pattern, a conservative default is used.
    """

    # Conservative default win rate when no history available for a pattern
    DEFAULT_WIN_RATE = 0.52

    # Per-pattern default win rates (seed values before real history accumulates)
    PATTERN_DEFAULTS: dict[str, float] = {
        # Momentum
        "rsi_oversold_macd_cross_up":   0.62,
        "macd_cross_up_high_volume":    0.58,
        "strong_roc_bull_regime":       0.55,
        "rsi_overbought_macd_cross_down": 0.60,
        "macd_cross_down_below_ema":    0.57,
        "strong_neg_roc_bear_regime":   0.54,
        # Trend following
        "golden_cross_high_volume":     0.68,
        "golden_cross":                 0.63,
        "full_ma_alignment_adx":        0.65,
        "near_52w_high_uptrend":        0.60,
        "death_cross":                  0.66,
        "below_sma50_downtrend":        0.61,
        "trend_weakening":              0.55,
        # Volatility breakout
        "bb_squeeze_breakout_up":       0.67,
        "bb_upper_breakout":            0.59,
        "vol_expansion_bull":           0.56,
        "bb_squeeze_breakout_down":     0.66,
        "bb_lower_breakdown":           0.58,
        "vol_expansion_bear":           0.55,
    }

    def __init__(self, store=None) -> None:
        """
        Args:
            store: Optional ParquetStore instance for loading historical signals.
                   If None, uses seed default win rates only.
        """
        self.config = get_config()
        self.store  = store

    def score(
        self,
        signal: RawSignal,
        ticker: str,
        features: Optional[dict] = None,
    ) -> ScoredSignal:
        """
        Compute final confidence score for a single raw signal.

        Args:
            signal:   RawSignal from a strategy
            ticker:   Ticker symbol
            features: Optional feature dict (uses signal.features_snapshot if not given)

        Returns:
            ScoredSignal with final confidence and block status.
        """
        feats = features or signal.features_snapshot

        # 1. Base confidence from historical win rate
        base = self._get_win_rate(signal.pattern, signal.strategy, ticker)

        # 2. Regime adjustment
        bull_regime = bool(feats.get("bull_regime", 0))
        regime_mult = 1.0 if bull_regime else 0.85

        # 3. Volume adjustment
        high_volume = bool(feats.get("high_volume", 0))
        vol_mult = 1.05 if high_volume else 0.95

        # 4. Final confidence (clamped)
        final = min(1.0, max(0.0, base * regime_mult * vol_mult))

        # 5. Block check
        min_conf = self.config.risk.min_confidence
        blocked = final < min_conf
        block_reason = (
            f"Confidence {final:.3f} below minimum {min_conf}" if blocked else ""
        )

        if blocked:
            logger.debug(
                f"[ConfidenceScorer] {ticker}/{signal.strategy} blocked: "
                f"confidence={final:.3f} < {min_conf}"
            )
        else:
            logger.info(
                f"[ConfidenceScorer] {ticker}/{signal.strategy} {signal.direction}: "
                f"confidence={final:.3f} (base={base:.3f}, regime={regime_mult}, vol={vol_mult})"
            )

        return ScoredSignal(
            ticker=ticker,
            direction=signal.direction,
            strategy=signal.strategy,
            pattern=signal.pattern,
            strength=signal.strength,
            confidence=final,
            base_confidence=base,
            regime_multiplier=regime_mult,
            volume_multiplier=vol_mult,
            reason=signal.reason,
            blocked=blocked,
            block_reason=block_reason,
            features_snapshot=feats,
        )

    def score_all(
        self,
        signals: List[RawSignal],
        ticker: str,
    ) -> List[ScoredSignal]:
        """
        Score all signals for a ticker. HOLD signals are skipped.

        Args:
            signals: List of RawSignal from SignalDetector
            ticker:  Ticker symbol

        Returns:
            List of ScoredSignal (excludes HOLDs and blocked signals).
        """
        results = []
        for raw in signals:
            if raw.direction == "HOLD":
                continue
            scored = self.score(raw, ticker)
            results.append(scored)
        return results

    def _get_win_rate(self, pattern: str, strategy: str, ticker: str) -> float:
        """
        Look up historical win rate for a pattern from Parquet store.
        Falls back to pattern default, then global default.

        Args:
            pattern:  Pattern identifier (e.g. "golden_cross_high_volume")
            strategy: Strategy name
            ticker:   Ticker symbol

        Returns:
            Win rate as float in [0.0, 1.0]
        """
        if self.store is not None:
            try:
                win_rate = self._compute_from_history(pattern, strategy, ticker)
                if win_rate is not None:
                    return win_rate
            except Exception as exc:
                logger.warning(
                    f"[ConfidenceScorer] Failed to load history for {pattern}: {exc}"
                )

        # Fall back to seed defaults
        default = self.PATTERN_DEFAULTS.get(pattern, self.DEFAULT_WIN_RATE)
        logger.debug(
            f"[ConfidenceScorer] Using seed win rate {default:.3f} for pattern={pattern}"
        )
        return default

    def _compute_from_history(
        self, pattern: str, strategy: str, ticker: str
    ) -> Optional[float]:
        """
        Compute win rate from stored signal outcomes in Parquet.

        Args:
            pattern:  Pattern identifier
            strategy: Strategy name
            ticker:   Ticker symbol

        Returns:
            Win rate or None if insufficient history.
        """
        lookback = self.config.signals.confidence_lookback
        try:
            df = self.store.load_signals()
            if df.empty:
                return None

            # Filter to relevant pattern+strategy+ticker outcomes
            mask = (
                (df["pattern"] == pattern) &
                (df["strategy"] == strategy) &
                (df["ticker"] == ticker) &
                (df["outcome"].notna())
            )
            subset = df[mask].tail(lookback)
            if len(subset) < 10:
                return None  # Not enough history

            win_rate = (subset["outcome"] == "WIN").sum() / len(subset)
            logger.debug(
                f"[ConfidenceScorer] Historical win rate for {ticker}/{pattern}: "
                f"{win_rate:.3f} ({len(subset)} trades)"
            )
            return float(win_rate)
        except Exception:
            return None

    @staticmethod
    def scored_signals_to_dataframe(
        scored: List[ScoredSignal],
    ) -> pd.DataFrame:
        """
        Convert scored signals to a DataFrame for Parquet storage.

        Args:
            scored: List of ScoredSignal

        Returns:
            DataFrame with one row per scored signal.
        """
        if not scored:
            return pd.DataFrame()

        rows = []
        for s in scored:
            rows.append({
                "ticker":            s.ticker,
                "direction":         s.direction,
                "strategy":          s.strategy,
                "pattern":           s.pattern,
                "strength":          s.strength,
                "confidence":        s.confidence,
                "base_confidence":   s.base_confidence,
                "regime_multiplier": s.regime_multiplier,
                "volume_multiplier": s.volume_multiplier,
                "reason":            s.reason,
                "blocked":           s.blocked,
                "block_reason":      s.block_reason,
            })
        return pd.DataFrame(rows)
