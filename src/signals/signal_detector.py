"""
signals/signal_detector.py — Runs all strategies and aggregates raw signals.

Orchestrates the three strategy classes and returns a list of RawSignal objects
(one per strategy) for a given ticker's feature snapshot.
"""
from __future__ import annotations

from typing import List

import pandas as pd
from loguru import logger

from src.config import get_config
from src.features.feature_engine import FeatureEngine
from src.signals.strategies.momentum import MomentumStrategy, RawSignal
from src.signals.strategies.trend_following import TrendFollowingStrategy
from src.signals.strategies.volatility_breakout import VolatilityBreakoutStrategy


class SignalDetector:
    """
    Runs all strategies on a feature-enriched DataFrame and returns signals.

    Usage:
        detector = SignalDetector()
        signals  = detector.detect("NVDA", feature_df)
        # signals is a list[RawSignal] — one entry per strategy
    """

    def __init__(self) -> None:
        self.config   = get_config()
        self.engine   = FeatureEngine()
        self.strategies = [
            MomentumStrategy(),
            TrendFollowingStrategy(),
            VolatilityBreakoutStrategy(),
        ]

    def detect(self, ticker: str, df: pd.DataFrame) -> List[RawSignal]:
        """
        Compute features if needed and run all strategies.

        Args:
            ticker: Ticker symbol (e.g. "NVDA")
            df:     OHLCV DataFrame — will have features computed if missing

        Returns:
            List of RawSignal, one per strategy. Empty list if data insufficient.
        """
        if df.empty or len(df) < 50:
            logger.warning(f"[SignalDetector] Insufficient data for {ticker}: {len(df)} rows")
            return []

        # Compute features if the DataFrame doesn't already have them
        if "rsi_14" not in df.columns:
            df = self.engine.compute_all(df, ticker)

        features = self.engine.get_latest_features(df)
        if not features:
            logger.warning(f"[SignalDetector] No features extracted for {ticker}")
            return []

        signals: List[RawSignal] = []
        for strategy in self.strategies:
            try:
                signal = strategy.evaluate(features, ticker)
                signals.append(signal)
                if signal.direction != "HOLD":
                    logger.info(
                        f"[{strategy.NAME}] {ticker}: {signal.direction} "
                        f"(strength={signal.strength:.2f}) — {signal.reason}"
                    )
            except Exception as exc:
                logger.error(
                    f"[SignalDetector] Strategy {strategy.NAME} failed for {ticker}: {exc}"
                )

        return signals

    def detect_actionable(self, ticker: str, df: pd.DataFrame) -> List[RawSignal]:
        """
        Return only BUY or SELL signals (filters out HOLD).

        Args:
            ticker: Ticker symbol
            df:     OHLCV DataFrame

        Returns:
            List of non-HOLD RawSignal objects.
        """
        return [s for s in self.detect(ticker, df) if s.direction != "HOLD"]

    @staticmethod
    def signals_to_dataframe(signals: List[RawSignal], ticker: str) -> pd.DataFrame:
        """
        Convert a list of RawSignal objects to a DataFrame for storage/inspection.

        Args:
            signals: List of RawSignal from detect()
            ticker:  Ticker symbol

        Returns:
            DataFrame with one row per signal.
        """
        if not signals:
            return pd.DataFrame()

        rows = []
        for s in signals:
            rows.append({
                "ticker":    ticker,
                "direction": s.direction,
                "strategy":  s.strategy,
                "strength":  s.strength,
                "reason":    s.reason,
                "pattern":   s.pattern,
            })
        return pd.DataFrame(rows)
