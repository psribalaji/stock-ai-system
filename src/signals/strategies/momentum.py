"""
signals/strategies/momentum.py — Momentum-based signal strategy.

Detects BUY/SELL signals based on RSI, MACD, and price momentum.
Returns raw signal candidates; confidence scoring is done by ConfidenceScorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.config import get_config


@dataclass
class RawSignal:
    """Output of a single strategy evaluation."""
    direction: str          # "BUY" | "SELL" | "HOLD"
    strategy: str           # strategy name
    strength: float         # 0.0–1.0 pre-confidence strength
    reason: str             # human-readable explanation
    pattern: str            # pattern identifier for win-rate lookup
    features_snapshot: dict = field(default_factory=dict)


class MomentumStrategy:
    """
    Momentum strategy: detects trend acceleration via RSI, MACD, and ROC.

    BUY conditions (any triggers):
      - RSI recovering from oversold (<35) with MACD cross up
      - MACD cross up + high volume
      - Strong positive ROC (>2% in 5 days) + RSI not overbought

    SELL conditions (any triggers):
      - RSI rolling over from overbought (>70) with MACD cross down
      - MACD cross down + price below EMA21
      - Strong negative ROC + RSI not oversold
    """

    NAME = "momentum"

    def __init__(self) -> None:
        self.config = get_config()

    def evaluate(self, features: dict, ticker: str = "") -> RawSignal:
        """
        Evaluate momentum conditions on latest feature snapshot.

        Args:
            features: Dict from FeatureEngine.get_latest_features()
            ticker:   Ticker symbol (for logging)

        Returns:
            RawSignal with direction, strength, and reason.
        """
        cfg = self.config.signals

        rsi          = features.get("rsi_14")
        macd_hist    = features.get("macd_hist")
        macd_cross_up   = features.get("macd_cross_up")
        macd_cross_down = features.get("macd_cross_down")
        roc_5        = features.get("roc_5")
        roc_20       = features.get("roc_20")
        high_volume  = features.get("high_volume")
        price_vs_ema21 = features.get("price_vs_sma50")  # proxy for EMA21 position
        bull_regime  = features.get("bull_regime")

        # ── BUY signals ──────────────────────────────────────────────

        # Pattern 1: RSI recovering from oversold + MACD cross up
        if (rsi is not None and rsi < cfg.rsi_oversold and
                macd_cross_up and macd_hist is not None and macd_hist > 0):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.75,
                reason=f"RSI oversold recovery ({rsi:.1f}) with MACD cross up",
                pattern="rsi_oversold_macd_cross_up",
                features_snapshot=features,
            )

        # Pattern 2: MACD cross up + high volume confirmation
        if macd_cross_up and high_volume:
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.65,
                reason="MACD cross up confirmed by high volume",
                pattern="macd_cross_up_high_volume",
                features_snapshot=features,
            )

        # Pattern 3: Strong ROC momentum, RSI mid-range (not overbought)
        if (roc_5 is not None and roc_5 > 2.0 and
                rsi is not None and rsi < cfg.rsi_overbought and
                bull_regime):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.60,
                reason=f"Strong 5d ROC ({roc_5:.1f}%) in bull regime",
                pattern="strong_roc_bull_regime",
                features_snapshot=features,
            )

        # ── SELL signals ──────────────────────────────────────────────

        # Pattern 4: RSI overbought rolling over + MACD cross down
        if (rsi is not None and rsi > cfg.rsi_overbought and
                macd_cross_down and macd_hist is not None and macd_hist < 0):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.75,
                reason=f"RSI overbought ({rsi:.1f}) with MACD cross down",
                pattern="rsi_overbought_macd_cross_down",
                features_snapshot=features,
            )

        # Pattern 5: MACD cross down + price below EMA21
        if (macd_cross_down and
                price_vs_ema21 is not None and price_vs_ema21 < 0):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.60,
                reason="MACD cross down with price below short-term MA",
                pattern="macd_cross_down_below_ema",
                features_snapshot=features,
            )

        # Pattern 6: Strong negative ROC + RSI mid-range (not oversold)
        if (roc_5 is not None and roc_5 < -2.0 and
                rsi is not None and rsi > cfg.rsi_oversold and
                not bull_regime):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.55,
                reason=f"Strong negative 5d ROC ({roc_5:.1f}%) in non-bull regime",
                pattern="strong_neg_roc_bear_regime",
                features_snapshot=features,
            )

        logger.debug(f"[{self.NAME}] HOLD for {ticker}: no momentum pattern triggered")
        return RawSignal(
            direction="HOLD",
            strategy=self.NAME,
            strength=0.0,
            reason="No momentum pattern triggered",
            pattern="hold",
            features_snapshot=features,
        )
