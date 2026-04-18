"""
signals/strategies/mean_reversion.py — Mean-reversion signal strategy.

Detects BUY/SELL signals when price deviates too far from its mean,
betting on a snap-back. Complements momentum/trend strategies by
profiting in sideways/range-bound markets.
"""
from __future__ import annotations

from loguru import logger

from src.config import get_config
from src.signals.strategies.momentum import RawSignal


class MeanReversionStrategy:
    """
    Mean-reversion strategy: buys oversold extremes, sells overbought extremes.

    BUY conditions (any triggers):
      - Price at/below lower BB + RSI < 30 + stochastic %K crossing up from < 20
      - RSI deeply oversold (< 25) + high volume confirmation
      - BB pct < 0 + price recovering (positive 1d return) + non-high-vol regime

    SELL conditions (any triggers):
      - Price at/above upper BB + RSI > 75 + stochastic %K crossing down from > 80
      - RSI deeply overbought (> 80) + high volume confirmation
      - BB pct > 1 + price declining (negative 1d return) + non-high-vol regime
    """

    NAME = "mean_reversion"

    def __init__(self) -> None:
        self.config = get_config()

    def evaluate(self, features: dict, ticker: str = "") -> RawSignal:
        rsi = features.get("rsi_14")
        bb_pct = features.get("bb_pct")
        stoch_k = features.get("stoch_k")
        stoch_d = features.get("stoch_d")
        high_volume = features.get("high_volume")
        high_vol_regime = features.get("high_vol_regime")
        return_1d = features.get("return_1d")

        # ── BUY signals ──────────────────────────────────────────────

        # Pattern 1: Lower BB bounce — price at lower band + RSI oversold + stoch crossing up
        if (bb_pct is not None and bb_pct <= 0.0
                and rsi is not None and rsi < 30
                and stoch_k is not None and stoch_d is not None
                and stoch_k < 20 and stoch_k > stoch_d):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.78,
                reason=f"Lower BB bounce: RSI={rsi:.1f}, stoch_k={stoch_k:.1f} crossing up",
                pattern="bb_lower_bounce",
                features_snapshot=features,
            )

        # Pattern 2: Deep oversold reversal with volume confirmation
        if (rsi is not None and rsi < 25 and high_volume):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.72,
                reason=f"Deep oversold reversal: RSI={rsi:.1f} with high volume",
                pattern="oversold_reversal",
                features_snapshot=features,
            )

        # Pattern 3: BB mean revert — below lower band but recovering, low-vol regime
        if (bb_pct is not None and bb_pct < 0.0
                and return_1d is not None and return_1d > 0
                and not high_vol_regime):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.60,
                reason=f"BB mean revert up: bb_pct={bb_pct:.2f}, recovering in low-vol regime",
                pattern="bb_mean_revert_up",
                features_snapshot=features,
            )

        # ── SELL signals ─────────────────────────────────────────────

        # Pattern 4: Upper BB rejection — price at upper band + RSI overbought + stoch crossing down
        if (bb_pct is not None and bb_pct >= 1.0
                and rsi is not None and rsi > 75
                and stoch_k is not None and stoch_d is not None
                and stoch_k > 80 and stoch_k < stoch_d):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.78,
                reason=f"Upper BB rejection: RSI={rsi:.1f}, stoch_k={stoch_k:.1f} crossing down",
                pattern="bb_upper_rejection",
                features_snapshot=features,
            )

        # Pattern 5: Deep overbought reversal with volume confirmation
        if (rsi is not None and rsi > 80 and high_volume):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.72,
                reason=f"Deep overbought reversal: RSI={rsi:.1f} with high volume",
                pattern="overbought_reversal",
                features_snapshot=features,
            )

        # Pattern 6: BB mean revert — above upper band but declining, low-vol regime
        if (bb_pct is not None and bb_pct > 1.0
                and return_1d is not None and return_1d < 0
                and not high_vol_regime):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.60,
                reason=f"BB mean revert down: bb_pct={bb_pct:.2f}, declining in low-vol regime",
                pattern="bb_mean_revert_down",
                features_snapshot=features,
            )

        logger.debug(f"[{self.NAME}] HOLD for {ticker}: no mean-reversion pattern triggered")
        return RawSignal(
            direction="HOLD",
            strategy=self.NAME,
            strength=0.0,
            reason="No mean-reversion pattern triggered",
            pattern="hold",
            features_snapshot=features,
        )
