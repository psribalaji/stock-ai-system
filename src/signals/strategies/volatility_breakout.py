"""
signals/strategies/volatility_breakout.py — Volatility breakout signal strategy.

Detects BUY/SELL signals based on Bollinger Band squeezes/expansions, ATR,
and historical volatility regime transitions.
"""
from __future__ import annotations

from loguru import logger

from src.config import get_config
from src.signals.strategies.momentum import RawSignal


class VolatilityBreakoutStrategy:
    """
    Volatility breakout strategy: capitalizes on vol compression → expansion.

    BUY conditions (any triggers):
      - BB squeeze (low bb_width) followed by price breakout above upper band
      - ATR expanding + price positive + high volume
      - Low-vol regime transitioning to high-vol with positive return

    SELL conditions (any triggers):
      - BB squeeze followed by price breakdown below lower band
      - ATR expanding + price negative + high volume
      - High-vol regime + sharp negative return
    """

    NAME = "volatility_breakout"

    # BB width percentile thresholds (approximate — relative to recent average)
    BB_SQUEEZE_THRESHOLD = 0.03   # < 3% band width = squeeze
    BB_EXPAND_THRESHOLD  = 0.06   # > 6% = expanding

    def __init__(self) -> None:
        self.config = get_config()

    def evaluate(self, features: dict, ticker: str = "") -> RawSignal:
        """
        Evaluate volatility breakout conditions on latest feature snapshot.

        Args:
            features: Dict from FeatureEngine.get_latest_features()
            ticker:   Ticker symbol (for logging)

        Returns:
            RawSignal with direction, strength, and reason.
        """
        bb_pct       = features.get("bb_pct")        # 0=lower, 1=upper
        bb_width     = features.get("bb_width")
        atr_14       = features.get("atr_14")
        hvol_20      = features.get("hvol_20")
        high_vol_regime = features.get("high_vol_regime")
        high_volume  = features.get("high_volume")
        return_1d    = features.get("return_1d")
        return_5d    = features.get("return_5d")
        bull_regime  = features.get("bull_regime")
        vol_ratio    = features.get("vol_ratio")

        # Derived: squeeze detected (narrow bands relative to threshold)
        squeeze = bb_width is not None and bb_width < self.BB_SQUEEZE_THRESHOLD

        # ── BUY signals ──────────────────────────────────────────────

        # Pattern 1: BB squeeze breakout above upper band
        if (squeeze and bb_pct is not None and bb_pct > 1.0 and
                high_volume and return_1d is not None and return_1d > 0):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.80,
                reason=f"BB squeeze breakout above upper band (bb_pct={bb_pct:.2f})",
                pattern="bb_squeeze_breakout_up",
                features_snapshot=features,
            )

        # Pattern 2: BB breakout above upper band (no prior squeeze required)
        if (bb_pct is not None and bb_pct > 1.0 and
                high_volume and return_1d is not None and return_1d > 0.01):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.65,
                reason=f"BB upper band breakout with volume (bb_pct={bb_pct:.2f})",
                pattern="bb_upper_breakout",
                features_snapshot=features,
            )

        # Pattern 3: Vol regime transitioning + positive momentum + bull regime
        if (high_vol_regime and bull_regime and
                return_5d is not None and return_5d > 0.03 and
                vol_ratio is not None and vol_ratio > 1.2):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.60,
                reason=f"High-vol expansion with positive 5d return ({return_5d*100:.1f}%)",
                pattern="vol_expansion_bull",
                features_snapshot=features,
            )

        # ── SELL signals ──────────────────────────────────────────────

        # Pattern 4: BB squeeze breakdown below lower band
        if (squeeze and bb_pct is not None and bb_pct < 0.0 and
                high_volume and return_1d is not None and return_1d < 0):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.80,
                reason=f"BB squeeze breakdown below lower band (bb_pct={bb_pct:.2f})",
                pattern="bb_squeeze_breakout_down",
                features_snapshot=features,
            )

        # Pattern 5: BB breakdown below lower band (no prior squeeze)
        if (bb_pct is not None and bb_pct < 0.0 and
                high_volume and return_1d is not None and return_1d < -0.01):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.65,
                reason=f"BB lower band breakdown with volume (bb_pct={bb_pct:.2f})",
                pattern="bb_lower_breakdown",
                features_snapshot=features,
            )

        # Pattern 6: High-vol regime + sharp negative return in bear regime
        if (high_vol_regime and not bull_regime and
                return_5d is not None and return_5d < -0.03 and
                vol_ratio is not None and vol_ratio > 1.2):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.60,
                reason=f"High-vol expansion with negative 5d return ({return_5d*100:.1f}%)",
                pattern="vol_expansion_bear",
                features_snapshot=features,
            )

        logger.debug(f"[{self.NAME}] HOLD for {ticker}: no vol breakout pattern triggered")
        return RawSignal(
            direction="HOLD",
            strategy=self.NAME,
            strength=0.0,
            reason="No volatility breakout pattern triggered",
            pattern="hold",
            features_snapshot=features,
        )
