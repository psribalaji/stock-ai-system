"""
signals/strategies/trend_following.py — Trend-following signal strategy.

Detects BUY/SELL signals based on moving average alignment, golden/death cross,
ADX trend strength, and 52-week high proximity.
"""
from __future__ import annotations

from loguru import logger

from src.config import get_config
from src.signals.strategies.momentum import RawSignal


class TrendFollowingStrategy:
    """
    Trend-following strategy: buys into confirmed uptrends, sells into downtrends.

    BUY conditions (any triggers):
      - Golden cross (SMA50 crosses above SMA200) with volume
      - Full MA alignment (price > EMA21 > SMA50 > SMA200) + ADX > 25
      - Price near 52-week high with strong uptrend (ADX > 20)

    SELL conditions (any triggers):
      - Death cross (SMA50 crosses below SMA200)
      - Price below SMA50 with ADX > 25 and downtrend (–DI > +DI)
      - Trend reversal: price > SMA200 but ADX weakening with negative MACD hist
    """

    NAME = "trend_following"

    def __init__(self) -> None:
        self.config = get_config()

    def evaluate(self, features: dict, ticker: str = "") -> RawSignal:
        """
        Evaluate trend conditions on latest feature snapshot.

        Args:
            features: Dict from FeatureEngine.get_latest_features()
            ticker:   Ticker symbol (for logging)

        Returns:
            RawSignal with direction, strength, and reason.
        """
        golden_cross = features.get("golden_cross")
        death_cross  = features.get("death_cross")
        ma_alignment = features.get("ma_alignment")
        adx          = features.get("adx")
        dmi_pos      = features.get("dmi_pos")
        dmi_neg      = features.get("dmi_neg")
        near_52w_high = features.get("near_52w_high")
        dist_52w_high = features.get("dist_from_52w_high")
        high_volume  = features.get("high_volume")
        price_vs_sma50  = features.get("price_vs_sma50")
        price_vs_sma200 = features.get("price_vs_sma200")
        macd_hist    = features.get("macd_hist")
        bull_regime  = features.get("bull_regime")

        # ── BUY signals ──────────────────────────────────────────────

        # Pattern 1: Golden cross with volume confirmation
        if golden_cross and high_volume:
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.80,
                reason="Golden cross (SMA50 > SMA200) confirmed by high volume",
                pattern="golden_cross_high_volume",
                features_snapshot=features,
            )

        # Pattern 2: Golden cross (no volume requirement — still strong)
        if golden_cross:
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.70,
                reason="Golden cross (SMA50 > SMA200)",
                pattern="golden_cross",
                features_snapshot=features,
            )

        # Pattern 3: Full MA alignment + strong ADX trend
        if (ma_alignment is not None and ma_alignment == 3 and
                adx is not None and adx > 25):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.72,
                reason=f"Full MA alignment with strong ADX ({adx:.1f})",
                pattern="full_ma_alignment_adx",
                features_snapshot=features,
            )

        # Pattern 4: Near 52-week high + uptrend (ADX > 20)
        if (near_52w_high and
                adx is not None and adx > 20 and
                dmi_pos is not None and dmi_neg is not None and
                dmi_pos > dmi_neg and bull_regime):
            return RawSignal(
                direction="BUY",
                strategy=self.NAME,
                strength=0.65,
                reason=f"Near 52w high with uptrend ADX ({adx:.1f})",
                pattern="near_52w_high_uptrend",
                features_snapshot=features,
            )

        # ── SELL signals ──────────────────────────────────────────────

        # Pattern 5: Death cross
        if death_cross:
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.75,
                reason="Death cross (SMA50 < SMA200)",
                pattern="death_cross",
                features_snapshot=features,
            )

        # Pattern 6: Price below SMA50 + strong downtrend (ADX > 25, –DI > +DI)
        if (price_vs_sma50 is not None and price_vs_sma50 < 0 and
                adx is not None and adx > 25 and
                dmi_neg is not None and dmi_pos is not None and
                dmi_neg > dmi_pos):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.68,
                reason=f"Below SMA50 with strong downtrend ADX ({adx:.1f}), –DI dominant",
                pattern="below_sma50_downtrend",
                features_snapshot=features,
            )

        # Pattern 7: Above SMA200 but momentum weakening (MACD hist turning negative)
        if (price_vs_sma200 is not None and price_vs_sma200 > 0 and
                macd_hist is not None and macd_hist < 0 and
                adx is not None and adx < 20 and
                not bull_regime):
            return RawSignal(
                direction="SELL",
                strategy=self.NAME,
                strength=0.55,
                reason="Trend weakening: MACD negative, low ADX in non-bull regime",
                pattern="trend_weakening",
                features_snapshot=features,
            )

        logger.debug(f"[{self.NAME}] HOLD for {ticker}: no trend pattern triggered")
        return RawSignal(
            direction="HOLD",
            strategy=self.NAME,
            strength=0.0,
            reason="No trend pattern triggered",
            pattern="hold",
            features_snapshot=features,
        )
