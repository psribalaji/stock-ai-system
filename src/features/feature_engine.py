"""
feature_engine.py — Computes all technical indicators for signal generation.
Uses pandas-ta for indicators. All computation is deterministic and testable.
NEVER calls LLM here — LLM is only for enrichment in LLMAnalysisService.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from src.config import get_config


class FeatureEngine:
    """
    Computes technical features on OHLCV data.
    Input:  DataFrame with open, high, low, close, volume columns
    Output: Same DataFrame + computed indicator columns

    All indicators are appended as new columns — original OHLCV is preserved.
    """

    def __init__(self):
        self.config = get_config()
        self._check_pandas_ta()

    @staticmethod
    def _check_pandas_ta() -> None:
        try:
            import pandas_ta  # noqa
        except ImportError:
            raise ImportError("pandas-ta not installed. Run: pip install pandas-ta")

    # ── MAIN ENTRY POINT ─────────────────────────────────────────

    def compute_all(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """
        Compute the full feature set for a ticker.
        Returns input DataFrame with all indicator columns appended.

        Call order matters — some indicators depend on others.
        """
        if df.empty or len(df) < 50:
            logger.warning(
                f"Insufficient data for {ticker}: {len(df)} rows "
                f"(need at least 50)"
            )
            return df

        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        df = self._add_moving_averages(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume_features(df)
        df = self._add_trend_features(df)
        df = self._add_returns(df)
        df = self._add_regime(df)

        logger.debug(
            f"Features computed for {ticker}: "
            f"{len([c for c in df.columns if c not in ['timestamp','open','high','low','close','volume','vwap','ticker','source']])} indicators"
        )
        return df

    # ── MOVING AVERAGES ──────────────────────────────────────────

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.signals
        import pandas_ta as ta

        # Simple MAs
        df["sma_20"]  = ta.sma(df["close"], length=20)
        df["sma_50"]  = ta.sma(df["close"], length=cfg.ma_short)
        df["sma_200"] = ta.sma(df["close"], length=cfg.ma_long)

        # Exponential MAs
        df["ema_9"]   = ta.ema(df["close"], length=9)
        df["ema_21"]  = ta.ema(df["close"], length=21)
        df["ema_50"]  = ta.ema(df["close"], length=50)

        # Price position relative to MAs (key signal features)
        df["price_vs_sma50"]  = (df["close"] - df["sma_50"])  / df["sma_50"]
        df["price_vs_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]

        # Golden cross / death cross
        df["golden_cross"] = (
            (df["sma_50"] > df["sma_200"]) &
            (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
        ).astype(int)
        df["death_cross"] = (
            (df["sma_50"] < df["sma_200"]) &
            (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
        ).astype(int)

        # MA alignment score: +1 for each in order (price > ema21 > sma50 > sma200)
        df["ma_alignment"] = (
            (df["close"] > df["ema_21"]).astype(int) +
            (df["ema_21"] > df["sma_50"]).astype(int) +
            (df["sma_50"] > df["sma_200"]).astype(int)
        )

        return df

    # ── MOMENTUM ─────────────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.signals
        import pandas_ta as ta

        # RSI
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["rsi_7"]  = ta.rsi(df["close"], length=7)

        # RSI signals
        df["rsi_oversold"]   = (df["rsi_14"] < cfg.rsi_oversold).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > cfg.rsi_overbought).astype(int)

        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["macd"]        = macd.iloc[:, 0]   # MACD line
            df["macd_signal"] = macd.iloc[:, 2]   # Signal line
            df["macd_hist"]   = macd.iloc[:, 1]   # Histogram
            df["macd_cross_up"]   = (
                (df["macd"] > df["macd_signal"]) &
                (df["macd"].shift(1) <= df["macd_signal"].shift(1))
            ).astype(int)
            df["macd_cross_down"] = (
                (df["macd"] < df["macd_signal"]) &
                (df["macd"].shift(1) >= df["macd_signal"].shift(1))
            ).astype(int)

        # Stochastic
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

        # Rate of change
        df["roc_5"]  = ta.roc(df["close"], length=5)
        df["roc_20"] = ta.roc(df["close"], length=20)

        # Williams %R
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        return df

    # ── VOLATILITY ───────────────────────────────────────────────

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas_ta as ta

        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None and not bb.empty:
            df["bb_upper"]  = bb.iloc[:, 0]
            df["bb_mid"]    = bb.iloc[:, 1]
            df["bb_lower"]  = bb.iloc[:, 2]
            df["bb_width"]  = bb.iloc[:, 3]  # (upper - lower) / mid
            df["bb_pct"]    = bb.iloc[:, 4]  # position within bands

        # ATR — Average True Range (for stop-loss sizing)
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Historical volatility (annualized)
        df["hvol_20"] = (
            df["close"].pct_change().rolling(20).std() * np.sqrt(252)
        )
        df["hvol_60"] = (
            df["close"].pct_change().rolling(60).std() * np.sqrt(252)
        )

        # Volatility regime: is current vol above its 60d average?
        df["high_vol_regime"] = (
            df["hvol_20"] > df["hvol_20"].rolling(60).mean()
        ).astype(int)

        return df

    # ── VOLUME FEATURES ──────────────────────────────────────────

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.signals
        import pandas_ta as ta

        # Volume moving averages
        df["vol_sma_20"] = df["volume"].rolling(20).mean()
        df["vol_sma_5"]  = df["volume"].rolling(5).mean()

        # Volume ratio — how much above/below average is today's volume?
        df["vol_ratio"] = df["volume"] / df["vol_sma_20"]

        # High volume flag (signal requires this)
        df["high_volume"] = (
            df["vol_ratio"] >= cfg.min_volume_ratio
        ).astype(int)

        # On Balance Volume
        df["obv"] = ta.obv(df["close"], df["volume"])

        # OBV trend (positive = volume confirming price up)
        df["obv_trend"] = (df["obv"] > df["obv"].rolling(20).mean()).astype(int)

        # Volume Price Trend
        try:
            df["vpt"] = ta.vp(df["close"], df["volume"]) if hasattr(ta, "vp") else np.nan
        except Exception:
            df["vpt"] = np.nan

        # Accumulation/Distribution
        df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])

        return df

    # ── TREND FEATURES ───────────────────────────────────────────

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas_ta as ta

        # ADX — trend strength (>25 = strong trend, <20 = weak/sideways)
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None and not adx.empty:
            df["adx"]    = adx.iloc[:, 0]  # ADX
            df["dmi_pos"] = adx.iloc[:, 1]  # +DI
            df["dmi_neg"] = adx.iloc[:, 2]  # -DI

        # Trend direction: +1 (up), 0 (sideways), -1 (down)
        df["trend_direction"] = np.where(
            df["close"] > df["sma_50"], 1,
            np.where(df["close"] < df["sma_50"], -1, 0)
        )

        # Consecutive up/down days
        daily_change = df["close"].diff()
        df["consec_up"]   = daily_change.gt(0).astype(int).groupby(
            (daily_change <= 0).cumsum()
        ).cumsum()
        df["consec_down"] = daily_change.lt(0).astype(int).groupby(
            (daily_change >= 0).cumsum()
        ).cumsum()

        # 52-week high/low proximity
        df["dist_from_52w_high"] = (
            df["close"] / df["close"].rolling(252).max() - 1
        )
        df["dist_from_52w_low"] = (
            df["close"] / df["close"].rolling(252).min() - 1
        )

        # Near 52-week high (within 5%)
        df["near_52w_high"] = (df["dist_from_52w_high"] > -0.05).astype(int)

        return df

    # ── RETURNS ──────────────────────────────────────────────────

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Simple returns
        df["return_1d"]  = df["close"].pct_change(1)
        df["return_5d"]  = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        df["return_60d"] = df["close"].pct_change(60)

        # Log returns (better for modeling)
        df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))

        # Rolling Sharpe proxy (20-day)
        r = df["log_return_1d"]
        df["rolling_sharpe_20"] = (
            r.rolling(20).mean() / (r.rolling(20).std() + 1e-9) * np.sqrt(252)
        )

        return df

    # ── MARKET REGIME ────────────────────────────────────────────

    def _add_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify current market regime.
        Used to adjust signal confidence based on macro environment.
        """
        # Regime: 1 = bull (price > 200d MA + low vol), 0 = bear/uncertain
        df["bull_regime"] = (
            (df["close"] > df["sma_200"]) &
            (df["hvol_20"] < df["hvol_20"].rolling(60).mean())
        ).astype(int)

        # Composite regime score 0–5 (higher = more bullish conditions)
        df["regime_score"] = (
            (df["close"] > df["sma_50"]).astype(int) +
            (df["close"] > df["sma_200"]).astype(int) +
            (df["sma_50"] > df["sma_200"]).astype(int) +
            (df["rsi_14"] > 50).astype(int) +
            (df["obv_trend"] == 1).astype(int)
        )

        return df

    # ── FEATURE SUMMARY ──────────────────────────────────────────

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return list of all feature column names (for ML use)."""
        return [
            # Moving averages
            "price_vs_sma50", "price_vs_sma200", "ma_alignment",
            "golden_cross", "death_cross",
            # Momentum
            "rsi_14", "rsi_7", "rsi_oversold", "rsi_overbought",
            "macd_hist", "macd_cross_up", "macd_cross_down",
            "roc_5", "roc_20",
            # Volatility
            "bb_pct", "bb_width", "atr_14", "hvol_20", "high_vol_regime",
            # Volume
            "vol_ratio", "high_volume", "obv_trend",
            # Trend
            "adx", "trend_direction", "near_52w_high",
            "dist_from_52w_high",
            # Returns
            "return_1d", "return_5d", "return_20d", "rolling_sharpe_20",
            # Regime
            "bull_regime", "regime_score",
        ]

    def get_latest_features(self, df: pd.DataFrame) -> dict:
        """
        Extract the most recent row's features as a flat dict.
        Used by SignalDetector to evaluate the current state.
        """
        if df.empty:
            return {}
        last = df.iloc[-1]
        features = {}
        for col in self.get_feature_columns():
            if col in last.index:
                val = last[col]
                features[col] = float(val) if pd.notna(val) else None
        return features
