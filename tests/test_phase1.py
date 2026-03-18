"""
tests/test_phase1.py — Phase 1 unit tests.
Run with: pytest tests/test_phase1.py -v

All tests run WITHOUT API keys using synthetic data.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engine import FeatureEngine
from src.signals.strategies.momentum import MomentumStrategy, RawSignal
from src.signals.strategies.trend_following import TrendFollowingStrategy
from src.signals.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.signals.signal_detector import SignalDetector
from src.signals.confidence_scorer import ConfidenceScorer, ScoredSignal
from src.risk.risk_manager import RiskManager, PortfolioState
from src.execution.decision_engine import DecisionEngine, TradeDecision


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """200 rows of synthetic OHLCV with realistic price behavior."""
    n = 200
    np.random.seed(42)
    dates = pd.date_range(
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="B",
    )
    returns = np.random.normal(0.0005, 0.015, n)
    close   = 100.0 * np.cumprod(1 + returns)
    high    = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low     = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_   = close * (1 + np.random.normal(0, 0.005, n))
    volume  = np.random.randint(1_000_000, 10_000_000, n)

    return pd.DataFrame({
        "timestamp": dates,
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
        "ticker": "TEST",
        "source": "synthetic",
    })


@pytest.fixture
def feature_df(sample_ohlcv):
    """OHLCV DataFrame with all features computed."""
    engine = FeatureEngine()
    return engine.compute_all(sample_ohlcv, "TEST")


@pytest.fixture
def features(feature_df):
    """Latest feature snapshot as dict."""
    engine = FeatureEngine()
    return engine.get_latest_features(feature_df)


@pytest.fixture
def buy_features():
    """Synthetic features that should trigger BUY signals."""
    return {
        "rsi_14":          30.0,   # oversold
        "rsi_7":           28.0,
        "rsi_oversold":    1.0,
        "rsi_overbought":  0.0,
        "macd_hist":       0.5,
        "macd_cross_up":   1.0,
        "macd_cross_down": 0.0,
        "roc_5":           3.0,    # positive momentum
        "roc_20":          5.0,
        "high_volume":     1.0,
        "vol_ratio":       2.0,
        "price_vs_sma50":  0.05,   # above short MA
        "price_vs_sma200": 0.10,   # above long MA
        "ma_alignment":    3.0,    # full alignment
        "golden_cross":    1.0,
        "death_cross":     0.0,
        "adx":             30.0,   # strong trend
        "dmi_pos":         25.0,
        "dmi_neg":         15.0,
        "near_52w_high":   1.0,
        "dist_from_52w_high": -0.02,
        "bull_regime":     1.0,
        "regime_score":    4.0,
        "bb_pct":          1.1,    # above upper band
        "bb_width":        0.02,   # squeeze
        "atr_14":          2.0,
        "hvol_20":         0.18,
        "high_vol_regime": 1.0,
        "return_1d":       0.02,   # positive
        "return_5d":       0.05,
        "return_20d":      0.10,
        "rolling_sharpe_20": 1.2,
        "obv_trend":       1.0,
        "trend_direction": 1.0,
    }


@pytest.fixture
def sell_features():
    """Synthetic features that should trigger SELL signals."""
    return {
        "rsi_14":          75.0,   # overbought
        "rsi_7":           78.0,
        "rsi_oversold":    0.0,
        "rsi_overbought":  1.0,
        "macd_hist":       -0.5,
        "macd_cross_up":   0.0,
        "macd_cross_down": 1.0,
        "roc_5":           -3.0,
        "roc_20":          -5.0,
        "high_volume":     1.0,
        "vol_ratio":       2.0,
        "price_vs_sma50":  -0.05,  # below short MA
        "price_vs_sma200": 0.05,   # slightly above long MA
        "ma_alignment":    0.0,
        "golden_cross":    0.0,
        "death_cross":     1.0,
        "adx":             28.0,
        "dmi_pos":         12.0,
        "dmi_neg":         28.0,   # –DI dominant
        "near_52w_high":   0.0,
        "dist_from_52w_high": -0.20,
        "bull_regime":     0.0,
        "regime_score":    1.0,
        "bb_pct":          -0.1,   # below lower band
        "bb_width":        0.02,
        "atr_14":          2.5,
        "hvol_20":         0.25,
        "high_vol_regime": 1.0,
        "return_1d":       -0.02,
        "return_5d":       -0.05,
        "return_20d":      -0.08,
        "rolling_sharpe_20": -0.5,
        "obv_trend":       0.0,
        "trend_direction": -1.0,
    }


@pytest.fixture
def healthy_portfolio():
    """Portfolio state with no risk violations."""
    return PortfolioState(
        total_value_usd=100_000.0,
        cash_usd=50_000.0,
        open_positions=2,
        crypto_exposure_usd=5_000.0,
        daily_pnl_pct=-0.005,      # -0.5% (within -2% limit)
        peak_value_usd=102_000.0,  # small drawdown
    )


@pytest.fixture
def scored_buy(buy_features):
    """Pre-built ScoredSignal for a BUY."""
    return ScoredSignal(
        ticker="TEST",
        direction="BUY",
        strategy="momentum",
        pattern="rsi_oversold_macd_cross_up",
        strength=0.75,
        confidence=0.72,
        base_confidence=0.62,
        regime_multiplier=1.0,
        volume_multiplier=1.05,
        reason="RSI oversold recovery with MACD cross up",
        blocked=False,
        block_reason="",
        features_snapshot=buy_features,
    )


# ── MomentumStrategy Tests ────────────────────────────────────────────────────

class TestMomentumStrategy:

    @pytest.fixture
    def strategy(self):
        return MomentumStrategy()

    def test_buy_rsi_oversold_macd_cross_up(self, strategy, buy_features):
        """RSI < 35 + MACD cross up → BUY."""
        signal = strategy.evaluate(buy_features, "TEST")
        assert signal.direction == "BUY"
        assert signal.strategy == "momentum"
        assert signal.strength > 0

    def test_sell_rsi_overbought_macd_cross_down(self, strategy, sell_features):
        """RSI > 70 + MACD cross down → SELL."""
        signal = strategy.evaluate(sell_features, "TEST")
        assert signal.direction == "SELL"
        assert signal.strategy == "momentum"

    def test_hold_neutral_features(self, strategy):
        """Neutral features → HOLD."""
        neutral = {
            "rsi_14": 50.0,
            "rsi_oversold": 0.0,
            "rsi_overbought": 0.0,
            "macd_hist": 0.01,
            "macd_cross_up": 0.0,
            "macd_cross_down": 0.0,
            "roc_5": 0.5,
            "roc_20": 1.0,
            "high_volume": 0.0,
            "price_vs_sma50": 0.01,
            "bull_regime": 1.0,
        }
        signal = strategy.evaluate(neutral, "TEST")
        assert signal.direction == "HOLD"
        assert signal.strength == 0.0

    def test_returns_raw_signal_type(self, strategy, buy_features):
        """evaluate() must return a RawSignal."""
        signal = strategy.evaluate(buy_features)
        assert isinstance(signal, RawSignal)

    def test_direction_is_valid(self, strategy, features):
        """Direction must be BUY, SELL, or HOLD."""
        signal = strategy.evaluate(features)
        assert signal.direction in {"BUY", "SELL", "HOLD"}

    def test_strength_range(self, strategy, buy_features):
        """Strength must be in [0.0, 1.0]."""
        signal = strategy.evaluate(buy_features)
        assert 0.0 <= signal.strength <= 1.0

    def test_reason_not_empty(self, strategy, buy_features):
        """Reason should always be populated."""
        signal = strategy.evaluate(buy_features)
        assert len(signal.reason) > 0

    def test_empty_features_returns_hold(self, strategy):
        """Empty features dict → HOLD without crashing."""
        signal = strategy.evaluate({})
        assert signal.direction == "HOLD"


# ── TrendFollowingStrategy Tests ──────────────────────────────────────────────

class TestTrendFollowingStrategy:

    @pytest.fixture
    def strategy(self):
        return TrendFollowingStrategy()

    def test_buy_golden_cross_high_volume(self, strategy, buy_features):
        """Golden cross + high volume → BUY."""
        signal = strategy.evaluate(buy_features, "TEST")
        assert signal.direction == "BUY"
        assert signal.strategy == "trend_following"

    def test_sell_death_cross(self, strategy, sell_features):
        """Death cross → SELL."""
        signal = strategy.evaluate(sell_features, "TEST")
        assert signal.direction == "SELL"

    def test_full_ma_alignment_adx_buy(self, strategy, buy_features):
        """Full MA alignment (3) + ADX > 25 → BUY."""
        feats = {**buy_features, "golden_cross": 0.0, "ma_alignment": 3.0, "adx": 30.0}
        signal = strategy.evaluate(feats, "TEST")
        assert signal.direction == "BUY"

    def test_hold_when_no_pattern(self, strategy):
        """No pattern match → HOLD."""
        neutral = {
            "golden_cross": 0.0, "death_cross": 0.0,
            "ma_alignment": 1.0, "adx": 15.0,
            "near_52w_high": 0.0, "bull_regime": 0.0,
            "price_vs_sma50": 0.01, "price_vs_sma200": 0.01,
            "dmi_pos": 15.0, "dmi_neg": 15.0,
            "macd_hist": 0.0, "high_volume": 0.0,
        }
        signal = strategy.evaluate(neutral, "TEST")
        assert signal.direction == "HOLD"

    def test_returns_raw_signal(self, strategy, buy_features):
        assert isinstance(strategy.evaluate(buy_features), RawSignal)

    def test_direction_valid(self, strategy, features):
        signal = strategy.evaluate(features)
        assert signal.direction in {"BUY", "SELL", "HOLD"}


# ── VolatilityBreakoutStrategy Tests ─────────────────────────────────────────

class TestVolatilityBreakoutStrategy:

    @pytest.fixture
    def strategy(self):
        return VolatilityBreakoutStrategy()

    def test_buy_bb_squeeze_breakout(self, strategy, buy_features):
        """BB squeeze + bb_pct > 1 + high volume → BUY."""
        signal = strategy.evaluate(buy_features, "TEST")
        assert signal.direction == "BUY"
        assert signal.strategy == "volatility_breakout"

    def test_sell_bb_squeeze_breakdown(self, strategy, sell_features):
        """BB squeeze + bb_pct < 0 + high volume → SELL."""
        signal = strategy.evaluate(sell_features, "TEST")
        assert signal.direction == "SELL"

    def test_hold_neutral_vol(self, strategy):
        """Mid-band, normal volume → HOLD."""
        neutral = {
            "bb_pct": 0.5,        # mid-band
            "bb_width": 0.05,     # not squeeze
            "high_volume": 0.0,
            "return_1d": 0.001,
            "return_5d": 0.005,
            "bull_regime": 1.0,
            "high_vol_regime": 0.0,
            "atr_14": 1.5,
            "hvol_20": 0.15,
            "vol_ratio": 0.9,
        }
        signal = strategy.evaluate(neutral, "TEST")
        assert signal.direction == "HOLD"

    def test_returns_raw_signal(self, strategy, buy_features):
        assert isinstance(strategy.evaluate(buy_features), RawSignal)

    def test_direction_valid(self, strategy, features):
        signal = strategy.evaluate(features)
        assert signal.direction in {"BUY", "SELL", "HOLD"}


# ── SignalDetector Tests ──────────────────────────────────────────────────────

class TestSignalDetector:

    @pytest.fixture
    def detector(self):
        return SignalDetector()

    def test_detect_returns_list(self, detector, sample_ohlcv):
        """detect() should return a list."""
        result = detector.detect("TEST", sample_ohlcv)
        assert isinstance(result, list)

    def test_detect_one_signal_per_strategy(self, detector, sample_ohlcv):
        """Should return exactly 3 signals (one per strategy)."""
        result = detector.detect("TEST", sample_ohlcv)
        assert len(result) == 3

    def test_detect_signal_directions_valid(self, detector, sample_ohlcv):
        """All signal directions must be valid."""
        signals = detector.detect("TEST", sample_ohlcv)
        for s in signals:
            assert s.direction in {"BUY", "SELL", "HOLD"}

    def test_detect_empty_df_returns_empty(self, detector):
        """Empty DataFrame → empty list."""
        result = detector.detect("TEST", pd.DataFrame())
        assert result == []

    def test_detect_small_df_returns_empty(self, detector, sample_ohlcv):
        """Less than 50 rows → empty list."""
        result = detector.detect("TEST", sample_ohlcv.head(30))
        assert result == []

    def test_detect_actionable_no_hold(self, detector, sample_ohlcv):
        """detect_actionable() must not return HOLD signals."""
        result = detector.detect_actionable("TEST", sample_ohlcv)
        for s in result:
            assert s.direction != "HOLD"

    def test_detect_computes_features_if_missing(self, detector, sample_ohlcv):
        """If rsi_14 is absent, detect() should compute features automatically."""
        assert "rsi_14" not in sample_ohlcv.columns
        result = detector.detect("TEST", sample_ohlcv)
        assert isinstance(result, list)

    def test_signals_to_dataframe(self, detector, sample_ohlcv):
        """signals_to_dataframe should produce a non-empty DataFrame."""
        signals = detector.detect("TEST", sample_ohlcv)
        df = SignalDetector.signals_to_dataframe(signals, "TEST")
        assert isinstance(df, pd.DataFrame)
        assert "direction" in df.columns
        assert "strategy" in df.columns

    def test_strategies_have_unique_names(self, detector):
        """Each strategy should have a distinct name."""
        names = [s.NAME for s in detector.strategies]
        assert len(names) == len(set(names))


# ── ConfidenceScorer Tests ────────────────────────────────────────────────────

class TestConfidenceScorer:

    @pytest.fixture
    def scorer(self):
        return ConfidenceScorer()

    @pytest.fixture
    def raw_buy(self, buy_features):
        return RawSignal(
            direction="BUY",
            strategy="momentum",
            strength=0.75,
            reason="test signal",
            pattern="rsi_oversold_macd_cross_up",
            features_snapshot=buy_features,
        )

    def test_score_returns_scored_signal(self, scorer, raw_buy):
        """score() must return a ScoredSignal."""
        result = scorer.score(raw_buy, "TEST")
        assert isinstance(result, ScoredSignal)

    def test_confidence_in_range(self, scorer, raw_buy):
        """Confidence must be in [0.0, 1.0]."""
        result = scorer.score(raw_buy, "TEST")
        assert 0.0 <= result.confidence <= 1.0

    def test_bull_regime_multiplier(self, scorer, raw_buy, buy_features):
        """Bull regime should give multiplier = 1.0."""
        result = scorer.score(raw_buy, "TEST", features=buy_features)
        assert result.regime_multiplier == 1.0

    def test_bear_regime_multiplier(self, scorer, raw_buy, buy_features):
        """Non-bull regime should give multiplier = 0.85."""
        bear_feats = {**buy_features, "bull_regime": 0.0}
        result = scorer.score(raw_buy, "TEST", features=bear_feats)
        assert result.regime_multiplier == 0.85

    def test_high_volume_multiplier(self, scorer, raw_buy, buy_features):
        """High volume → multiplier 1.05."""
        result = scorer.score(raw_buy, "TEST", features=buy_features)
        assert result.volume_multiplier == 1.05

    def test_low_volume_multiplier(self, scorer, raw_buy, buy_features):
        """Low volume → multiplier 0.95."""
        low_vol = {**buy_features, "high_volume": 0.0}
        result = scorer.score(raw_buy, "TEST", features=low_vol)
        assert result.volume_multiplier == 0.95

    def test_blocked_below_min_confidence(self, scorer, buy_features):
        """Low base win rate → signal blocked."""
        low_signal = RawSignal(
            direction="BUY",
            strategy="momentum",
            strength=0.3,
            reason="weak signal",
            pattern="hold",  # no default → DEFAULT_WIN_RATE=0.52
            features_snapshot={**buy_features, "bull_regime": 0.0, "high_volume": 0.0},
        )
        result = scorer.score(low_signal, "TEST")
        # 0.52 * 0.85 * 0.95 ≈ 0.42 → blocked
        assert result.blocked is True
        assert len(result.block_reason) > 0

    def test_score_all_skips_hold(self, scorer, buy_features):
        """score_all() should skip HOLD signals."""
        hold_signal = RawSignal(
            direction="HOLD", strategy="momentum", strength=0.0,
            reason="hold", pattern="hold", features_snapshot=buy_features,
        )
        buy_signal = RawSignal(
            direction="BUY", strategy="trend_following", strength=0.7,
            reason="test", pattern="golden_cross", features_snapshot=buy_features,
        )
        results = scorer.score_all([hold_signal, buy_signal], "TEST")
        strategies = [r.strategy for r in results]
        assert "momentum" not in strategies  # HOLD skipped

    def test_scored_signals_to_dataframe(self, scorer, raw_buy):
        """scored_signals_to_dataframe should return a DataFrame."""
        scored = scorer.score(raw_buy, "TEST")
        df = ConfidenceScorer.scored_signals_to_dataframe([scored])
        assert isinstance(df, pd.DataFrame)
        assert "confidence" in df.columns
        assert "blocked" in df.columns

    def test_known_pattern_gets_seed_rate(self, scorer, raw_buy):
        """Known pattern should get seed default (not global 0.52)."""
        result = scorer.score(raw_buy, "TEST")
        assert result.base_confidence == ConfidenceScorer.PATTERN_DEFAULTS["rsi_oversold_macd_cross_up"]

    def test_unknown_pattern_gets_default(self, scorer, buy_features):
        """Unknown pattern → DEFAULT_WIN_RATE."""
        unknown = RawSignal(
            direction="BUY", strategy="test", strength=0.5,
            reason="test", pattern="unknown_pattern_xyz",
            features_snapshot=buy_features,
        )
        result = scorer.score(unknown, "TEST")
        assert result.base_confidence == ConfidenceScorer.DEFAULT_WIN_RATE


# ── RiskManager Tests ─────────────────────────────────────────────────────────

class TestRiskManager:

    @pytest.fixture
    def rm(self):
        return RiskManager()

    def test_approve_healthy_signal(self, rm, scored_buy, healthy_portfolio):
        """Healthy signal + healthy portfolio → approved."""
        decision = rm.validate(scored_buy, 100.0, healthy_portfolio, "TEST")
        assert decision.approved is True
        assert decision.block_reason == ""

    def test_position_size_pct(self, rm, scored_buy, healthy_portfolio):
        """Position size should be at most max_position_pct."""
        decision = rm.validate(scored_buy, 100.0, healthy_portfolio, "TEST")
        assert decision.position_size_pct <= 0.05

    def test_stop_loss_calculated(self, rm, scored_buy, healthy_portfolio):
        """Stop loss should be 7% below entry."""
        entry = 100.0
        decision = rm.validate(scored_buy, entry, healthy_portfolio, "TEST")
        expected_stop = entry * (1 - 0.07)
        assert abs(decision.stop_loss_price - expected_stop) < 0.01

    def test_block_low_confidence(self, rm, healthy_portfolio):
        """Signal with confidence below 0.60 → blocked."""
        low_conf = ScoredSignal(
            ticker="TEST", direction="BUY", strategy="momentum",
            pattern="test", strength=0.5, confidence=0.45,
            base_confidence=0.45, regime_multiplier=1.0, volume_multiplier=1.0,
            reason="weak", blocked=False, block_reason="",
            features_snapshot={},
        )
        decision = rm.validate(low_conf, 100.0, healthy_portfolio, "TEST")
        assert decision.approved is False
        assert "confidence" in decision.block_reason.lower()

    def test_block_max_positions_reached(self, rm, scored_buy):
        """Max positions (5) → blocked."""
        full_portfolio = PortfolioState(
            total_value_usd=100_000.0,
            cash_usd=50_000.0,
            open_positions=5,           # at max
            crypto_exposure_usd=0.0,
            daily_pnl_pct=-0.005,
            peak_value_usd=105_000.0,
        )
        decision = rm.validate(scored_buy, 100.0, full_portfolio, "TEST")
        assert decision.approved is False
        assert "positions" in decision.block_reason.lower()

    def test_circuit_breaker_daily_loss(self, rm, scored_buy):
        """Daily loss ≥ 2% → circuit breaker triggers."""
        bad_day = PortfolioState(
            total_value_usd=98_000.0,
            cash_usd=50_000.0,
            open_positions=1,
            crypto_exposure_usd=0.0,
            daily_pnl_pct=-0.022,       # worse than -2%
            peak_value_usd=100_000.0,
        )
        decision = rm.validate(scored_buy, 100.0, bad_day, "TEST")
        assert decision.approved is False
        assert rm.is_paused is True

    def test_circuit_breaker_persists(self, rm, scored_buy, healthy_portfolio):
        """Once paused, all subsequent signals blocked."""
        rm._paused = True
        decision = rm.validate(scored_buy, 100.0, healthy_portfolio, "TEST")
        assert decision.approved is False

    def test_reset_daily_clears_pause(self, rm):
        """reset_daily() should clear the circuit breaker."""
        rm._paused = True
        rm.reset_daily()
        assert rm.is_paused is False

    def test_kill_switch_max_drawdown(self, rm, scored_buy):
        """Max drawdown ≥ 15% → kill switch."""
        crashed = PortfolioState(
            total_value_usd=82_000.0,   # 18% down from 100k peak
            cash_usd=40_000.0,
            open_positions=1,
            crypto_exposure_usd=0.0,
            daily_pnl_pct=-0.005,
            peak_value_usd=100_000.0,
        )
        decision = rm.validate(scored_buy, 100.0, crashed, "TEST")
        assert decision.approved is False
        assert rm.is_killed is True

    def test_block_crypto_over_limit(self, rm, healthy_portfolio):
        """Crypto exposure ≥ 10% → blocked for crypto tickers."""
        crypto_maxed = PortfolioState(
            total_value_usd=100_000.0,
            cash_usd=50_000.0,
            open_positions=2,
            crypto_exposure_usd=10_500.0,   # > 10%
            daily_pnl_pct=-0.005,
            peak_value_usd=102_000.0,
        )
        btc_signal = ScoredSignal(
            ticker="BTC", direction="BUY", strategy="momentum",
            pattern="golden_cross", strength=0.7, confidence=0.72,
            base_confidence=0.65, regime_multiplier=1.0, volume_multiplier=1.05,
            reason="test", blocked=False, block_reason="",
            features_snapshot={"bull_regime": 1.0, "high_volume": 1.0},
        )
        decision = rm.validate(btc_signal, 50_000.0, crypto_maxed, "BTC")
        assert decision.approved is False
        assert "crypto" in decision.block_reason.lower()

    def test_position_capped_at_cash(self, rm, scored_buy):
        """Position size should not exceed available cash."""
        low_cash = PortfolioState(
            total_value_usd=100_000.0,
            cash_usd=2_000.0,           # less than 5% of 100k = 5000
            open_positions=1,
            crypto_exposure_usd=0.0,
            daily_pnl_pct=0.001,
            peak_value_usd=100_000.0,
        )
        decision = rm.validate(scored_buy, 100.0, low_cash, "TEST")
        assert decision.approved is True
        assert decision.position_size_usd <= 2_000.0

    def test_block_no_cash(self, rm, scored_buy):
        """No cash → blocked."""
        no_cash = PortfolioState(
            total_value_usd=100_000.0,
            cash_usd=0.0,
            open_positions=1,
            crypto_exposure_usd=0.0,
            daily_pnl_pct=0.001,
            peak_value_usd=100_000.0,
        )
        decision = rm.validate(scored_buy, 100.0, no_cash, "TEST")
        assert decision.approved is False

    def test_get_status(self, rm):
        """get_status() should return a dict with expected keys."""
        status = rm.get_status()
        assert "paused" in status
        assert "killed" in status
        assert "max_position_pct" in status


# ── DecisionEngine Tests ──────────────────────────────────────────────────────

class TestDecisionEngine:

    @pytest.fixture
    def engine(self):
        """DecisionEngine without LLM (no API key needed)."""
        return DecisionEngine(llm_service=None)

    def test_decide_returns_list(self, engine, sample_ohlcv, healthy_portfolio):
        """decide() should return a list."""
        result = engine.decide("TEST", sample_ohlcv, 120.0, healthy_portfolio)
        assert isinstance(result, list)

    def test_decide_empty_df_returns_empty(self, engine, healthy_portfolio):
        """Empty DataFrame → empty list."""
        result = engine.decide("TEST", pd.DataFrame(), 100.0, healthy_portfolio)
        assert result == []

    def test_decide_all_approved_are_trade_decisions(self, engine, sample_ohlcv, healthy_portfolio):
        """All returned decisions should be TradeDecision instances."""
        decisions = engine.decide("TEST", sample_ohlcv, 120.0, healthy_portfolio)
        for d in decisions:
            assert isinstance(d, TradeDecision)

    def test_approved_decisions_have_stop_loss(self, engine, sample_ohlcv, healthy_portfolio):
        """Every approved decision must have a stop loss price set."""
        decisions = engine.decide("TEST", sample_ohlcv, 120.0, healthy_portfolio)
        for d in decisions:
            assert d.stop_loss_price > 0

    def test_approved_decisions_have_position_size(self, engine, sample_ohlcv, healthy_portfolio):
        """Every approved decision must have a position size > 0."""
        decisions = engine.decide("TEST", sample_ohlcv, 120.0, healthy_portfolio)
        for d in decisions:
            assert d.position_size_usd > 0
            assert 0 < d.position_size_pct <= 0.05

    def test_approved_decisions_have_confidence(self, engine, sample_ohlcv, healthy_portfolio):
        """Approved decisions must have confidence >= 0.60."""
        decisions = engine.decide("TEST", sample_ohlcv, 120.0, healthy_portfolio)
        for d in decisions:
            assert d.confidence >= 0.60

    def test_blocked_portfolio_yields_no_decisions(self, engine, sample_ohlcv):
        """Portfolio at kill switch drawdown → no approvals."""
        crashed = PortfolioState(
            total_value_usd=80_000.0,
            cash_usd=40_000.0,
            open_positions=1,
            crypto_exposure_usd=0.0,
            daily_pnl_pct=-0.01,
            peak_value_usd=100_000.0,   # 20% drawdown
        )
        decisions = engine.decide("TEST", sample_ohlcv, 120.0, crashed)
        assert decisions == []

    def test_decisions_to_dataframe(self, engine, sample_ohlcv, healthy_portfolio):
        """decisions_to_dataframe() should return a DataFrame."""
        decisions = engine.decide("TEST", sample_ohlcv, 120.0, healthy_portfolio)
        df = DecisionEngine.decisions_to_dataframe(decisions)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "ticker" in df.columns
            assert "direction" in df.columns
            assert "confidence" in df.columns

    def test_decide_all_multiple_tickers(self, engine, sample_ohlcv, healthy_portfolio):
        """decide_all() should handle multiple tickers without crashing."""
        data_map  = {"A": sample_ohlcv.copy(), "B": sample_ohlcv.copy()}
        price_map = {"A": 100.0, "B": 200.0}
        results = engine.decide_all(["A", "B"], data_map, price_map, healthy_portfolio)
        assert isinstance(results, list)

    def test_decide_all_skips_missing_tickers(self, engine, sample_ohlcv, healthy_portfolio):
        """Tickers with no data in data_map are skipped gracefully."""
        data_map  = {"TEST": sample_ohlcv}
        price_map = {"TEST": 100.0}
        results = engine.decide_all(["TEST", "MISSING"], data_map, price_map, healthy_portfolio)
        assert isinstance(results, list)

    def test_timestamp_is_utc(self, engine, sample_ohlcv, healthy_portfolio):
        """TradeDecision timestamps must be UTC-aware."""
        decisions = engine.decide("TEST", sample_ohlcv, 120.0, healthy_portfolio)
        for d in decisions:
            assert d.timestamp.tzinfo is not None


# ── LLMAnalysisService Tests (no API calls) ───────────────────────────────────

class TestLLMAnalysisService:

    def test_import_without_api_key(self):
        """Module should import without needing ANTHROPIC_API_KEY."""
        from src.llm.llm_analysis_service import LLMAnalysisService
        svc = LLMAnalysisService()
        assert svc is not None

    def test_fallback_reasoning_structure(self, scored_buy):
        """_fallback_reasoning should return dict with required keys."""
        from src.llm.llm_analysis_service import LLMAnalysisService
        svc = LLMAnalysisService()
        result = svc._fallback_reasoning(scored_buy)
        assert "reasoning" in result
        assert "summary" in result
        assert "llm_used" in result
        assert result["llm_used"] is False

    def test_enrich_falls_back_gracefully(self, scored_buy):
        """enrich() with no API key → falls back to fallback_reasoning."""
        from src.llm.llm_analysis_service import LLMAnalysisService
        svc = LLMAnalysisService()
        # Mock _call_llm to raise an exception (simulating missing key)
        svc._call_llm = MagicMock(side_effect=RuntimeError("No API key"))
        result = svc.enrich(scored_buy, news_summary="Test news")
        assert isinstance(result, dict)
        assert "reasoning" in result
        assert "llm_used" in result

    def test_build_prompt_contains_ticker(self, scored_buy):
        """Prompt should contain the ticker name."""
        from src.llm.llm_analysis_service import LLMAnalysisService
        prompt = LLMAnalysisService._build_prompt(scored_buy, None)
        assert "TEST" in prompt

    def test_build_prompt_contains_direction(self, scored_buy):
        """Prompt should contain the signal direction."""
        from src.llm.llm_analysis_service import LLMAnalysisService
        prompt = LLMAnalysisService._build_prompt(scored_buy, None)
        assert "BUY" in prompt

    def test_build_prompt_with_news(self, scored_buy):
        """Prompt with news should include news text."""
        from src.llm.llm_analysis_service import LLMAnalysisService
        prompt = LLMAnalysisService._build_prompt(scored_buy, "Strong earnings report")
        assert "Strong earnings report" in prompt


# ── Integration: Full pipeline smoke test ─────────────────────────────────────

class TestFullPipelineIntegration:

    def test_pipeline_runs_end_to_end(self, sample_ohlcv, healthy_portfolio):
        """
        Smoke test: run the full pipeline from OHLCV → decisions.
        No mocks — uses real feature computation and strategy evaluation.
        """
        engine = DecisionEngine(llm_service=None)
        decisions = engine.decide(
            ticker="NVDA",
            df=sample_ohlcv,
            entry_price=120.0,
            portfolio=healthy_portfolio,
        )
        # Should not crash; decisions may be empty (no pattern triggered)
        assert isinstance(decisions, list)
        for d in decisions:
            assert d.ticker == "NVDA"
            assert d.direction in {"BUY", "SELL"}
            assert d.confidence >= 0.60
            assert d.position_size_usd > 0
            assert d.stop_loss_price > 0

    def test_signal_detector_feeds_scorer_feeds_risk(
        self, sample_ohlcv, healthy_portfolio
    ):
        """Each pipeline stage produces valid output for the next stage."""
        # Stage 1: Detect
        detector = SignalDetector()
        raw_signals = detector.detect("TEST", sample_ohlcv)
        assert isinstance(raw_signals, list)
        assert len(raw_signals) == 3  # one per strategy

        # Stage 2: Score
        scorer = ConfidenceScorer()
        scored = scorer.score_all(raw_signals, "TEST")
        assert isinstance(scored, list)
        for s in scored:
            assert 0.0 <= s.confidence <= 1.0

        # Stage 3: Risk
        rm = RiskManager()
        for s in scored:
            if not s.blocked:
                decision = rm.validate(s, 100.0, healthy_portfolio, "TEST")
                assert isinstance(decision.approved, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
