"""
tests/test_phase0.py — Phase 0 unit tests.
Run with: pytest tests/test_phase0.py -v

These tests run WITHOUT API keys using synthetic data.
All tests should pass before you run scripts/setup.py.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.storage import ParquetStore
from src.features.feature_engine import FeatureEngine
from src.config import get_config


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    """ParquetStore backed by a temp directory — no real files touched."""
    return ParquetStore(base_path=str(tmp_path))


@pytest.fixture
def sample_ohlcv():
    """200 rows of synthetic OHLCV data with realistic price behavior."""
    n = 200
    np.random.seed(42)
    dates = pd.date_range(
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="B",  # Business days
    )
    # Generate a realistic random walk
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


# ── ParquetStore Tests ───────────────────────────────────────────

class TestParquetStore:

    def test_save_and_load_ohlcv(self, tmp_store, sample_ohlcv):
        """Round-trip: save OHLCV then load it back."""
        tmp_store.save_ohlcv("TEST", sample_ohlcv)
        loaded = tmp_store.load_ohlcv("TEST")
        assert len(loaded) == len(sample_ohlcv)
        assert set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(
            loaded.columns
        )

    def test_save_merges_duplicates(self, tmp_store, sample_ohlcv):
        """Saving the same data twice should not double the row count."""
        tmp_store.save_ohlcv("TEST", sample_ohlcv)
        tmp_store.save_ohlcv("TEST", sample_ohlcv)
        loaded = tmp_store.load_ohlcv("TEST")
        assert len(loaded) == len(sample_ohlcv), "Duplicate rows found after double save"

    def test_save_appends_new_rows(self, tmp_store, sample_ohlcv):
        """Saving data then saving newer data should give combined rows."""
        first_half  = sample_ohlcv.iloc[:100]
        second_half = sample_ohlcv.iloc[100:]
        tmp_store.save_ohlcv("TEST", first_half)
        tmp_store.save_ohlcv("TEST", second_half)
        loaded = tmp_store.load_ohlcv("TEST")
        assert len(loaded) == len(sample_ohlcv)

    def test_load_date_filter(self, tmp_store, sample_ohlcv):
        """Date filter should return only rows in range."""
        tmp_store.save_ohlcv("TEST", sample_ohlcv)
        start = date(2023, 3, 1)
        end   = date(2023, 6, 1)
        loaded = tmp_store.load_ohlcv("TEST", start=start, end=end)
        dates = pd.to_datetime(loaded["timestamp"]).dt.date
        assert dates.min() >= start
        assert dates.max() <= end

    def test_load_missing_ticker_returns_empty(self, tmp_store):
        """Loading a ticker with no data returns an empty DataFrame."""
        df = tmp_store.load_ohlcv("DOESNOTEXIST")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_validate_clean_data(self, tmp_store, sample_ohlcv):
        """Clean synthetic data should pass validation."""
        report = tmp_store.validate_ohlcv(sample_ohlcv, "TEST")
        assert report["valid"] is True
        assert len(report["issues"]) == 0

    def test_validate_detects_nulls(self, tmp_store, sample_ohlcv):
        """Null values should be flagged in validation."""
        dirty = sample_ohlcv.copy()
        dirty.loc[5, "close"] = None
        report = tmp_store.validate_ohlcv(dirty, "TEST")
        assert not report["valid"]
        assert any("NULL" in issue for issue in report["issues"])

    def test_validate_detects_zero_prices(self, tmp_store, sample_ohlcv):
        """Zero prices should be flagged."""
        dirty = sample_ohlcv.copy()
        dirty.loc[10, "close"] = 0.0
        report = tmp_store.validate_ohlcv(dirty, "TEST")
        assert not report["valid"]
        assert any("ZERO" in issue for issue in report["issues"])

    def test_save_and_load_signals(self, tmp_store):
        """Round-trip test for signal storage."""
        signals = pd.DataFrame({
            "timestamp":  [pd.Timestamp.now(tz="UTC")],
            "ticker":     ["NVDA"],
            "direction":  ["BUY"],
            "confidence": [0.78],
            "strategy":   ["momentum"],
        })
        today = date.today()
        tmp_store.save_signals(signals, today)
        loaded = tmp_store.load_signals(start=today, end=today)
        assert len(loaded) == 1
        assert loaded.iloc[0]["ticker"] == "NVDA"

    def test_save_and_load_audit(self, tmp_store):
        """Round-trip test for audit log."""
        audit = pd.DataFrame({
            "trade_id":            ["trade-001"],
            "ticker":              ["NVDA"],
            "side":                ["BUY"],
            "qty":                 [10.0],
            "fill_price":          [185.50],
            "timestamp_submitted": [pd.Timestamp.now(tz="UTC")],
            "mode":                ["paper"],
        })
        tmp_store.save_audit(audit)
        loaded = tmp_store.load_audit()
        assert len(loaded) == 1
        assert loaded.iloc[0]["trade_id"] == "trade-001"

    def test_get_stats(self, tmp_store, sample_ohlcv):
        """Stats should reflect stored data."""
        tmp_store.save_ohlcv("TEST", sample_ohlcv)
        stats = tmp_store.get_stats()
        assert "TEST" in stats["tickers_stored"]
        assert stats["total_ohlcv_rows"] == len(sample_ohlcv)


# ── FeatureEngine Tests ──────────────────────────────────────────

class TestFeatureEngine:

    @pytest.fixture
    def engine(self):
        return FeatureEngine()

    def test_compute_all_adds_features(self, engine, sample_ohlcv):
        """compute_all should add indicator columns to the DataFrame."""
        original_cols = set(sample_ohlcv.columns)
        result = engine.compute_all(sample_ohlcv, "TEST")
        new_cols = set(result.columns) - original_cols
        assert len(new_cols) > 20, f"Expected 20+ new columns, got {len(new_cols)}"

    def test_moving_averages_present(self, engine, sample_ohlcv):
        result = engine.compute_all(sample_ohlcv, "TEST")
        for col in ["sma_20", "sma_50", "sma_200", "ema_9", "ema_21"]:
            assert col in result.columns, f"Missing: {col}"

    def test_momentum_indicators_present(self, engine, sample_ohlcv):
        result = engine.compute_all(sample_ohlcv, "TEST")
        for col in ["rsi_14", "macd_hist", "roc_5", "roc_20"]:
            assert col in result.columns, f"Missing: {col}"

    def test_volatility_indicators_present(self, engine, sample_ohlcv):
        result = engine.compute_all(sample_ohlcv, "TEST")
        for col in ["bb_upper", "bb_lower", "atr_14", "hvol_20"]:
            assert col in result.columns, f"Missing: {col}"

    def test_volume_features_present(self, engine, sample_ohlcv):
        result = engine.compute_all(sample_ohlcv, "TEST")
        for col in ["vol_ratio", "high_volume", "obv"]:
            assert col in result.columns, f"Missing: {col}"

    def test_regime_score_range(self, engine, sample_ohlcv):
        """Regime score should be between 0 and 5."""
        result = engine.compute_all(sample_ohlcv, "TEST")
        assert "regime_score" in result.columns
        valid = result["regime_score"].dropna()
        assert valid.between(0, 5).all(), "Regime score out of 0–5 range"

    def test_rsi_range(self, engine, sample_ohlcv):
        """RSI should always be between 0 and 100."""
        result = engine.compute_all(sample_ohlcv, "TEST")
        rsi = result["rsi_14"].dropna()
        assert rsi.between(0, 100).all(), "RSI out of 0–100 range"

    def test_ma_alignment_range(self, engine, sample_ohlcv):
        """MA alignment score should be 0, 1, 2, or 3."""
        result = engine.compute_all(sample_ohlcv, "TEST")
        vals = result["ma_alignment"].dropna()
        assert vals.isin([0, 1, 2, 3]).all()

    def test_empty_dataframe_handled(self, engine):
        """Empty DataFrame should return empty without crashing."""
        result = engine.compute_all(pd.DataFrame(), "TEST")
        assert result.empty

    def test_insufficient_data_handled(self, engine, sample_ohlcv):
        """Less than 50 rows should return original DataFrame."""
        small = sample_ohlcv.head(30)
        result = engine.compute_all(small, "TEST")
        assert len(result) == 30

    def test_get_latest_features_returns_dict(self, engine, sample_ohlcv):
        """get_latest_features should return a non-empty dict."""
        enriched = engine.compute_all(sample_ohlcv, "TEST")
        features = engine.get_latest_features(enriched)
        assert isinstance(features, dict)
        assert len(features) > 10
        assert "rsi_14" in features

    def test_feature_columns_list(self, engine):
        """Feature column list should be non-empty and contain expected entries."""
        cols = FeatureEngine.get_feature_columns()
        assert len(cols) > 20
        assert "rsi_14" in cols
        assert "vol_ratio" in cols
        assert "regime_score" in cols

    def test_original_ohlcv_preserved(self, engine, sample_ohlcv):
        """Original OHLCV columns should not be modified."""
        original_close = sample_ohlcv["close"].copy()
        result = engine.compute_all(sample_ohlcv.copy(), "TEST")
        pd.testing.assert_series_equal(
            result["close"].reset_index(drop=True),
            original_close.reset_index(drop=True),
            check_names=False,
        )


# ── Config Tests ─────────────────────────────────────────────────

class TestConfig:

    def test_config_loads(self):
        """Config should load without errors."""
        config = get_config()
        assert config is not None

    def test_paper_mode_default(self):
        """Default trading mode should be paper."""
        config = get_config()
        assert config.trading.mode == "paper"
        assert config.is_paper is True
        assert config.is_live is False

    def test_risk_params_reasonable(self):
        """Risk params should be within safe bounds."""
        config = get_config()
        assert 0 < config.risk.max_position_pct <= 0.10
        assert 0 < config.risk.max_drawdown_pct <= 0.30
        assert 0 < config.risk.stop_loss_pct <= 0.15
        assert config.risk.min_confidence >= 0.50

    def test_asset_universe_not_empty(self):
        """Asset lists should be populated."""
        config = get_config()
        assert len(config.assets.stocks) > 0
        assert len(config.assets.all_tradeable) > 0
        assert "NVDA" in config.assets.stocks

    def test_quality_gate_thresholds(self):
        """Quality gate thresholds should be sensible."""
        config = get_config()
        qg = config.quality_gate
        assert qg.min_sharpe >= 0.5
        assert qg.min_win_rate >= 0.40
        assert qg.max_drawdown <= 0.35
        assert qg.min_trades >= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
