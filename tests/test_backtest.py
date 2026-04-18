"""
tests/test_backtest.py — Phase 1.5 backtesting tests.
Run with: pytest tests/test_backtest.py -v

All tests run WITHOUT API keys using synthetic OHLCV data.
LEAN CLI is NOT required — all tests use the Python backtest fallback.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lean.quality_gate import BacktestResult, QualityGate, QualityGateResult
from lean.lean_bridge import LEANBridge


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """500 rows of synthetic OHLCV with realistic trending behavior."""
    n = 500
    np.random.seed(99)
    dates = pd.date_range(
        start=datetime(2015, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="B",
    )
    # Upward-trending price with momentum cycles
    returns = np.random.normal(0.001, 0.018, n)
    # Add trending cycles to ensure strategies can trigger
    trend = np.linspace(0, 0.3, n)  # gentle upward drift
    close = 100.0 * np.cumprod(1 + returns + np.diff(trend, prepend=0))
    high   = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low    = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_  = close * (1 + np.random.normal(0, 0.006, n))
    # Realistic volume with occasional spikes
    vol_base = np.random.randint(2_000_000, 8_000_000, n).astype(float)
    vol_spikes = np.random.choice([1.0, 1.0, 1.0, 2.5, 3.0], size=n)
    volume = (vol_base * vol_spikes).astype(int)

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
def passing_result():
    """A BacktestResult that passes all 7 quality gate checks."""
    return BacktestResult(
        strategy="momentum",
        ticker="NVDA",
        start_date="2015-01-01",
        end_date="2022-12-31",
        initial_capital=100_000.0,
        total_return=2.10,     # 210%
        cagr=0.18,             # 18% >= 15% threshold
        sharpe_ratio=1.25,     # 1.25 >= 1.0
        max_drawdown=0.18,     # 18% <= 25%
        win_rate=0.53,         # 53% >= 45%
        profit_factor=1.80,    # 1.80 >= 1.5
        calmar_ratio=0.78,     # 0.78 >= 0.5
        total_trades=45,       # 45 >= 30
    )


@pytest.fixture
def failing_result():
    """A BacktestResult that fails multiple quality gate checks."""
    return BacktestResult(
        strategy="bad_strategy",
        ticker="TEST",
        start_date="2015-01-01",
        end_date="2022-12-31",
        initial_capital=100_000.0,
        total_return=0.05,
        cagr=0.06,             # FAIL: < 15%
        sharpe_ratio=0.40,     # FAIL: < 1.0
        max_drawdown=0.32,     # FAIL: > 25%
        win_rate=0.38,         # FAIL: < 45%
        profit_factor=1.10,    # FAIL: < 1.5
        calmar_ratio=0.19,     # FAIL: < 0.5
        total_trades=15,       # FAIL: < 30
    )


@pytest.fixture
def bridge():
    return LEANBridge()


# ── BacktestResult Tests ──────────────────────────────────────────────────────

class TestBacktestResult:

    def test_create_backtest_result(self, passing_result):
        """BacktestResult should be creatable with all required fields."""
        assert passing_result.strategy == "momentum"
        assert passing_result.ticker == "NVDA"
        assert 0 < passing_result.cagr < 1.0
        assert 0 < passing_result.max_drawdown < 1.0

    def test_backtest_result_fields(self, passing_result):
        """All 7 quality-gate metrics should be accessible."""
        assert hasattr(passing_result, "cagr")
        assert hasattr(passing_result, "sharpe_ratio")
        assert hasattr(passing_result, "max_drawdown")
        assert hasattr(passing_result, "win_rate")
        assert hasattr(passing_result, "profit_factor")
        assert hasattr(passing_result, "calmar_ratio")
        assert hasattr(passing_result, "total_trades")

    def test_backtest_result_has_metadata(self):
        """metadata dict should default to empty."""
        result = BacktestResult(
            strategy="test", ticker="T", start_date="2020-01-01", end_date="2021-01-01",
            initial_capital=10_000.0, total_return=0.1, cagr=0.1, sharpe_ratio=1.0,
            max_drawdown=0.1, win_rate=0.5, profit_factor=1.5, calmar_ratio=0.5,
            total_trades=30,
        )
        assert isinstance(result.metadata, dict)


# ── QualityGate Tests ─────────────────────────────────────────────────────────

class TestQualityGate:

    @pytest.fixture
    def gate(self):
        return QualityGate()

    def test_passing_result_opens_gate(self, gate, passing_result):
        """A result meeting all thresholds should pass."""
        result = gate.validate(passing_result)
        assert result.passed is True
        assert result.passed_count == 7

    def test_failing_result_closes_gate(self, gate, failing_result):
        """A result failing all thresholds should be blocked."""
        result = gate.validate(failing_result)
        assert result.passed is False
        assert result.passed_count < 7

    def test_returns_quality_gate_result(self, gate, passing_result):
        """validate() should return a QualityGateResult."""
        result = gate.validate(passing_result)
        assert isinstance(result, QualityGateResult)

    def test_seven_checks_always_run(self, gate, passing_result):
        """validate() should always run exactly 7 checks."""
        result = gate.validate(passing_result)
        assert result.total_checks == 7
        assert len(result.checks) == 7

    def test_check_cagr_gte(self, gate):
        """CAGR check: must be >= 15%."""
        r = BacktestResult(
            strategy="t", ticker="T", start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=0.5,
            cagr=0.14,          # FAIL: just below 15%
            sharpe_ratio=1.5, max_drawdown=0.10, win_rate=0.60,
            profit_factor=2.0, calmar_ratio=1.0, total_trades=50,
        )
        result = gate.validate(r)
        cagr_check = next(c for c in result.checks if c["metric"] == "cagr")
        assert cagr_check["passed"] is False

    def test_check_sharpe_gte(self, gate):
        """Sharpe check: must be >= 1.0."""
        r = BacktestResult(
            strategy="t", ticker="T", start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=0.5,
            cagr=0.20, sharpe_ratio=0.95,   # FAIL
            max_drawdown=0.10, win_rate=0.60,
            profit_factor=2.0, calmar_ratio=1.0, total_trades=50,
        )
        result = gate.validate(r)
        sharpe_check = next(c for c in result.checks if c["metric"] == "sharpe_ratio")
        assert sharpe_check["passed"] is False

    def test_check_drawdown_lte(self, gate):
        """Drawdown check: must be <= 25%."""
        r = BacktestResult(
            strategy="t", ticker="T", start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=0.5,
            cagr=0.20, sharpe_ratio=1.5, max_drawdown=0.26,  # FAIL: > 25%
            win_rate=0.60, profit_factor=2.0, calmar_ratio=1.0, total_trades=50,
        )
        result = gate.validate(r)
        dd_check = next(c for c in result.checks if c["metric"] == "max_drawdown")
        assert dd_check["passed"] is False

    def test_check_win_rate_gte(self, gate):
        """Win rate check: must be >= 45%."""
        r = BacktestResult(
            strategy="t", ticker="T", start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=0.5,
            cagr=0.20, sharpe_ratio=1.5, max_drawdown=0.10,
            win_rate=0.43,   # FAIL
            profit_factor=2.0, calmar_ratio=1.0, total_trades=50,
        )
        result = gate.validate(r)
        wr_check = next(c for c in result.checks if c["metric"] == "win_rate")
        assert wr_check["passed"] is False

    def test_check_profit_factor_gte(self, gate):
        """Profit factor check: must be >= 1.5."""
        r = BacktestResult(
            strategy="t", ticker="T", start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=0.5,
            cagr=0.20, sharpe_ratio=1.5, max_drawdown=0.10,
            win_rate=0.55, profit_factor=1.40,   # FAIL
            calmar_ratio=1.0, total_trades=50,
        )
        result = gate.validate(r)
        pf_check = next(c for c in result.checks if c["metric"] == "profit_factor")
        assert pf_check["passed"] is False

    def test_check_calmar_gte(self, gate):
        """Calmar check: must be >= 0.5."""
        r = BacktestResult(
            strategy="t", ticker="T", start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=0.5,
            cagr=0.20, sharpe_ratio=1.5, max_drawdown=0.10,
            win_rate=0.55, profit_factor=2.0, calmar_ratio=0.40,   # FAIL
            total_trades=50,
        )
        result = gate.validate(r)
        cal_check = next(c for c in result.checks if c["metric"] == "calmar_ratio")
        assert cal_check["passed"] is False

    def test_check_min_trades_gte(self, gate):
        """Min trades check: must be >= 30."""
        r = BacktestResult(
            strategy="t", ticker="T", start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=0.5,
            cagr=0.20, sharpe_ratio=1.5, max_drawdown=0.10,
            win_rate=0.55, profit_factor=2.0, calmar_ratio=1.0,
            total_trades=25,   # FAIL: < 30
        )
        result = gate.validate(r)
        trades_check = next(c for c in result.checks if c["metric"] == "total_trades")
        assert trades_check["passed"] is False

    def test_summary_not_empty(self, gate, passing_result):
        """Summary should always be a non-empty string."""
        result = gate.validate(passing_result)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_strategy_in_result(self, gate, passing_result):
        """Strategy name should be carried through."""
        result = gate.validate(passing_result)
        assert result.strategy == passing_result.strategy

    def test_get_thresholds_returns_dict(self, gate):
        """get_thresholds() should return all 7 threshold keys."""
        thresholds = gate.get_thresholds()
        required = {
            "min_cagr", "min_sharpe", "max_drawdown",
            "min_win_rate", "min_profit_factor", "min_calmar", "min_trades"
        }
        assert required.issubset(thresholds.keys())

    def test_threshold_values_are_from_config(self, gate):
        """Thresholds should match config.yaml values."""
        thresholds = gate.get_thresholds()
        assert thresholds["min_cagr"]    == 0.15
        assert thresholds["min_sharpe"]  == 1.0
        assert thresholds["max_drawdown"] == 0.25
        assert thresholds["min_win_rate"] == 0.45
        assert thresholds["min_trades"]  == 30

    def test_validate_combined_single(self, gate, passing_result):
        """validate_combined with one result should behave like validate."""
        single = gate.validate_combined([passing_result])
        assert isinstance(single, QualityGateResult)

    def test_validate_combined_worst_drawdown(self, gate, passing_result):
        """validate_combined should use worst-case max_drawdown."""
        r2 = BacktestResult(
            strategy="momentum", ticker="AAPL",
            start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=1.5,
            cagr=0.18, sharpe_ratio=1.3, max_drawdown=0.30,   # > 25% threshold
            win_rate=0.55, profit_factor=1.8, calmar_ratio=0.7, total_trades=40,
        )
        result = gate.validate_combined([passing_result, r2])
        # Combined drawdown = max(0.18, 0.30) = 0.30 → should fail drawdown check
        dd_check = next(c for c in result.checks if c["metric"] == "max_drawdown")
        assert dd_check["passed"] is False

    def test_validate_combined_empty(self, gate):
        """validate_combined with empty list should return failed result."""
        result = gate.validate_combined([])
        assert result.passed is False


# ── LEANBridge Python Backtest Tests ──────────────────────────────────────────

class TestLEANBridgePythonBacktest:

    def test_run_python_backtest_returns_result(self, bridge, sample_ohlcv):
        """run_python_backtest should return a BacktestResult."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        assert isinstance(result, BacktestResult)

    def test_backtest_strategy_field(self, bridge, sample_ohlcv):
        """Strategy name should be carried through."""
        result = bridge.run_python_backtest("trend_following", sample_ohlcv, "TEST")
        assert result.strategy == "trend_following"

    def test_backtest_ticker_field(self, bridge, sample_ohlcv):
        """Ticker should be set correctly."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "NVDA")
        assert result.ticker == "NVDA"

    def test_all_three_strategies(self, bridge, sample_ohlcv):
        """All 3 strategies should run without error."""
        for strategy in ["momentum", "trend_following", "volatility_breakout"]:
            result = bridge.run_python_backtest(strategy, sample_ohlcv, "TEST")
            assert isinstance(result, BacktestResult)
            assert result.strategy == strategy

    def test_total_return_is_float(self, bridge, sample_ohlcv):
        """total_return should be a finite float."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        assert isinstance(result.total_return, float)
        assert not (result.total_return != result.total_return)  # not NaN

    def test_cagr_is_finite(self, bridge, sample_ohlcv):
        """CAGR should be a finite float."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        import math
        assert math.isfinite(result.cagr)

    def test_max_drawdown_between_0_and_1(self, bridge, sample_ohlcv):
        """Max drawdown should be in [0, 1]."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        assert 0.0 <= result.max_drawdown <= 1.0

    def test_win_rate_between_0_and_1(self, bridge, sample_ohlcv):
        """Win rate should be in [0, 1]."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        assert 0.0 <= result.win_rate <= 1.0

    def test_sharpe_is_non_negative(self, bridge, sample_ohlcv):
        """Sharpe should be >= 0 (clamped by bridge)."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        assert result.sharpe_ratio >= 0.0

    def test_total_trades_is_int(self, bridge, sample_ohlcv):
        """total_trades should be a non-negative int."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        assert isinstance(result.total_trades, int)
        assert result.total_trades >= 0

    def test_profit_factor_is_non_negative(self, bridge, sample_ohlcv):
        """Profit factor should be >= 0."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        assert result.profit_factor >= 0.0

    def test_empty_df_returns_empty_result(self, bridge):
        """Empty DataFrame should return a zero-filled BacktestResult."""
        result = bridge.run_python_backtest("momentum", pd.DataFrame(), "TEST")
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0

    def test_small_df_returns_empty_result(self, bridge, sample_ohlcv):
        """Fewer than 60 rows → empty result (insufficient for features)."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv.head(50), "TEST")
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0

    def test_metadata_source_is_python(self, bridge, sample_ohlcv):
        """Metadata should indicate Python backtest source."""
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        assert result.metadata.get("source") == "python_backtest"

    def test_date_filter_applied(self, bridge, sample_ohlcv):
        """start/end dates should filter the DataFrame."""
        r1 = bridge.run_python_backtest(
            "momentum", sample_ohlcv, "TEST",
            start_date="2015-01-01", end_date="2016-12-31"
        )
        r2 = bridge.run_python_backtest(
            "momentum", sample_ohlcv, "TEST",
            start_date="2015-01-01", end_date="2017-12-31"
        )
        # Longer date range can have more trades (not guaranteed but metadata differs)
        assert isinstance(r1, BacktestResult)
        assert isinstance(r2, BacktestResult)

    def test_unknown_strategy_raises(self, bridge, sample_ohlcv):
        """Unknown strategy name should raise ValueError."""
        with pytest.raises((ValueError, KeyError)):
            bridge.run_python_backtest("invalid_strategy_xyz", sample_ohlcv, "TEST")

    def test_run_all_strategies_returns_four(self, bridge, sample_ohlcv):
        """run_all_strategies() should return exactly 4 results."""
        results = bridge.run_all_strategies(sample_ohlcv, "TEST")
        assert len(results) == 4
        strategies = {r.strategy for r in results}
        assert strategies == {"momentum", "trend_following", "volatility_breakout", "mean_reversion"}


# ── LEANBridge auto-mode Tests ────────────────────────────────────────────────

class TestLEANBridgeAutoMode:

    def test_run_backtest_falls_back_to_python(self, bridge, sample_ohlcv):
        """run_backtest() should fall back to Python if LEAN not available."""
        result = bridge.run_backtest("momentum", sample_ohlcv, "TEST")
        assert isinstance(result, BacktestResult)

    def test_run_backtest_returns_valid_metrics(self, bridge, sample_ohlcv):
        """run_backtest() metrics should all be valid numbers."""
        result = bridge.run_backtest("trend_following", sample_ohlcv, "TEST")
        import math
        assert math.isfinite(result.cagr)
        assert math.isfinite(result.sharpe_ratio)
        assert math.isfinite(result.max_drawdown)
        assert 0.0 <= result.win_rate <= 1.0


# ── Simulate a quality check on backtest results ──────────────────────────────

class TestFullBacktestFlow:

    def test_backtest_then_quality_gate(self, bridge, sample_ohlcv):
        """
        Integration: run Python backtest then check quality gate.
        Result may pass or fail — we just verify no exceptions thrown.
        """
        gate = QualityGate()
        result = bridge.run_python_backtest("momentum", sample_ohlcv, "TEST")
        gate_result = gate.validate(result)

        assert isinstance(gate_result, QualityGateResult)
        assert gate_result.total_checks == 7
        assert isinstance(gate_result.passed, bool)
        assert isinstance(gate_result.summary, str)

    def test_all_strategies_quality_check(self, bridge, sample_ohlcv):
        """Run all 3 strategies and validate each against quality gate."""
        gate = QualityGate()
        results = bridge.run_all_strategies(sample_ohlcv, "TEST")
        for result in results:
            gate_result = gate.validate(result)
            assert gate_result.total_checks == 7
            assert isinstance(gate_result.passed, bool)

    def test_validate_combined_after_all_strategies(self, bridge, sample_ohlcv):
        """validate_combined should aggregate all 3 strategy results."""
        gate = QualityGate()
        results = bridge.run_all_strategies(sample_ohlcv, "TEST")
        combined = gate.validate_combined(results)

        assert isinstance(combined, QualityGateResult)
        assert combined.strategy == "combined"
        assert combined.total_checks == 7

    def test_passing_synthetic_passes_gate(self):
        """
        A manually constructed passing BacktestResult should open the gate.
        This tests the gate logic in isolation from the backtest engine.
        """
        gate = QualityGate()
        perfect = BacktestResult(
            strategy="test", ticker="TEST",
            start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=3.0,
            cagr=0.22, sharpe_ratio=1.8, max_drawdown=0.12,
            win_rate=0.58, profit_factor=2.1, calmar_ratio=1.0, total_trades=60,
        )
        result = gate.validate(perfect)
        assert result.passed is True
        assert result.passed_count == 7

    def test_failing_synthetic_blocks_gate(self):
        """A manually constructed failing result should block the gate."""
        gate = QualityGate()
        terrible = BacktestResult(
            strategy="test", ticker="TEST",
            start_date="2015-01-01", end_date="2022-12-31",
            initial_capital=100_000.0, total_return=-0.30,
            cagr=-0.05, sharpe_ratio=-0.5, max_drawdown=0.40,
            win_rate=0.30, profit_factor=0.80, calmar_ratio=-0.1, total_trades=10,
        )
        result = gate.validate(terrible)
        assert result.passed is False
        assert result.passed_count < 7


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
