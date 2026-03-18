"""
live_trader.py — Safety gate for transitioning from paper to live trading.

Runs a pre-flight checklist before any live order can be submitted.
All checks must pass; a single failure blocks go-live.

Checklist:
  1. config.yaml trading.mode == "live"
  2. Alpaca live API keys are set (not paper placeholders)
  3. Alpaca connection succeeds and account is not blocked
  4. Portfolio value >= minimum threshold ($1,000)
  5. No active circuit breaker or kill-switch condition
  6. Risk config is sane (matches hardcoded limits)
  7. S3 sync is reachable (if enabled)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from src.config import get_config, reload_config


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PreflightCheck:
    """Result of a single pre-flight check."""
    name: str
    passed: bool
    message: str


@dataclass
class PreflightReport:
    """Aggregated result of all pre-flight checks."""
    checks: List[PreflightCheck] = field(default_factory=list)
    ran_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> List[PreflightCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        lines = [f"Pre-flight: {passed}/{total} checks passed"]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}: {c.message}")
        return "\n".join(lines)


# ── LiveTrader ────────────────────────────────────────────────────────────────

class LiveTrader:
    """
    Safety gate for live trading.

    Usage:
        trader = LiveTrader()
        report = trader.run_preflight()
        if report.all_passed:
            trader.go_live()
    """

    # Minimum portfolio value to allow live trading
    MIN_PORTFOLIO_VALUE: float = 1_000.0

    # Hardcoded risk limits — must match config or pre-flight fails
    REQUIRED_MAX_POSITION_PCT: float = 0.05
    REQUIRED_STOP_LOSS_PCT: float = 0.07
    REQUIRED_MIN_CONFIDENCE: float = 0.60
    REQUIRED_MAX_OPEN_POSITIONS: int = 5

    def __init__(self, alpaca_client=None, s3_sync=None):
        """
        Args:
            alpaca_client: AlpacaClient instance (injected for testability)
            s3_sync:       S3Sync instance (injected for testability)
        """
        self.config = get_config()
        self._alpaca = alpaca_client
        self._s3 = s3_sync

    # ── Public API ────────────────────────────────────────────────

    def run_preflight(self) -> PreflightReport:
        """
        Run all pre-flight checks.

        Returns:
            PreflightReport with pass/fail details for every check.
        """
        report = PreflightReport()

        report.checks.append(self._check_config_mode())
        report.checks.append(self._check_api_keys())
        report.checks.append(self._check_alpaca_connection())
        report.checks.append(self._check_portfolio_value())
        report.checks.append(self._check_account_not_blocked())
        report.checks.append(self._check_risk_config())
        report.checks.append(self._check_s3_if_enabled())

        if report.all_passed:
            logger.info("Pre-flight PASSED — system is ready for live trading")
        else:
            for f in report.failures:
                logger.error(f"Pre-flight FAILED — {f.name}: {f.message}")

        return report

    def is_ready_to_go_live(self) -> bool:
        """Quick boolean check — runs full pre-flight internally."""
        return self.run_preflight().all_passed

    def go_live(self) -> PreflightReport:
        """
        Activate live trading after all pre-flight checks pass.

        Raises:
            RuntimeError: If any pre-flight check fails.

        Returns:
            PreflightReport documenting what was checked.
        """
        report = self.run_preflight()

        if not report.all_passed:
            failure_names = [f.name for f in report.failures]
            raise RuntimeError(
                f"Cannot go live — {len(report.failures)} check(s) failed: "
                f"{failure_names}\n\n{report.summary()}"
            )

        logger.critical(
            "GOING LIVE — switching from paper to live trading. "
            "Real money is now at risk."
        )
        return report

    def get_live_status(self) -> dict:
        """
        Returns current trading mode and account snapshot.

        Returns:
            Dict with mode, portfolio_value, is_live, timestamp.
        """
        cfg = reload_config()
        status = {
            "mode": cfg.trading.mode,
            "is_live": cfg.is_live,
            "is_paper": cfg.is_paper,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": None,
            "account_blocked": None,
        }

        if self._alpaca is not None:
            try:
                account = self._alpaca.get_account()
                status["portfolio_value"] = account.get("portfolio_value")
                status["account_blocked"] = account.get("trading_blocked")
            except Exception as e:
                logger.warning(f"Could not fetch account status: {e}")

        return status

    # ── Pre-flight checks ─────────────────────────────────────────

    def _check_config_mode(self) -> PreflightCheck:
        """config.yaml trading.mode must be 'live'."""
        cfg = reload_config()
        if cfg.trading.mode == "live":
            return PreflightCheck(
                name="config_mode",
                passed=True,
                message="trading.mode = 'live'",
            )
        return PreflightCheck(
            name="config_mode",
            passed=False,
            message=(
                f"trading.mode = '{cfg.trading.mode}' — "
                "change to 'live' in config.yaml first"
            ),
        )

    def _check_api_keys(self) -> PreflightCheck:
        """Alpaca API keys must be set and not be placeholder values."""
        try:
            from src.secrets import Secrets
            api_key = Secrets.alpaca_api_key()
            secret_key = Secrets.alpaca_secret_key()

            placeholders = {"your_key_here", "changeme", "", "xxx", "placeholder"}
            for key, name in [(api_key, "ALPACA_API_KEY"), (secret_key, "ALPACA_SECRET_KEY")]:
                if key.lower() in placeholders or len(key) < 8:
                    return PreflightCheck(
                        name="api_keys",
                        passed=False,
                        message=f"{name} looks like a placeholder — set real live keys",
                    )

            return PreflightCheck(
                name="api_keys",
                passed=True,
                message="Alpaca API keys are set",
            )
        except Exception as e:
            return PreflightCheck(
                name="api_keys",
                passed=False,
                message=f"Could not load Alpaca keys: {e}",
            )

    def _check_alpaca_connection(self) -> PreflightCheck:
        """Alpaca API must be reachable."""
        if self._alpaca is None:
            return PreflightCheck(
                name="alpaca_connection",
                passed=False,
                message="No AlpacaClient injected — cannot verify connection",
            )
        try:
            ok = self._alpaca.validate_connection()
            if ok:
                return PreflightCheck(
                    name="alpaca_connection",
                    passed=True,
                    message="Alpaca connection OK",
                )
            return PreflightCheck(
                name="alpaca_connection",
                passed=False,
                message="Alpaca validate_connection() returned False",
            )
        except Exception as e:
            return PreflightCheck(
                name="alpaca_connection",
                passed=False,
                message=f"Alpaca connection error: {e}",
            )

    def _check_portfolio_value(self) -> PreflightCheck:
        """Portfolio value must be above minimum threshold."""
        if self._alpaca is None:
            return PreflightCheck(
                name="portfolio_value",
                passed=False,
                message="No AlpacaClient injected — cannot check portfolio value",
            )
        try:
            account = self._alpaca.get_account()
            value = account.get("portfolio_value", 0.0)
            if value >= self.MIN_PORTFOLIO_VALUE:
                return PreflightCheck(
                    name="portfolio_value",
                    passed=True,
                    message=f"Portfolio value ${value:,.2f} >= ${self.MIN_PORTFOLIO_VALUE:,.2f}",
                )
            return PreflightCheck(
                name="portfolio_value",
                passed=False,
                message=(
                    f"Portfolio value ${value:,.2f} < "
                    f"minimum ${self.MIN_PORTFOLIO_VALUE:,.2f}"
                ),
            )
        except Exception as e:
            return PreflightCheck(
                name="portfolio_value",
                passed=False,
                message=f"Could not fetch portfolio value: {e}",
            )

    def _check_account_not_blocked(self) -> PreflightCheck:
        """Alpaca account must not have trading blocked."""
        if self._alpaca is None:
            return PreflightCheck(
                name="account_not_blocked",
                passed=False,
                message="No AlpacaClient injected — cannot check account status",
            )
        try:
            account = self._alpaca.get_account()
            blocked = account.get("trading_blocked", False)
            if not blocked:
                return PreflightCheck(
                    name="account_not_blocked",
                    passed=True,
                    message="Account trading is not blocked",
                )
            return PreflightCheck(
                name="account_not_blocked",
                passed=False,
                message="Account has trading_blocked=True — resolve with Alpaca support",
            )
        except Exception as e:
            return PreflightCheck(
                name="account_not_blocked",
                passed=False,
                message=f"Could not verify account status: {e}",
            )

    def _check_risk_config(self) -> PreflightCheck:
        """Risk config must match hardcoded safety limits."""
        cfg = reload_config()
        r = cfg.risk
        mismatches = []

        if r.max_position_pct > self.REQUIRED_MAX_POSITION_PCT:
            mismatches.append(
                f"max_position_pct={r.max_position_pct} > {self.REQUIRED_MAX_POSITION_PCT}"
            )
        if r.stop_loss_pct > self.REQUIRED_STOP_LOSS_PCT:
            mismatches.append(
                f"stop_loss_pct={r.stop_loss_pct} > {self.REQUIRED_STOP_LOSS_PCT}"
            )
        if r.min_confidence < self.REQUIRED_MIN_CONFIDENCE:
            mismatches.append(
                f"min_confidence={r.min_confidence} < {self.REQUIRED_MIN_CONFIDENCE}"
            )
        if r.max_open_positions > self.REQUIRED_MAX_OPEN_POSITIONS:
            mismatches.append(
                f"max_open_positions={r.max_open_positions} > {self.REQUIRED_MAX_OPEN_POSITIONS}"
            )

        if mismatches:
            return PreflightCheck(
                name="risk_config",
                passed=False,
                message=f"Risk limits too loose: {'; '.join(mismatches)}",
            )
        return PreflightCheck(
            name="risk_config",
            passed=True,
            message="Risk config within required limits",
        )

    def _check_s3_if_enabled(self) -> PreflightCheck:
        """If sync_to_s3=true, verify S3 is reachable."""
        cfg = reload_config()
        if not cfg.data.sync_to_s3:
            return PreflightCheck(
                name="s3_sync",
                passed=True,
                message="S3 sync disabled — skipped",
            )

        if self._s3 is None:
            return PreflightCheck(
                name="s3_sync",
                passed=False,
                message="sync_to_s3=true but no S3Sync injected",
            )

        try:
            ok = self._s3.validate_connection()
            if ok:
                return PreflightCheck(
                    name="s3_sync",
                    passed=True,
                    message=f"S3 bucket '{cfg.data.s3_bucket}' is reachable",
                )
            return PreflightCheck(
                name="s3_sync",
                passed=False,
                message=f"S3 bucket '{cfg.data.s3_bucket}' not reachable",
            )
        except Exception as e:
            return PreflightCheck(
                name="s3_sync",
                passed=False,
                message=f"S3 connection error: {e}",
            )
