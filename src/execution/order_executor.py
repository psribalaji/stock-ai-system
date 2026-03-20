"""
execution/order_executor.py — Translates TradeDecision objects into Alpaca orders.

Safety rules (hard-coded, never bypassed):
  - Will not place live orders unless config.trading.mode == "live"
  - Will not place any order when RiskManager kill switch is active
  - Every order attempt is written to the audit log (success or failure)
  - Quantity is always derived from position_size_usd / entry_price (rounded down)
  - Stop loss is always attached on BUY orders

Usage:
    executor = OrderExecutor()
    results  = executor.execute_all(decisions)  # List[TradeDecision] → List[OrderResult]
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.config import get_config
from src.execution.decision_engine import TradeDecision
from src.ingestion.storage import ParquetStore


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """Outcome of a single order submission attempt."""
    trade_id: str
    timestamp_submitted: datetime
    ticker: str
    direction: str
    strategy: str
    qty: float
    entry_price: float
    stop_loss_price: float
    position_size_usd: float
    confidence: float
    order_id: str           # Alpaca order ID (empty on failure)
    status: str             # "submitted" | "skipped" | "failed"
    mode: str               # "paper" | "live"
    error: str              # Empty on success
    approved: bool          # Always True (only approved decisions reach executor)
    block_reason: str       # Propagated from TradeDecision


# ── Executor ──────────────────────────────────────────────────────────────────

class OrderExecutor:
    """
    Submits approved TradeDecisions as Alpaca market orders and logs to audit.

    Args:
        alpaca_client: Pre-built AlpacaClient (created lazily if None — avoids
                       requiring API keys during unit tests)
        store:         ParquetStore for audit logging
        dry_run:       If True, log orders but never submit (safe for testing)
    """

    def __init__(
        self,
        alpaca_client=None,
        store: Optional[ParquetStore] = None,
        dry_run: bool = False,
    ) -> None:
        self.config   = get_config()
        self._alpaca  = alpaca_client   # lazy init via _get_client()
        self.store    = store or ParquetStore(self.config.data.storage_path)
        self.dry_run  = dry_run
        self._kill_switch_active = False

        mode = "DRY-RUN" if dry_run else self.config.trading.mode.upper()
        logger.info(f"[OrderExecutor] Initialised — mode: {mode}")

    # ── Public API ────────────────────────────────────────────────────────────

    def execute_all(self, decisions: List[TradeDecision]) -> List[OrderResult]:
        """
        Execute all approved decisions and return results.

        Args:
            decisions: List of TradeDecision (only approved ones should be passed)

        Returns:
            List of OrderResult, one per decision.
        """
        results = []
        for decision in decisions:
            result = self.execute(decision)
            results.append(result)

        # Persist all results to audit log
        if results:
            self._save_audit(results)

        submitted = sum(1 for r in results if r.status == "submitted")
        skipped   = sum(1 for r in results if r.status == "skipped")
        failed    = sum(1 for r in results if r.status == "failed")
        logger.info(
            f"[OrderExecutor] Batch complete: "
            f"{submitted} submitted, {skipped} skipped, {failed} failed"
        )
        return results

    def execute(self, decision: TradeDecision) -> OrderResult:
        """
        Execute a single TradeDecision.

        Args:
            decision: An approved TradeDecision from DecisionEngine

        Returns:
            OrderResult with submission details.
        """
        trade_id = str(uuid.uuid4())

        # ── Safety checks ─────────────────────────────────────────────────
        skip_reason = self._check_safety(decision)
        if not skip_reason:
            skip_reason = self._check_position(decision)
        if skip_reason:
            logger.warning(
                f"[OrderExecutor] SKIPPED {decision.ticker} "
                f"{decision.direction}: {skip_reason}"
            )
            return self._make_result(decision, trade_id,
                                     status="skipped", error=skip_reason)

        # ── Compute quantity ──────────────────────────────────────────────
        qty = self._compute_qty(decision)
        if qty <= 0:
            msg = (f"Computed qty={qty:.4f} — position too small to trade "
                   f"(size_usd=${decision.position_size_usd:.2f}, "
                   f"price=${decision.entry_price:.2f})")
            logger.warning(f"[OrderExecutor] SKIPPED {decision.ticker}: {msg}")
            return self._make_result(decision, trade_id, qty=qty,
                                     status="skipped", error=msg)

        # ── Dry run ───────────────────────────────────────────────────────
        if self.dry_run:
            logger.info(
                f"[OrderExecutor] DRY-RUN: would {decision.direction} "
                f"{qty:.4f} {decision.ticker} @ ~${decision.entry_price:.2f} "
                f"(stop ${decision.stop_loss_price:.2f})"
            )
            return self._make_result(decision, trade_id, qty=qty,
                                     status="submitted", order_id="DRY-RUN")

        # ── Submit order ──────────────────────────────────────────────────
        try:
            client  = self._get_client()
            stop    = decision.stop_loss_price if decision.direction == "BUY" else None
            order   = client.submit_market_order(
                ticker=decision.ticker,
                qty=qty,
                side=decision.direction.lower(),
                stop_loss_price=stop,
            )
            order_id = order.get("order_id", "")
            logger.info(
                f"[OrderExecutor] ORDER SUBMITTED: {decision.direction} "
                f"{qty:.4f} {decision.ticker} | order_id={order_id}"
            )
            return self._make_result(decision, trade_id, qty=qty,
                                     status="submitted", order_id=order_id)

        except Exception as exc:
            logger.error(
                f"[OrderExecutor] ORDER FAILED: {decision.direction} "
                f"{decision.ticker}: {exc}"
            )
            return self._make_result(decision, trade_id, qty=qty,
                                     status="failed", error=str(exc))

    def activate_kill_switch(self) -> None:
        """
        Activate the kill switch — blocks all future order submissions.
        Also cancels open orders and closes all positions via Alpaca.
        """
        logger.critical("[OrderExecutor] KILL SWITCH ACTIVATED")
        self._kill_switch_active = True
        try:
            client = self._get_client()
            client.close_all_positions()
        except Exception as exc:
            logger.error(f"[OrderExecutor] Kill switch close-all failed: {exc}")

    def deactivate_kill_switch(self) -> None:
        """Manually deactivate the kill switch (requires human intervention)."""
        self._kill_switch_active = False
        logger.info("[OrderExecutor] Kill switch deactivated — trading resumed")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_position(self, decision: TradeDecision) -> str:
        """
        Return a skip reason if position state conflicts with this order:
          - BUY when we already hold a position → skip (prevents wash trade)
          - SELL when we hold no position → skip (prevents accidental short)
        Returns empty string if safe to proceed.
        """
        if self.dry_run:
            return ""
        try:
            positions = self._get_client().get_positions()
            held_tickers = set(positions["ticker"].tolist()) if not positions.empty else set()
        except Exception as exc:
            logger.warning(f"[OrderExecutor] Could not fetch positions for {decision.ticker}: {exc}")
            return ""

        if decision.direction == "BUY" and decision.ticker in held_tickers:
            return f"already holding {decision.ticker} — skipping duplicate BUY"

        if decision.direction == "SELL" and decision.ticker not in held_tickers:
            return f"no position in {decision.ticker} — skipping SELL (would be short)"

        return ""

    def _check_safety(self, decision: TradeDecision) -> str:
        """
        Return a non-empty reason string if the order should be blocked,
        or an empty string if it's safe to proceed.
        """
        if self._kill_switch_active:
            return "kill switch is active"

        if not decision.approved:
            return f"decision not approved: {decision.block_reason}"

        mode = self.config.trading.mode
        if mode not in ("paper", "live"):
            return f"unknown trading mode '{mode}'"

        return ""

    def _compute_qty(self, decision: TradeDecision) -> float:
        """
        Compute share quantity from position size USD and entry price.
        Rounds down to 2 decimal places (Alpaca supports fractional shares).
        """
        if decision.entry_price <= 0:
            return 0.0
        raw = decision.position_size_usd / decision.entry_price
        return max(0.0, round(raw - 0.005, 2))   # floor to 2dp

    def _get_client(self):
        """Lazy-init the Alpaca client on first use."""
        if self._alpaca is None:
            from src.ingestion.alpaca_client import AlpacaClient
            self._alpaca = AlpacaClient()
        return self._alpaca

    def _make_result(
        self,
        decision: TradeDecision,
        trade_id: str,
        qty: float = 0.0,
        status: str = "skipped",
        order_id: str = "",
        error: str = "",
    ) -> OrderResult:
        return OrderResult(
            trade_id=trade_id,
            timestamp_submitted=datetime.now(timezone.utc),
            ticker=decision.ticker,
            direction=decision.direction,
            strategy=decision.strategy,
            qty=qty,
            entry_price=decision.entry_price,
            stop_loss_price=decision.stop_loss_price,
            position_size_usd=decision.position_size_usd,
            confidence=decision.confidence,
            order_id=order_id,
            status=status,
            mode=self.config.trading.mode,
            error=error,
            approved=decision.approved,
            block_reason=decision.block_reason,
        )

    def _save_audit(self, results: List[OrderResult]) -> None:
        """Persist all order results to the monthly audit Parquet file."""
        rows = [
            {
                "trade_id":            r.trade_id,
                "timestamp_submitted": r.timestamp_submitted,
                "ticker":              r.ticker,
                "direction":           r.direction,
                "strategy":            r.strategy,
                "qty":                 r.qty,
                "entry_price":         r.entry_price,
                "stop_loss_price":     r.stop_loss_price,
                "position_size_usd":   r.position_size_usd,
                "confidence":          r.confidence,
                "order_id":            r.order_id,
                "status":              r.status,
                "mode":                r.mode,
                "error":               r.error,
                "approved":            r.approved,
                "block_reason":        r.block_reason,
            }
            for r in results
        ]
        df = pd.DataFrame(rows)
        try:
            self.store.save_audit(df)
        except Exception as exc:
            logger.error(f"[OrderExecutor] Failed to save audit log: {exc}")
