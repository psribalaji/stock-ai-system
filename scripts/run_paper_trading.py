"""
scripts/run_paper_trading.py — Start the paper trading system.

Wires together:
  TradingScheduler → DecisionEngine → OrderExecutor → Alpaca (paper)

What runs automatically:
  Every 5 min (market hours):  signal pipeline → approved decisions → paper orders
  5am daily:                   fetch latest OHLCV from Polygon
  Sunday 8pm:                  ModelMonitor drift check
  Quarterly:                   recalibrate strategy baselines

Usage:
    source .venv/bin/activate

    # Normal mode — waits for market hours
    python scripts/run_paper_trading.py

    # Dry-run mode — fires pipeline ONCE immediately, prints decisions, exits
    # Orders are logged but NOT submitted to Alpaca (safe to run any time)
    python scripts/run_paper_trading.py --dry-run-now

Press Ctrl+C to stop (normal mode).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import get_config
from src.execution.order_executor import OrderExecutor
from src.execution.decision_engine import DecisionEngine
from src.monitoring.model_monitor import ModelMonitor
from src.ingestion.storage import ParquetStore
from src.scheduler.scheduler import TradingScheduler

console = Console()


def on_decisions(decisions) -> None:
    """Called by scheduler after each signal pipeline run."""
    approved = [d for d in decisions if d.approved]
    if not approved:
        return

    console.print(f"\n[bold cyan]── New Decisions ({len(approved)} approved) ──[/bold cyan]")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Ticker",     style="cyan")
    table.add_column("Direction",  style="green")
    table.add_column("Strategy")
    table.add_column("Confidence", justify="right")
    table.add_column("Size USD",   justify="right")
    table.add_column("Stop Loss",  justify="right")

    for d in approved:
        color = "green" if d.direction == "BUY" else "red"
        table.add_row(
            d.ticker,
            f"[{color}]{d.direction}[/{color}]",
            d.strategy,
            f"{d.confidence:.1%}",
            f"${d.position_size_usd:,.0f}",
            f"${d.stop_loss_price:.2f}",
        )
    console.print(table)


def on_drift(report) -> None:
    """Called by scheduler after weekly drift check."""
    if report.has_alerts:
        console.print(f"\n[bold yellow]⚠ Drift alerts:[/bold yellow]")
        for alert in report.alerts:
            level = "red" if alert.severity == "CRITICAL" else "yellow"
            console.print(f"  [{level}][{alert.severity}][/{level}] "
                          f"{alert.strategy}: {alert.alert_type} — {alert.message}")
    else:
        console.print("\n[green]✓ Drift check: all strategies nominal[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="StockAI paper trading runner")
    parser.add_argument(
        "--dry-run-now",
        action="store_true",
        help="Fire the signal pipeline once immediately (no orders submitted), then exit.",
    )
    args = parser.parse_args()

    cfg = get_config()

    mode_label = "[yellow]DRY-RUN (one-shot)[/yellow]" if args.dry_run_now else f"[yellow]{cfg.trading.mode.upper()}[/yellow]"
    console.print(Panel.fit(
        "[bold cyan]StockAI — Paper Trading[/bold cyan]\n"
        f"Mode: {mode_label}  |  "
        f"Universe: {', '.join(cfg.assets.all_tradeable)}\n"
        f"Signal interval: every {cfg.schedule.signal_interval_min} min (market hours)",
        border_style="cyan",
    ))

    # ── Wire up components ────────────────────────────────────────────────────
    store    = ParquetStore(cfg.data.storage_path)
    monitor  = ModelMonitor()
    # dry_run=True means orders are logged but never sent to Alpaca
    executor = OrderExecutor(dry_run=args.dry_run_now)
    engine   = DecisionEngine()

    # Record completed trades in monitor (loads from existing audit log)
    audit = store.load_audit()
    if not audit.empty:
        required = {"strategy", "ticker", "pnl_pct", "won"}
        if required.issubset(audit.columns):
            count = monitor.record_trades_from_df(audit)
            logger.info(f"[PaperTrading] Loaded {count} historical trades into monitor")

    if args.dry_run_now:
        _run_once(engine, executor, store)
        return

    # ── Start scheduler ───────────────────────────────────────────────────────
    scheduler = TradingScheduler(
        decision_engine=engine,
        monitor=monitor,
        store=store,
        executor=executor,          # enables job_position_check (trailing stops + TPs)
        on_decisions=lambda decisions: _handle_decisions(decisions, executor, on_decisions),
        on_drift=on_drift,
    )

    console.print("\n[bold]Scheduler jobs registered:[/bold]")
    for job_id in scheduler.get_jobs():
        console.print(f"  • {job_id}")

    console.print(
        "\n[green]✓ Paper trading started — press Ctrl+C to stop[/green]\n"
        "[dim]Orders will be submitted to Alpaca paper account automatically "
        "during market hours (9:30am–4:00pm ET, Mon–Fri)[/dim]\n"
    )

    scheduler.start()   # blocking — runs until Ctrl+C


def _run_once(engine: DecisionEngine, executor: OrderExecutor, store: ParquetStore) -> None:
    """
    Fire the full signal pipeline once across all tickers and print results.
    Used by --dry-run-now. Orders are logged but not submitted (executor.dry_run=True).
    """
    cfg = get_config()
    console.print("\n[bold cyan]── Dry-Run: one-shot pipeline ──[/bold cyan]")

    from src.ingestion.alpaca_client import AlpacaClient
    from src.risk.risk_manager import PortfolioState

    alpaca   = AlpacaClient()
    tickers  = cfg.assets.all_tradeable
    data_map: dict = {}
    price_map: dict = {}

    console.print(f"Loading OHLCV for {len(tickers)} tickers...")
    for ticker in tickers:
        df = store.load_ohlcv(ticker)
        if df.empty:
            console.print(f"  [yellow]⚠[/yellow]  {ticker}: no local data — skipping")
            continue
        price = alpaca.get_latest_price(ticker)
        if price is None:
            price = float(df["close"].iloc[-1])   # fall back to last close
            console.print(f"  [dim]{ticker}: live price unavailable, using last close ${price:.2f}[/dim]")
        data_map[ticker]  = df
        price_map[ticker] = price

    console.print(f"  [green]✓[/green] {len(data_map)} tickers loaded\n")

    portfolio = PortfolioState(
        total_value_usd=100_000.0,
        cash_usd=100_000.0,
        open_positions=0,
        crypto_exposure_usd=0.0,
        daily_pnl_pct=0.0,
        peak_value_usd=100_000.0,
    )

    console.print("Running signal pipeline...")
    decisions = engine.decide_all(
        tickers=list(data_map),
        data_map=data_map,
        price_map=price_map,
        portfolio=portfolio,
    )

    approved = [d for d in decisions if d.approved]
    console.print(f"  [green]✓[/green] Pipeline complete — {len(approved)} approved decision(s)\n")

    if approved:
        on_decisions(approved)
        decision_map = {d.ticker: d for d in approved}
        results = executor.execute_all(approved)
        for r in results:
            if r.status == "submitted" and r.direction == "BUY":
                d = decision_map[r.ticker]
                console.print(
                    f"  [dim]trailing_stop=${d.trailing_stop_price:.2f}  "
                    f"take_profit=${d.take_profit_price:.2f}  "
                    f"ATR=${d.trailing_stop_atr:.2f}[/dim]"
                )
    else:
        console.print("  [dim]No signals passed confidence + risk checks this run.[/dim]")
        console.print("  [dim]This is normal — try again during high-volume market hours.[/dim]")


def _handle_decisions(decisions, executor: OrderExecutor, display_callback) -> None:
    """Submit approved decisions as paper orders, then display results."""
    approved = [d for d in decisions if d.approved]
    if not approved:
        logger.info("[PaperTrading] Pipeline ran — no approved decisions this cycle")
        return

    # Display signal summary
    display_callback(approved)

    # Submit to Alpaca paper account
    # Build a lookup so we can register stops after submission
    decision_map = {d.ticker: d for d in approved}
    results = executor.execute_all(approved)

    # Register trailing stops + take-profits for every submitted BUY so that
    # the scheduler's 5-min job_position_check loop has something to track.
    for r in results:
        if r.status == "submitted" and r.direction == "BUY":
            d = decision_map[r.ticker]
            executor.register_trailing_stop(r.ticker, r.entry_price, d.trailing_stop_atr)
            executor.register_take_profit(r.ticker, d.take_profit_price)

    submitted = [r for r in results if r.status == "submitted"]
    skipped   = [r for r in results if r.status == "skipped"]
    failed    = [r for r in results if r.status == "failed"]

    console.print(
        f"[bold]Orders:[/bold] "
        f"[green]{len(submitted)} submitted[/green]  "
        f"[yellow]{len(skipped)} skipped[/yellow]  "
        f"[red]{len(failed)} failed[/red]"
    )

    for r in submitted:
        console.print(
            f"  [green]✓[/green] {r.direction} {r.qty:.2f}x {r.ticker} "
            f"@ ~${r.entry_price:.2f}  stop=${r.stop_loss_price:.2f}  "
            f"order_id={r.order_id}"
        )
    for r in failed:
        console.print(f"  [red]✗[/red] {r.ticker}: {r.error}")


if __name__ == "__main__":
    main()
