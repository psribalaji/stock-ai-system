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
    python scripts/run_paper_trading.py

Press Ctrl+C to stop.
"""
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
    cfg = get_config()

    console.print(Panel.fit(
        "[bold cyan]StockAI — Paper Trading[/bold cyan]\n"
        f"Mode: [yellow]{cfg.trading.mode.upper()}[/yellow]  |  "
        f"Universe: {', '.join(cfg.assets.all_tradeable)}\n"
        f"Signal interval: every {cfg.schedule.signal_interval_min} min (market hours)",
        border_style="cyan",
    ))

    # ── Wire up components ────────────────────────────────────────────────────
    store    = ParquetStore(cfg.data.storage_path)
    monitor  = ModelMonitor()
    executor = OrderExecutor()          # reads config.trading.mode → paper
    engine   = DecisionEngine()

    # Record completed trades in monitor (loads from existing audit log)
    audit = store.load_audit()
    if not audit.empty:
        required = {"strategy", "ticker", "pnl_pct", "won"}
        if required.issubset(audit.columns):
            count = monitor.record_trades_from_df(audit)
            logger.info(f"[PaperTrading] Loaded {count} historical trades into monitor")

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
