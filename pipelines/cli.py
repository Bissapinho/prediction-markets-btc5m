"""Command-line interface `pm`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

from pipelines.common.config import load_config
from pipelines.common.log import setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)
ingest_app = typer.Typer(no_args_is_help=True, help="Download of the historical data.")
build_app = typer.Typer(no_args_is_help=True, help="Rebuild of the clean / features layers.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(build_app, name="build")

_state: dict = {}


@app.callback()
def main(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Configuration file."),
    log_level: str = typer.Option("INFO", "--log-level"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file (rotated daily)."),
) -> None:
    setup_logging(log_level, log_file)
    _state["cfg"] = load_config(config)


@ingest_app.command("markets")
def ingest_markets(
    n: Optional[int] = typer.Option(None, "--n", help="Number of markets (default: config n_markets)."),
    end_ts: Optional[int] = typer.Option(None, "--end-ts", help="Start (Unix s) of the most recent market."),
    force: bool = typer.Option(False, "--force", help="Redo the windows already ok."),
) -> None:
    from pipelines.markets import ingest_markets as run

    counts = asyncio.run(run(_state["cfg"], n_markets=n, end_ts=end_ts, force=force))
    typer.echo(" ".join(f"{k}={v}" for k, v in counts.items()))


@ingest_app.command("trades")
def ingest_trades(
    force: bool = typer.Option(False, "--force", help="Redo the markets already ok."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum number of markets (testing)."),
) -> None:
    from pipelines.trades import ingest_trades as run

    counts = asyncio.run(run(_state["cfg"], force=force, limit=limit))
    typer.echo(" ".join(f"{k}={v}" for k, v in counts.items()))


@ingest_app.command("binance")
def ingest_binance(
    start: Optional[str] = typer.Option(None, "--start", help="YYYY-MM-DD (default: market period minus margin)."),
    end: Optional[str] = typer.Option(None, "--end", help="YYYY-MM-DD (default: market period plus margin, capped at yesterday)."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    from datetime import date

    from pipelines.binance import ingest_binance as run

    counts = asyncio.run(run(_state["cfg"], start=date.fromisoformat(start) if start else None,
                             end=date.fromisoformat(end) if end else None, force=force))
    typer.echo(" ".join(f"{k}={v}" for k, v in counts.items()))


@build_app.command("clean")
def build_clean() -> None:
    from pipelines.common.storage import Layout
    from pipelines.markets import build_markets_clean
    from pipelines.trades import build_trades_clean

    cfg = _state["cfg"]
    layout = Layout(cfg.storage.root)
    df = build_markets_clean(cfg, layout)
    typer.echo(f"markets: {df.height} rows")
    typer.echo(f"trades: {build_trades_clean(cfg, layout)} rows")


@build_app.command("features")
def build_features() -> None:
    from pipelines.common.storage import Layout
    from pipelines.features import build_features as run

    cfg = _state["cfg"]
    for name, n in run(cfg, Layout(cfg.storage.root)).items():
        typer.echo(f"{name}: {n} rows")


@app.command("qa")
def qa() -> None:
    """Quality checks -> reports/qa_YYYYMMDD.md."""
    from pipelines.common.storage import Layout
    from pipelines.qa import run_qa

    cfg = _state["cfg"]
    typer.echo(str(run_qa(cfg, Layout(cfg.storage.root))))



if __name__ == "__main__":
    app()
