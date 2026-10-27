"""Pipeline `binance`: daily BTCUSDT spot aggTrades from data.binance.vision.

- raw:      `raw/binance/aggtrades/date=YYYY-MM-DD/BTCUSDT-aggTrades-YYYY-MM-DD.zip` (original archive, checksum verified)
- clean:    `clean/btc_aggtrades/date=YYYY-MM-DD/aggtrades.parquet`
- manifest: one status per day: ok, not_available (not published yet), checksum_error, error

Format (checked on 2026-10-18): CSV without header, timestamps in microseconds,
columns agg_id, price, qty, first_trade_id, last_trade_id, transact_time, is_buyer_maker, is_best_match.
The file of a day is published the next day.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import time
import zipfile
from datetime import date, datetime, timedelta, timezone

import polars as pl

from pipelines.common.config import Config
from pipelines.common.http import HttpClient, HttpError
from pipelines.common.log import get_logger, log
from pipelines.common.manifest import Manifest
from pipelines.common.storage import Layout, write_bytes, write_parquet

logger = get_logger(__name__)

PIPELINE = "binance"

CSV_COLUMNS = ["agg_id", "price", "qty", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"]
CSV_SCHEMA = {
    "agg_id": pl.Int64, "price": pl.Float64, "qty": pl.Float64, "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64, "transact_time": pl.Int64, "is_buyer_maker": pl.Utf8, "is_best_match": pl.Utf8,
}


def period_from_markets(cfg: Config, layout: Layout) -> tuple[date, date]:
    """Days covering all markets, plus `margin_days` on each side, capped at yesterday (UTC)."""
    mk = pl.read_parquet(layout.clean("markets", "markets.parquet"), columns=["start_ts_ms", "end_ts_ms"])
    lo = datetime.fromtimestamp(mk["start_ts_ms"].min() / 1000, tz=timezone.utc).date()
    hi = datetime.fromtimestamp(mk["end_ts_ms"].max() / 1000, tz=timezone.utc).date()
    m = timedelta(days=cfg.binance.margin_days)
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    return lo - m, min(hi + m, yesterday)


def parse_aggtrades_csv(data: bytes) -> pl.DataFrame:
    """aggTrades CSV -> typed DataFrame with ts_ms (UTC). Accepts a header and timestamps in ms or µs."""
    has_header = not data[:1].isdigit()
    df = pl.read_csv(io.BytesIO(data), has_header=has_header, new_columns=CSV_COLUMNS, schema_overrides=CSV_SCHEMA)
    # Binance spot moved to microseconds in 2025; older files are in ms.
    unit_us = df["transact_time"].max() > 10**14
    ts_ms = (pl.col("transact_time") // 1000) if unit_us else pl.col("transact_time")
    return df.select(
        "agg_id",
        "price",
        "qty",
        "first_trade_id",
        "last_trade_id",
        ts_ms.alias("ts_ms"),
        (pl.col("transact_time") if unit_us else pl.col("transact_time") * 1000).alias("ts_us"),
        (pl.col("is_buyer_maker").str.to_lowercase() == "true").alias("is_buyer_maker"),
    ).sort("agg_id")


async def _one_day(client: HttpClient, cfg: Config, layout: Layout, manifest: Manifest, d: date) -> str:
    sym = cfg.binance.symbol
    name = f"{sym}-aggTrades-{d.isoformat()}.zip"
    path = f"/data/spot/daily/aggTrades/{sym}/{name}"
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    t0 = time.monotonic()
    try:
        checksum_txt = (await client.get(path + ".CHECKSUM")).text
        zip_bytes = (await client.get(path)).content
    except HttpError as e:
        if e.status == 404:
            manifest.record(d.isoformat(), "not_available", started_at=started, source="binance_vision")
            return "not_available"
        manifest.record(d.isoformat(), "error", started_at=started, error=repr(e)[:500], source="binance_vision")
        return "error"
    except Exception as e:  # noqa: BLE001
        manifest.record(d.isoformat(), "error", started_at=started, error=repr(e)[:500], source="binance_vision")
        log(logger, "day error", logging.ERROR, pipeline=PIPELINE, unit=d.isoformat(), error=repr(e)[:300])
        return "error"

    expected = checksum_txt.split()[0].lower()
    actual = hashlib.sha256(zip_bytes).hexdigest()
    if actual != expected:
        manifest.record(d.isoformat(), "checksum_error", started_at=started, source="binance_vision",
                        error=f"sha256 {actual} != {expected}")
        log(logger, "checksum error", logging.ERROR, pipeline=PIPELINE, unit=d.isoformat())
        return "checksum_error"

    write_bytes(layout.raw("binance", "aggtrades", f"date={d.isoformat()}", name), zip_bytes)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        df = parse_aggtrades_csv(z.read(z.namelist()[0]))
    write_parquet(df, layout.clean("btc_aggtrades", f"date={d.isoformat()}", "aggtrades.parquet"))
    manifest.record(d.isoformat(), "ok", n_rows=df.height, started_at=started, source="binance_vision", sha256=actual)
    log(logger, "day done", pipeline=PIPELINE, unit=d.isoformat(), n_rows=df.height,
        mb=round(len(zip_bytes) / 1e6, 1), duration_s=round(time.monotonic() - t0, 1))
    return "ok"


async def ingest_binance(cfg: Config, start: date | None = None, end: date | None = None, force: bool = False) -> dict[str, int]:
    layout = Layout(cfg.storage.root)
    if start is None or end is None:
        p_start, p_end = period_from_markets(cfg, layout)
        start, end = start or p_start, end or p_end
    manifest = Manifest(layout, PIPELINE)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if force or manifest.status(d.isoformat()) != "ok"]
    log(logger, "binance start", pipeline=PIPELINE, start=start.isoformat(), end=end.isoformat(),
        n_days=len(days), n_todo=len(todo))
    counts = {"ok": 0, "not_available": 0, "checksum_error": 0, "error": 0}
    sem = asyncio.Semaphore(cfg.binance.concurrency)
    async with HttpClient(cfg.binance.base_url, rate_per_s=10, timeout_s=cfg.binance.timeout_s,
                          max_retries=cfg.binance.max_retries) as client:

        async def run(d: date) -> None:
            async with sem:
                counts[await _one_day(client, cfg, layout, manifest, d)] += 1

        await asyncio.gather(*(run(d) for d in todo))
    log(logger, "binance done", pipeline=PIPELINE, **counts)
    return counts
