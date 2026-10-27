"""Pipeline `markets`: market metadata from the venue's public market API.

Strategy (docs/data_notes.md):
1. List the events of the series by `endDate` windows of `window_hours`
   (`/events?series_id=..&end_date_min=..&end_date_max=..`). A 6 h window = 73 events,
   a single page; this avoids the offset cap of /events (~3000).
2. For each expected start missing from the list, fall back to `/events?slug=...`.
3. Each page is written as is under `raw/gamma/...` (never rewritten).
4. `build_markets_clean` rebuilds `clean/markets/markets.parquet` from the whole raw layer.
5. One status per market goes to the manifest: ok, missing, not_resolved, error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pipelines.common.config import Config
from pipelines.common.http import HttpClient, HttpError
from pipelines.common.log import get_logger, log
from pipelines.common.manifest import Manifest
from pipelines.common.storage import Layout, read_json_zst, write_parquet, write_zst
from pipelines.common.timeutil import floor_to, iso_to_ms, ms_to_iso, now_ms, utc_date

logger = get_logger(__name__)

PIPELINE = "markets"
FETCH_PIPELINE = "markets_fetch"
_SLUG_TS_RE = re.compile(r"-(\d{9,11})$")
_FETCHED_RE = re.compile(r"(\d{13})")

MARKETS_SCHEMA = {
    "market_id": pl.Utf8,
    "event_id": pl.Utf8,
    "slug": pl.Utf8,
    "asset": pl.Utf8,
    "duration_s": pl.Int32,
    "start_ts_ms": pl.Int64,
    "end_ts_ms": pl.Int64,
    "event_start_time_ms": pl.Int64,
    "token_up": pl.Utf8,
    "token_down": pl.Utf8,
    "outcome": pl.Utf8,
    "closed": pl.Boolean,
    "uma_resolution_status": pl.Utf8,
    "closed_time_ms": pl.Int64,
    "fee_type": pl.Utf8,
    "fee_rate": pl.Float64,
    "fee_exponent": pl.Float64,
    "fee_taker_only": pl.Boolean,
    "fee_schedule_json": pl.Utf8,
    "twap_lookback_s": pl.Int32,
    "price_start": pl.Float64,
    "price_end": pl.Float64,
    "resolution_price_source": pl.Utf8,
    "volume_usdc": pl.Float64,
    "tick_size": pl.Float64,
    "min_order_size": pl.Float64,
    "fetched_at_ms": pl.Int64,
    "ingested_at": pl.Datetime("ms", "UTC"),
}


# --------------------------------------------------------------------------- parsing


def slug_start_ts(slug: str) -> int | None:
    m = _SLUG_TS_RE.search(slug)
    return int(m.group(1)) if m else None


def _json_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        out = json.loads(v)
        return out if isinstance(out, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _float(v: Any) -> float | None:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _outcome(closed: bool, outcomes: list, prices: list) -> str | None:
    """'up' / 'down' only if the market is closed and the prices are exactly 1/0."""
    if not closed or len(outcomes) != 2 or len(prices) != 2:
        return None
    by_name = {str(o).lower(): _float(p) for o, p in zip(outcomes, prices)}
    up, down = by_name.get("up"), by_name.get("down")
    if up == 1.0 and down == 0.0:
        return "up"
    if up == 0.0 and down == 1.0:
        return "down"
    return None


def parse_event(event: dict, fetched_at_ms: int, default_asset: str, default_duration_s: int) -> list[dict]:
    """One market-API event -> one `markets` row per market (normally a single one)."""
    rows = []
    meta = event.get("eventMetadata") or {}
    price_start = _float(meta.get("priceToBeat"))
    price_end = _float(meta.get("finalPrice"))
    for m in event.get("markets") or []:
        slug = m.get("slug") or event.get("slug")
        start_ts = slug_start_ts(slug or "")
        if start_ts is None:
            continue
        crypto_cfg = m.get("cryptoMarketConfig") or {}
        duration_s = default_duration_s
        outcomes = _json_list(m.get("outcomes"))
        tokens = _json_list(m.get("clobTokenIds"))
        prices = _json_list(m.get("outcomePrices"))
        tok = {str(o).lower(): t for o, t in zip(outcomes, tokens)}
        closed = bool(m.get("closed"))
        fee = m.get("feeSchedule") or {}
        rows.append(
            {
                "market_id": m.get("conditionId"),
                "event_id": str(event.get("id")) if event.get("id") is not None else None,
                "slug": slug,
                "asset": crypto_cfg.get("asset") or default_asset,
                "duration_s": duration_s,
                # Start = slug timestamp. `startDate` is the creation date (docs/data_notes.md).
                "start_ts_ms": start_ts * 1000,
                "end_ts_ms": iso_to_ms(m.get("endDate")) or (start_ts + duration_s) * 1000,
                "event_start_time_ms": iso_to_ms(m.get("eventStartTime") or event.get("startTime")),
                "token_up": tok.get("up"),
                "token_down": tok.get("down"),
                "outcome": _outcome(closed, outcomes, prices),
                "closed": closed,
                "uma_resolution_status": m.get("umaResolutionStatus"),
                "closed_time_ms": iso_to_ms(m.get("closedTime") or event.get("closedTime")),
                "fee_type": m.get("feeType"),
                "fee_rate": _float(fee.get("rate")),
                "fee_exponent": _float(fee.get("exponent")),
                "fee_taker_only": fee.get("takerOnly"),
                "fee_schedule_json": json.dumps(fee, sort_keys=True) if fee else None,
                "twap_lookback_s": crypto_cfg.get("twapLookbackSeconds"),
                "price_start": price_start,
                "price_end": price_end,
                "resolution_price_source": "gamma" if price_start is not None and price_end is not None else None,
                "volume_usdc": _float(m.get("volumeNum") if m.get("volumeNum") is not None else m.get("volume")),
                "tick_size": _float(m.get("orderPriceMinTickSize")),
                "min_order_size": _float(m.get("orderMinSize")),
                "fetched_at_ms": fetched_at_ms,
            }
        )
    return rows


# --------------------------------------------------------------------------- raw -> clean


def _raw_event_files(layout: Layout) -> list[Path]:
    files = []
    for sub in ("events", "events_by_slug"):
        d = layout.raw("gamma", sub)
        if d.exists():
            files.extend(d.rglob("*.json.zst"))
    return files


def build_markets_clean(cfg: Config, layout: Layout) -> pl.DataFrame:
    """Rebuilds `clean/markets/markets.parquet` from the whole raw market-API layer.

    For a given slug, the most recent fetch wins.
    """
    t0 = time.monotonic()
    rows: list[dict] = []
    for path in _raw_event_files(layout):
        m = _FETCHED_RE.search(path.name)
        fetched_at_ms = int(m.group(1)) if m else 0
        payload = read_json_zst(path)
        events = payload if isinstance(payload, list) else [payload]
        for ev in events:
            rows.extend(parse_event(ev, fetched_at_ms, cfg.asset, cfg.duration_s))
    ingested_at = datetime.now(timezone.utc)
    for r in rows:
        r["ingested_at"] = ingested_at
    df = pl.DataFrame(rows, schema=MARKETS_SCHEMA)
    df = (
        df.sort("fetched_at_ms")
        .unique(subset=["slug"], keep="last")
        .filter((pl.col("asset") == cfg.asset) & (pl.col("duration_s") == cfg.duration_s))
        .sort("start_ts_ms")
    )
    write_parquet(df, layout.clean("markets", "markets.parquet"))
    log(logger, "markets clean built", pipeline=PIPELINE, n_rows=df.height, n_files=len(_raw_event_files(layout)),
        duration_s=round(time.monotonic() - t0, 2))
    return df


# --------------------------------------------------------------------------- fetch


@dataclass(frozen=True)
class Window:
    start_ms: int
    end_ms: int

    @property
    def key(self) -> str:
        return datetime.fromtimestamp(self.start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H")


def target_starts(end_ts: int, n_markets: int, duration_s: int) -> list[int]:
    """Expected start timestamps (s): end_ts - k*duration, k = 0..n-1."""
    return [end_ts - k * duration_s for k in range(n_markets)]


def windows_for(starts: list[int], duration_s: int, window_hours: int) -> list[Window]:
    """Aligned `endDate` windows covering all target markets."""
    step = window_hours * 3600
    lo = floor_to(min(starts) + duration_s, step)
    hi = max(starts) + duration_s
    return [Window(w * 1000, (w + step) * 1000) for w in range(lo, hi + 1, step)]


def default_end_ts(cfg: Config) -> int:
    """Start of the last window closed for at least `settle_margin_s`."""
    d = cfg.duration_s
    return floor_to(now_ms() // 1000 - cfg.gamma.settle_margin_s - d, d)


async def _fetch_window(client: HttpClient, cfg: Config, layout: Layout, w: Window) -> list[dict]:
    """Fetches every page of a window and writes them to the raw layer. Returns the events."""
    events: list[dict] = []
    offset, page = 0, 0
    while True:
        params = {
            "series_id": cfg.gamma.series_id,
            "end_date_min": ms_to_iso(w.start_ms),
            "end_date_max": ms_to_iso(w.end_ms),
            "order": "startTime",
            "ascending": "true",
            "limit": cfg.gamma.page_limit,
            "offset": offset,
        }
        data, raw = await client.get_json("/events", params)
        fetched = now_ms()
        date, hour = w.key.split("T")
        write_zst(layout.raw("gamma", "events", f"date={date}", f"window={hour}", f"{fetched}-p{page:02d}.json.zst"), raw)
        events.extend(data)
        if len(data) < cfg.gamma.page_limit:
            return events
        offset += cfg.gamma.page_limit
        page += 1


async def _fetch_slug(client: HttpClient, layout: Layout, slug: str, start_ts: int) -> bool:
    """Fallback by slug. Returns True if the event exists."""
    data, raw = await client.get_json("/events", {"slug": slug})
    if not data:
        return False
    write_zst(layout.raw("gamma", "events_by_slug", f"date={utc_date(start_ts * 1000)}", f"{slug}-{now_ms()}.json.zst"), raw)
    return True


async def ingest_markets(
    cfg: Config, n_markets: int | None = None, end_ts: int | None = None, force: bool = False, concurrency: int = 4
) -> dict[str, int]:
    layout = Layout(cfg.storage.root)
    d = cfg.duration_s
    n_markets = n_markets or cfg.n_markets
    end_ts = end_ts if end_ts is not None else default_end_ts(cfg)
    if end_ts % d:
        raise ValueError(f"end_ts={end_ts} is not a multiple of {d}")
    starts = target_starts(end_ts, n_markets, d)
    windows = windows_for(starts, d, cfg.gamma.window_hours)
    fetch_manifest = Manifest(layout, FETCH_PIPELINE)
    market_manifest = Manifest(layout, PIPELINE)
    settled_before_ms = now_ms() - cfg.gamma.settle_margin_s * 1000

    todo = [w for w in windows if force or fetch_manifest.status(w.key) != "ok"]
    log(logger, "markets start", pipeline=PIPELINE, n_markets=n_markets, end_ts=end_ts,
        first_start=ms_to_iso(min(starts) * 1000), n_windows=len(windows), n_windows_todo=len(todo))

    sem = asyncio.Semaphore(concurrency)
    async with HttpClient(cfg.gamma.base_url, cfg.gamma.rate_per_s, cfg.gamma.timeout_s, cfg.gamma.max_retries) as client:

        async def run_window(w: Window) -> None:
            async with sem:
                started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                t0 = time.monotonic()
                try:
                    events = await _fetch_window(client, cfg, layout, w)
                except Exception as e:  # noqa: BLE001 — a failed window does not stop the pipeline
                    fetch_manifest.record(w.key, "error", started_at=started, error=repr(e)[:500], source="gamma_series")
                    log(logger, "window error", logging.ERROR, pipeline=FETCH_PIPELINE, unit=w.key, error=repr(e)[:300])
                    return
                # A window is "ok" only if it is over and all its markets are closed;
                # otherwise it is fetched again on the next run.
                all_closed = all(m.get("closed") for ev in events for m in ev.get("markets") or [])
                status = "ok" if w.end_ms <= settled_before_ms and all_closed else "partial"
                fetch_manifest.record(w.key, status, n_rows=len(events), started_at=started, source="gamma_series")
                log(logger, "window done", pipeline=FETCH_PIPELINE, unit=w.key, status=status, n_rows=len(events),
                    duration_s=round(time.monotonic() - t0, 2))

        await asyncio.gather(*(run_window(w) for w in todo))

        # Fallback by slug for the markets expected but missing from the list.
        df = build_markets_clean(cfg, layout)
        have = set(df["slug"].to_list())
        missing = [s for s in starts if cfg.slug(s) not in have]
        prev = market_manifest.latest()
        # A market already confirmed missing by the slug fallback is not requested again (unless --force).
        retry = [
            s for s in missing
            if force or not (prev.get(cfg.slug(s), {}).get("status") == "missing"
                             and prev[cfg.slug(s)].get("source") == "gamma_slug")
        ]
        slug_errors: dict[str, str] = {}
        found_any = False
        if retry:
            log(logger, "slug fallback", pipeline=PIPELINE, n_missing=len(missing), n_retry=len(retry))

            async def run_slug(s: int) -> None:
                nonlocal found_any
                async with sem:
                    try:
                        found_any |= await _fetch_slug(client, layout, cfg.slug(s), s)
                    except HttpError as e:
                        if e.status != 404:
                            slug_errors[cfg.slug(s)] = repr(e)[:500]
                    except Exception as e:  # noqa: BLE001
                        slug_errors[cfg.slug(s)] = repr(e)[:500]

            await asyncio.gather(*(run_slug(s) for s in retry))
            if found_any:
                df = build_markets_clean(cfg, layout)

    # Status of each target market.
    by_slug = {r["slug"]: r for r in df.select("slug", "closed", "outcome", "token_up", "token_down").iter_rows(named=True)}
    records, counts = [], {"ok": 0, "missing": 0, "not_resolved": 0, "error": 0}
    retried = {cfg.slug(s) for s in retry}
    for s in starts:
        slug = cfg.slug(s)
        r = by_slug.get(slug)
        if slug in slug_errors:
            status, err, source = "error", slug_errors[slug], "gamma_slug"
        elif r is None:
            status, err = "missing", None
            source = "gamma_slug" if slug in retried else prev.get(slug, {}).get("source", "gamma_series")
        elif not r["token_up"] or not r["token_down"]:
            status, err, source = "error", "missing token ids", "gamma"
        elif r["closed"] and r["outcome"] is not None:
            status, err, source = "ok", None, "gamma"
        else:
            status, err, source = "not_resolved", None, "gamma"
        counts[status] += 1
        if market_manifest.status(slug) != status or err:
            records.append({"unit_key": slug, "status": status, "n_rows": int(r is not None), "error": err, "source": source})
    market_manifest.record_many(records)
    log(logger, "markets done", pipeline=PIPELINE, **counts, n_manifest_updates=len(records))
    return counts
