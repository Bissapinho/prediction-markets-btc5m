"""Pipeline `trades` (spec 3.2), source Data API (S4) validée on-chain (docs/data_notes.md).

Le subgraph Goldsky (S2) est hors service. La Data API en mode taker
(`/trades?market=<conditionId>&limit=10000`, `takerOnly=true` par défaut) donne une ligne par match,
identique aux fills on-chain (prix, taille, timestamp de bloc), en une requête par marché.

- raw   : `raw/dataapi/trades/date=YYYY-MM-DD/<slug>-<fetched_ms>.json.zst` (réponse brute)
- clean : `clean/trades/date=YYYY-MM-DD/trades.parquet`, trié par (market_id, ts_ms, seq)
- manifeste : un statut par marché : ok, empty, truncated, error

La date de partition est la date UTC du début du marché, pas celle du trade.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pipelines.common.config import Config
from pipelines.common.http import HttpClient
from pipelines.common.log import get_logger, log
from pipelines.common.manifest import Manifest
from pipelines.common.storage import Layout, read_json_zst, write_parquet, write_zst
from pipelines.common.timeutil import now_ms, utc_date

logger = get_logger(__name__)

PIPELINE = "trades"
_FILE_RE = re.compile(r"^(?P<slug>.+)-(?P<fetched>\d{13})\.json\.zst$")

TRADES_SCHEMA = {
    "trade_key": pl.Utf8,
    "tx_hash": pl.Utf8,
    "seq": pl.Int32,
    "ts_ms": pl.Int64,
    "market_id": pl.Utf8,
    "token_id": pl.Utf8,
    "outcome": pl.Utf8,
    "side": pl.Utf8,
    "price": pl.Float64,
    "size_shares": pl.Float64,
    "notional_usdc": pl.Float64,
    "fee_usdc": pl.Float64,
    "taker": pl.Utf8,
    "t_in_window_ms": pl.Int64,
}


def fee_usdc(price: float, shares: float, rate: float, exponent: float) -> float:
    """Frais taker (docs/data_notes.md), vérifié on-chain) : rate · (p(1−p))^exponent · shares."""
    return rate * (price * (1.0 - price)) ** exponent * shares


def trade_key(tx: str, token: str, wallet: str, side: str, size: float, price: float, ts: int) -> str:
    """Clé unique de substitution : la Data API ne donne pas le log_index."""
    raw = f"{tx}|{token}|{wallet}|{side}|{size:.6f}|{price:.10f}|{ts}"
    return hashlib.blake2b(raw.encode(), digest_size=12).hexdigest()


# --------------------------------------------------------------------------- fetch


def _markets_to_fetch(cfg: Config, layout: Layout, force: bool) -> pl.DataFrame:
    """Marchés résolus de la table `markets` qui ne sont pas encore `ok` dans le manifeste."""
    path = layout.clean("markets", "markets.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer `pm ingest markets` d'abord")
    mk = pl.read_parquet(path).filter(pl.col("closed") & pl.col("outcome").is_not_null())
    if force:
        return mk
    done = {k for k, r in Manifest(layout, PIPELINE).latest().items() if r["status"] in ("ok", "empty")}
    return mk.filter(~pl.col("slug").is_in(list(done)))


async def ingest_trades(cfg: Config, force: bool = False, limit: int | None = None) -> dict[str, int]:
    layout = Layout(cfg.storage.root)
    manifest = Manifest(layout, PIPELINE)
    todo = _markets_to_fetch(cfg, layout, force).sort("start_ts_ms", descending=True)
    if limit:
        todo = todo.head(limit)
    dc = cfg.data_api
    counts = {"ok": 0, "empty": 0, "truncated": 0, "error": 0}
    n_total = todo.height
    log(logger, "trades start", pipeline=PIPELINE, n_markets=n_total, concurrency=dc.concurrency, rate_per_s=dc.rate_per_s)
    t_start = time.monotonic()
    sem = asyncio.Semaphore(dc.concurrency)
    done = 0

    async with HttpClient(dc.base_url, dc.rate_per_s, dc.timeout_s, dc.max_retries) as client:

        async def one(m: dict) -> None:
            nonlocal done
            async with sem:
                started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                try:
                    data, raw = await client.get_json("/trades", {"market": m["market_id"], "limit": dc.page_limit})
                    if not isinstance(data, list):
                        raise ValueError(f"réponse inattendue : {str(data)[:200]}")
                    write_zst(layout.raw("dataapi", "trades", f"date={utc_date(m['start_ts_ms'])}",
                                         f"{m['slug']}-{now_ms()}.json.zst"), raw)
                    n = len(data)
                    # Au plafond de page, des trades anciens peuvent manquer (offset max 10 000 : pas de pagination possible).
                    status = "truncated" if n >= dc.page_limit else ("ok" if n else "empty")
                    first_ts = min((x["timestamp"] for x in data), default=None)
                    manifest.record(m["slug"], status, n_rows=n, started_at=started, source="data_api",
                                    first_trade_ts=first_ts, market_id=m["market_id"])
                except Exception as e:  # noqa: BLE001 — un marché en échec ne stoppe pas le pipeline
                    status = "error"
                    manifest.record(m["slug"], status, started_at=started, error=repr(e)[:500], source="data_api",
                                    market_id=m["market_id"])
                    log(logger, "market error", logging.ERROR, pipeline=PIPELINE, unit=m["slug"], error=repr(e)[:300])
                counts[status] += 1
                done += 1
                if done % 500 == 0 or done == n_total:
                    elapsed = time.monotonic() - t_start
                    log(logger, "progress", pipeline=PIPELINE, done=done, total=n_total,
                        req_per_s=round(done / elapsed, 1), eta_s=round((n_total - done) * elapsed / done), **counts)

        await asyncio.gather(*(one(m) for m in todo.iter_rows(named=True)))

    log(logger, "trades done", pipeline=PIPELINE, duration_s=round(time.monotonic() - t_start, 1), **counts)
    return counts


# --------------------------------------------------------------------------- raw -> clean


def _latest_raw_files(date_dir: Path) -> list[Path]:
    """Un fichier par marché : la récupération la plus récente."""
    best: dict[str, tuple[int, Path]] = {}
    for p in date_dir.glob("*.json.zst"):
        m = _FILE_RE.match(p.name)
        if m and (m["slug"] not in best or int(m["fetched"]) > best[m["slug"]][0]):
            best[m["slug"]] = (int(m["fetched"]), p)
    return [p for _, p in best.values()]


def build_trades_clean(cfg: Config, layout: Layout) -> int:
    """Reconstruit `clean/trades` à partir du raw, une partition par date."""
    t0 = time.monotonic()
    mk = pl.read_parquet(layout.clean("markets", "markets.parquet"))
    by_slug = {r["slug"]: r for r in mk.iter_rows(named=True)}
    root = layout.raw("dataapi", "trades")
    total = 0
    for date_dir in sorted(root.glob("date=*")) if root.exists() else []:
        cols: dict[str, list] = defaultdict(list)
        for path in _latest_raw_files(date_dir):
            slug = _FILE_RE.match(path.name)["slug"]
            m = by_slug.get(slug)
            if m is None:
                continue
            rows = read_json_zst(path)
            n = len(rows)
            rate, expo = m["fee_rate"] or 0.0, m["fee_exponent"] or 1.0
            for i, x in enumerate(rows):
                ts_ms = int(x["timestamp"]) * 1000
                price, size = float(x["price"]), float(x["size"])
                cols["trade_key"].append(trade_key(x["transactionHash"], x["asset"], x["proxyWallet"], x["side"], size, price, x["timestamp"]))
                cols["tx_hash"].append(x["transactionHash"])
                # L'API renvoie du plus récent au plus ancien : seq croissant = ordre chronologique.
                cols["seq"].append(n - 1 - i)
                cols["ts_ms"].append(ts_ms)
                cols["market_id"].append(m["market_id"])
                cols["token_id"].append(x["asset"])
                cols["outcome"].append(str(x["outcome"]).lower())
                cols["side"].append(str(x["side"]).lower())
                cols["price"].append(price)
                cols["size_shares"].append(size)
                cols["notional_usdc"].append(price * size)
                cols["fee_usdc"].append(fee_usdc(price, size, rate, expo))
                cols["taker"].append(x["proxyWallet"])
                cols["t_in_window_ms"].append(ts_ms - m["start_ts_ms"])
        if not cols:
            continue
        df = (
            pl.DataFrame(cols, schema=TRADES_SCHEMA)
            .unique(subset=["trade_key"], keep="first", maintain_order=True)
            .sort(["market_id", "ts_ms", "seq"])
        )
        write_parquet(df, layout.clean("trades", date_dir.name, "trades.parquet"))
        total += df.height
    log(logger, "trades clean built", pipeline=PIPELINE, n_rows=total, duration_s=round(time.monotonic() - t0, 1))
    return total
