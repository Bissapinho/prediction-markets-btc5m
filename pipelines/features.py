"""Analysis tables `btc_1s` and `market_1s`, built with DuckDB from the clean layer.

Conventions (docs/data_notes.md):
- A second `ts_s` covers [ts_s, ts_s + 1). "last" values are the state **at the end** of the second.
- `btc_1s`: 1-second Binance bars. Seconds without a trade carry the previous close (`is_filled`).
- `market_1s`: one row per market and per second t = 0..299, with ts_s = start + t.
  * `p_up_last` / `p_down_last`: price of the last (taker) Up / Down trade at or before ts_s,
    including trades before the open (ASOF). Within a second, the order comes from the trade-history API (`seq`).
  * `p_up_implied`: the more recent of the two, expressed as Up (p_up_last, or 1 − p_down_last).
  * `up_age_s` / `down_age_s`: age of the last trade in seconds (0 = trade within the second).
  * `btc_close`: Binance close of the second; `btc_s0`: close of the second start − 1;
    `btc_ret_from_start` = ln(btc_close / btc_s0).
"""

from __future__ import annotations

import shutil
import time

import duckdb

from pipelines.common.config import Config
from pipelines.common.log import get_logger, log
from pipelines.common.storage import Layout

logger = get_logger(__name__)


def _posix(p) -> str:
    return str(p).replace("\\", "/")


def _copy_partitioned(con: duckdb.DuckDBPyConnection, query: str, out_dir, partition_col: str = "date") -> None:
    """Writes the result of a query as partitioned (Hive) Parquet, replacing the whole table."""
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY ({query}) TO '{_posix(tmp)}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY ({partition_col}), FILENAME_PATTERN 'part_{{i}}')"
    )
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)


def build_btc_1s(cfg: Config, layout: Layout, con: duckdb.DuckDBPyConnection | None = None) -> int:
    t0 = time.monotonic()
    con = con or duckdb.connect()
    src = _posix(layout.clean("btc_aggtrades")) + "/*/*.parquet"
    query = f"""
        WITH agg AS (
            SELECT ts_ms // 1000 AS ts_s,
                   arg_min(price, agg_id) AS open, max(price) AS high, min(price) AS low,
                   arg_max(price, agg_id) AS close,
                   sum(price * qty) / sum(qty) AS vwap,
                   sum(qty) AS volume_btc, count(*)::INTEGER AS n_trades
            FROM read_parquet('{src}')
            GROUP BY 1
        ),
        grid AS (
            SELECT unnest(generate_series((SELECT min(ts_s) FROM agg), (SELECT max(ts_s) FROM agg))) AS ts_s
        ),
        j AS (
            SELECT g.ts_s, a.open, a.high, a.low, a.close, a.vwap, a.volume_btc, a.n_trades,
                   last_value(a.close IGNORE NULLS) OVER (ORDER BY g.ts_s) AS close_ff
            FROM grid g LEFT JOIN agg a USING (ts_s)
        )
        SELECT ts_s,
               coalesce(open, close_ff) AS open, coalesce(high, close_ff) AS high,
               coalesce(low, close_ff) AS low, close_ff AS close, coalesce(vwap, close_ff) AS vwap,
               coalesce(volume_btc, 0) AS volume_btc, coalesce(n_trades, 0) AS n_trades,
               n_trades IS NULL AS is_filled,
               strftime(make_timestamp(ts_s * 1000000), '%Y-%m-%d') AS date
        FROM j ORDER BY ts_s
    """
    _copy_partitioned(con, query, layout.features("btc_1s"))
    n = con.execute(f"SELECT count(*) FROM read_parquet('{_posix(layout.features('btc_1s'))}/*/*.parquet')").fetchone()[0]
    log(logger, "btc_1s built", pipeline="features", n_rows=n, duration_s=round(time.monotonic() - t0, 1))
    return n


def build_market_1s(cfg: Config, layout: Layout, con: duckdb.DuckDBPyConnection | None = None) -> int:
    t0 = time.monotonic()
    con = con or duckdb.connect()
    d = cfg.duration_s
    markets = _posix(layout.clean("markets", "markets.parquet"))
    trades = _posix(layout.clean("trades")) + "/*/*.parquet"
    btc = _posix(layout.features("btc_1s")) + "/*/*.parquet"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mk AS
        SELECT market_id, start_ts_ms // 1000 AS start_s, outcome = 'up' AS outcome_up
        FROM read_parquet('{markets}') WHERE closed AND outcome IS NOT NULL
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sec AS
        SELECT market_id, outcome, ts_ms // 1000 AS ts_s,
               arg_max(price, seq) AS p_last, sum(size_shares) AS vol, count(*)::INTEGER AS n
        FROM read_parquet('{trades}', hive_partitioning = false)
        GROUP BY ALL
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE grid AS
        SELECT mk.market_id, r.t::INTEGER AS t, mk.start_s + r.t AS ts_s, mk.start_s, mk.outcome_up
        FROM mk, range(0, {d}) r(t)
    """)
    query = f"""
        WITH up AS (SELECT market_id, ts_s, p_last, vol, n FROM sec WHERE outcome = 'up'),
             dn AS (SELECT market_id, ts_s, p_last, vol, n FROM sec WHERE outcome = 'down'),
             btc AS (SELECT ts_s, close FROM read_parquet('{btc}', hive_partitioning = false)),
             j AS (
                SELECT g.*,
                       u.p_last AS p_up_last, g.ts_s - u.ts_s AS up_age_s,
                       w.p_last AS p_down_last, g.ts_s - w.ts_s AS down_age_s
                FROM grid g
                ASOF LEFT JOIN up u ON g.market_id = u.market_id AND g.ts_s >= u.ts_s
                ASOF LEFT JOIN dn w ON g.market_id = w.market_id AND g.ts_s >= w.ts_s
             )
        SELECT j.market_id, j.t, j.ts_s,
               j.p_up_last, j.p_down_last,
               CASE WHEN j.p_down_last IS NULL OR j.up_age_s <= j.down_age_s THEN j.p_up_last
                    ELSE 1 - j.p_down_last END AS p_up_implied,
               j.up_age_s::INTEGER AS up_age_s, j.down_age_s::INTEGER AS down_age_s,
               coalesce(u2.vol, 0) AS vol_up, coalesce(d2.vol, 0) AS vol_down,
               coalesce(u2.n, 0) AS n_trades_up, coalesce(d2.n, 0) AS n_trades_down,
               b.close AS btc_close, b0.close AS btc_s0,
               ln(b.close / b0.close) AS btc_ret_from_start,
               ({d} - j.t)::INTEGER AS tau_s,
               j.outcome_up,
               strftime(make_timestamp(j.start_s * 1000000), '%Y-%m-%d') AS date
        FROM j
        LEFT JOIN up u2 ON u2.market_id = j.market_id AND u2.ts_s = j.ts_s
        LEFT JOIN dn d2 ON d2.market_id = j.market_id AND d2.ts_s = j.ts_s
        LEFT JOIN btc b ON b.ts_s = j.ts_s
        LEFT JOIN btc b0 ON b0.ts_s = j.start_s - 1
        ORDER BY j.market_id, j.t
    """
    _copy_partitioned(con, query, layout.features("market_1s"))
    n = con.execute(f"SELECT count(*) FROM read_parquet('{_posix(layout.features('market_1s'))}/*/*.parquet')").fetchone()[0]
    log(logger, "market_1s built", pipeline="features", n_rows=n, duration_s=round(time.monotonic() - t0, 1))
    return n


def build_features(cfg: Config, layout: Layout) -> dict[str, int]:
    con = duckdb.connect()
    return {"btc_1s": build_btc_1s(cfg, layout, con), "market_1s": build_market_1s(cfg, layout, con)}
