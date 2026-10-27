"""Loads the analysis data as numpy arrays (N markets × seconds), ordered by market start."""

from __future__ import annotations

from dataclasses import dataclass

import os

import duckdb
import numpy as np

from pipelines.common.storage import Layout

T = 300           # seconds per window
T_EXT = 310       # extended grid for the fills at t + L
ASK_MAX_AGE = 3   # an ask is valid only if it is ≤ 3 s old


@dataclass
class Data:
    market_id: np.ndarray      # (N,)
    start_s: np.ndarray        # (N,) int64
    day: np.ndarray            # (N,) str YYYY-MM-DD
    y: np.ndarray              # (N,) bool, outcome Up
    price_start: np.ndarray    # (N,) Chainlink
    price_end: np.ndarray
    p_implied: np.ndarray      # (N, T) p_up_implied from market_1s
    n_trades: np.ndarray       # (N, T) Up + Down trades in the second
    ask_up: np.ndarray         # (N, T_EXT) NaN if no taker Up buy in the last ASK_MAX_AGE s
    ask_dn: np.ndarray
    btc_ts0: int               # ts_s of the first element of btc_close
    btc_close: np.ndarray      # (n_s,) Binance 1 s close, contiguous

    @property
    def n(self) -> int:
        return len(self.market_id)


def _f(x) -> np.ndarray:
    """DuckDB column -> float64, NULL -> NaN (fetchnumpy returns masked arrays)."""
    return np.ma.filled(np.ma.asarray(x).astype(np.float64), np.nan)


def _p(path) -> str:
    return str(path).replace("\\", "/")


# First day analysed (PM_START). Before ~2026-08-10 the markets resolved on the Chainlink spot price, not on the
# 60 s TWAP: the fair-value model only applies afterwards (docs/data_notes.md).
START = os.environ.get("PM_START", "2026-08-17")
# Simulated buy price: "prudent" (default, mirror-book floor) or any other value (last taker buy only, optimistic).
ASK_MODE = os.environ.get("PM_ASK", "prudent")


def load(layout: Layout) -> Data:
    con = duckdb.connect()
    markets = _p(layout.clean("markets", "markets.parquet"))
    trades = _p(layout.clean("trades")) + "/*/*.parquet"
    m1s = _p(layout.features("market_1s")) + "/*/*.parquet"
    btc = _p(layout.features("btc_1s")) + "/*/*.parquet"

    b = con.execute(f"SELECT ts_s, close FROM read_parquet('{btc}', hive_partitioning = false) ORDER BY ts_s").fetchnumpy()
    btc_ts0, btc_close = int(b["ts_s"][0]), b["close"].astype(np.float64)
    assert len(btc_close) == int(b["ts_s"][-1]) - btc_ts0 + 1, "btc_1s must be contiguous"
    btc_end = btc_ts0 + len(btc_close) - 1

    # Markets kept: resolved, ≥ 1 trade in the window, Binance from start − 1800 to start + 300.
    con.execute(f"""
        CREATE TEMP TABLE mk AS
        WITH n AS (
            SELECT market_id, count(*) n FROM read_parquet('{trades}', hive_partitioning = false)
            WHERE t_in_window_ms >= 0 AND t_in_window_ms < {T * 1000} GROUP BY 1
        )
        SELECT m.market_id, m.start_ts_ms // 1000 AS start_s, m.outcome = 'up' AS y, m.price_start, m.price_end,
               strftime(make_timestamp(m.start_ts_ms * 1000), '%Y-%m-%d') AS day
        FROM read_parquet('{markets}') m JOIN n USING (market_id)
        WHERE m.closed AND m.outcome IS NOT NULL
          AND m.start_ts_ms // 1000 - 1800 >= {btc_ts0} AND m.start_ts_ms // 1000 + {T} <= {btc_end}
          AND m.start_ts_ms >= epoch_ms(TIMESTAMP '{START}')
        ORDER BY start_s
    """)
    mk = con.execute("SELECT * FROM mk ORDER BY start_s").fetchnumpy()
    n = len(mk["market_id"])

    m = con.execute(f"""
        SELECT s.p_up_implied, s.n_trades_up + s.n_trades_down AS n_tr
        FROM mk JOIN read_parquet('{m1s}', hive_partitioning = false) s USING (market_id)
        ORDER BY mk.start_s, s.t
    """).fetchnumpy()
    p_implied = _f(m["p_up_implied"]).reshape(n, T)
    n_trades = m["n_tr"].astype(np.int32).reshape(n, T)

    # Last taker buy price per second and per outcome, then ASOF on the extended grid.
    con.execute(f"""
        CREATE TEMP TABLE buys AS
        SELECT market_id, outcome, ts_ms // 1000 AS ts_s, arg_max(price, seq) AS p
        FROM read_parquet('{trades}', hive_partitioning = false) WHERE side = 'buy' GROUP BY ALL
    """)
    a = con.execute(f"""
        WITH g AS (SELECT mk.market_id, mk.start_s, r.t, mk.start_s + r.t AS ts_s FROM mk, range(0, {T_EXT}) r(t)),
             u AS (SELECT market_id, ts_s, p FROM buys WHERE outcome = 'up'),
             d AS (SELECT market_id, ts_s, p FROM buys WHERE outcome = 'down')
        SELECT CASE WHEN g.ts_s - u.ts_s <= {ASK_MAX_AGE} THEN u.p END AS ask_up,
               CASE WHEN g.ts_s - d.ts_s <= {ASK_MAX_AGE} THEN d.p END AS ask_dn
        FROM g
        ASOF LEFT JOIN u ON g.market_id = u.market_id AND g.ts_s >= u.ts_s
        ASOF LEFT JOIN d ON g.market_id = d.market_id AND g.ts_s >= d.ts_s
        ORDER BY g.start_s, g.t
    """).fetchnumpy()
    ask_up = _f(a["ask_up"]).reshape(n, T_EXT)
    ask_dn = _f(a["ask_dn"]).reshape(n, T_EXT)
    if ASK_MODE == "prudent":
        # The Up and Down books mirror each other: a Down buy at q proves an Up bid at 1 − q, so an Up ask ≥ 1 − q.
        # Without this floor, the last Up buy can be stale and below the true ask (pair artefact, docs/data_notes.md).
        # The floor never replaces a missing ask (np.maximum keeps NaN, unlike np.fmax).
        floor_up, floor_dn = np.nan_to_num(1 - ask_dn, nan=0.0), np.nan_to_num(1 - ask_up, nan=0.0)
        ask_up, ask_dn = np.maximum(ask_up, floor_up), np.maximum(ask_dn, floor_dn)

    return Data(
        market_id=mk["market_id"], start_s=mk["start_s"].astype(np.int64), day=np.asarray(mk["day"]).astype(str),
        y=mk["y"].astype(bool), price_start=_f(mk["price_start"]), price_end=_f(mk["price_end"]),
        p_implied=p_implied, n_trades=n_trades, ask_up=ask_up, ask_dn=ask_dn, btc_ts0=btc_ts0, btc_close=btc_close,
    )
