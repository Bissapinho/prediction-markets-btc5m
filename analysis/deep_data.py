"""Data set of the per-second analyses (results/deep_cache.npz).

Every market series is re-aligned by DELTA = 2 s (block timestamp → execution time, docs/data_notes.md).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from analysis.data import T, load
from analysis.run import align, market_price, model_inputs, p_model, sigma_pre
from pipelines.common.storage import Layout

DELTA = 2
CACHE = Path("results/deep_cache.npz")


def _p(path) -> str:
    return str(path).replace("\\", "/")


def build(k: float = 1.15, tau: float = 4e-5) -> dict:
    layout = Layout("data")
    d = align(load(layout), DELTA)
    n = d.n
    sigma = sigma_pre(d)
    m, V, S, A0 = model_inputs(d)
    P = p_model(m, V, sigma, k, tau)
    pm = market_price(d)

    con = duckdb.connect()
    con.register("mk", {"market_id": d.market_id, "start_s": d.start_s, "i": np.arange(n)})
    trades = _p(layout.clean("trades")) + "/*/*.parquet"
    # Taker flow per re-aligned second: + = pushes towards Up (Up buy or Down sell), − = towards Down.
    f = con.execute(f"""
        WITH t AS (
            SELECT mk.i, (tr.ts_ms // 1000 - mk.start_s - {DELTA})::INTEGER AS s, tr.outcome, tr.side,
                   tr.size_shares AS q, tr.price AS p
            FROM read_parquet('{trades}', hive_partitioning = false) tr JOIN mk USING (market_id)
        )
        SELECT i, s,
               sum(CASE WHEN (outcome = 'up') = (side = 'buy') THEN q ELSE -q END) AS flow_sh,
               sum(CASE WHEN (outcome = 'up') = (side = 'buy') THEN q * p ELSE -q * p END) AS flow_usd,
               count(*) AS n, sum(q * p) AS usd
        FROM t WHERE s >= 0 AND s < {T} GROUP BY ALL
    """).fetchnumpy()
    flow_sh = np.zeros((n, T)); flow_usd = np.zeros((n, T)); usd = np.zeros((n, T))
    flow_sh[f["i"], f["s"]] = f["flow_sh"]; flow_usd[f["i"], f["s"]] = f["flow_usd"]; usd[f["i"], f["s"]] = f["usd"]

    # Individual trades (size distribution, flow).
    w = con.execute(f"""
        SELECT mk.i, (tr.ts_ms // 1000 - mk.start_s - {DELTA})::INTEGER AS s, tr.outcome = 'up' AS is_up,
               tr.side = 'buy' AS is_buy, tr.price, tr.size_shares, tr.fee_usdc
        FROM read_parquet('{trades}', hive_partitioning = false) tr JOIN mk USING (market_id)
    """).fetchnumpy()

    out = dict(
        market_id=np.asarray(d.market_id).astype(str), start_s=d.start_s, day=d.day, y=d.y, price_start=d.price_start, price_end=d.price_end,
        pm=pm, p_implied=d.p_implied, ask_up=d.ask_up[:, :T], ask_dn=d.ask_dn[:, :T], n_trades=d.n_trades,
        P=P, m=m, V=V, S=S, A0=A0, sigma=sigma,
        btc_pre=__import__("analysis.run", fromlist=["btc_matrix"]).btc_matrix(d, -3600, 0),
        flow_sh=flow_sh, flow_usd=flow_usd, usd=usd,
        tr_i=w["i"].astype(np.int32), tr_s=w["s"].astype(np.int16),
        tr_up=w["is_up"].astype(bool), tr_buy=w["is_buy"].astype(bool), tr_price=w["price"].astype(np.float32),
        tr_size=w["size_shares"].astype(np.float32), tr_fee=w["fee_usdc"].astype(np.float32),
    )
    np.savez_compressed(CACHE, **out)
    return out


def load_cache() -> dict:
    z = np.load(CACHE, allow_pickle=True)  # file written by build() above
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    o = build()
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in o.items()})
