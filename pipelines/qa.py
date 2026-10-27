"""Quality checks. Writes `reports/qa_YYYYMMDD.md`.

Notes (docs/data_notes.md):
- check 3: the `volume` field of the market API is not reliable (null on ~7 %, sometimes wrong) and the taker-mode
  trade-history API does not give the maker notional. Completeness is therefore checked in other ways (markets at the
  page cap, first trade after the open, markets without a trade in the window); the ratio to the API volume is
  indicative only.
- check 6: the key is `trade_key` (no log_index in the trade-history API), and a tx_hash must be unique in taker mode.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pipelines.common.config import Config
from pipelines.common.log import get_logger, log
from pipelines.common.manifest import Manifest
from pipelines.common.storage import Layout
from pipelines.features import _posix

logger = get_logger(__name__)


def _md_table(rows: list[tuple], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def _verdict(ok: bool, warn: bool = False) -> str:
    return "🟢" if ok else ("🟡" if warn else "🔴")


def run_qa(cfg: Config, layout: Layout, out_dir: Path = Path("reports")) -> Path:
    con = duckdb.connect()
    d = cfg.duration_s
    markets = _posix(layout.clean("markets", "markets.parquet"))
    trades = _posix(layout.clean("trades")) + "/*/*.parquet"
    btc = _posix(layout.features("btc_1s")) + "/*/*.parquet"
    m1s = _posix(layout.features("market_1s")) + "/*/*.parquet"
    con.execute(f"CREATE VIEW mk AS SELECT * FROM read_parquet('{markets}') WHERE closed AND outcome IS NOT NULL")
    con.execute(f"CREATE VIEW tr AS SELECT * FROM read_parquet('{trades}', hive_partitioning = false)")
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    sections: list[str] = []
    summary: list[tuple[str, str, str]] = []

    # 1. Continuity
    mm = Manifest(layout, "markets").latest()
    st = {}
    for r in mm.values():
        st[r["status"]] = st.get(r["status"], 0) + 1
    missing = sorted(k for k, r in mm.items() if r["status"] != "ok")
    ok1 = st.get("ok", 0) >= 0.95 * max(1, sum(st.values()))
    sections.append(f"## 1. Market continuity\n\nManifest statuses: `{st}`.\n\n"
                    + (f"Markets not ok: {', '.join(missing[:50])}{' …' if len(missing) > 50 else ''}\n" if missing else "No missing market.\n"))
    summary.append(("1. Continuity", _verdict(ok1), f"{st.get('ok', 0)} ok / {sum(st.values())}"))

    # 2. Trade coverage
    tm = Manifest(layout, "trades").latest()
    tst: dict[str, int] = {}
    for r in tm.values():
        tst[r["status"]] = tst.get(r["status"], 0) + 1
    no_trade = q(f"""
        SELECT strftime(make_timestamp(mk.start_ts_ms * 1000), '%Y-%m-%d %H:%M') AS start_utc, mk.slug
        FROM mk LEFT JOIN (SELECT market_id, count(*) n FROM tr WHERE t_in_window_ms BETWEEN 0 AND {d * 1000 - 1} GROUP BY 1) c
          USING (market_id)
        WHERE coalesce(c.n, 0) = 0 ORDER BY 1
    """)
    n_mk = q("SELECT count(*) FROM mk")[0][0]
    ok2 = len(no_trade) <= 0.005 * n_mk
    sections.append(f"## 2. Trade coverage\n\n`trades` manifest: `{tst}`.\n\n"
                    f"Markets without any trade in [0, {d}) s: **{len(no_trade)}** / {n_mk}.\n\n"
                    + (_md_table(no_trade[:60], ["start (UTC)", "slug"]) + "\n" if no_trade else ""))
    summary.append(("2. Coverage", _verdict(ok2, len(no_trade) <= 0.02 * n_mk), f"{len(no_trade)} markets without a trade"))

    # 3. Completeness / volume
    trunc = [k for k, r in tm.items() if r["status"] == "truncated"]
    late_first = q("""
        SELECT count(*) FROM (SELECT market_id, min(t_in_window_ms) f FROM tr GROUP BY 1) WHERE f > 0
    """)[0][0]
    per_mkt = q("""
        SELECT quantile_cont(n, [0.01, 0.5, 0.99]), max(n),
               quantile_cont(notional / nullif(volume_usdc, 0), [0.1, 0.5, 0.9])
        FROM (SELECT market_id, count(*) n, sum(notional_usdc) notional FROM tr GROUP BY 1) JOIN mk USING (market_id)
    """)[0]
    ok3 = not trunc and late_first <= 0.005 * n_mk
    sections.append("## 3. Trade completeness\n\n"
                    f"- Markets at the page cap (possible truncation): **{len(trunc)}** {trunc[:20]}\n"
                    f"- Markets whose first trade is after the open: **{late_first}**\n"
                    f"- Taker trades per market: p1/p50/p99 = {[round(x) for x in per_mkt[0]]}, max = {per_mkt[1]}\n"
                    f"- Indicative: taker notional / market-API volume, p10/p50/p90 = {[round(x, 3) for x in per_mkt[2]]} "
                    "(the API volume also counts makers, so the ratio is < 1 by construction)\n")
    summary.append(("3. Completeness", _verdict(ok3, len(trunc) <= 5), f"{len(trunc)} truncated, {late_first} late first trade"))

    # 4. Bounds
    b = q(f"""
        SELECT count(*) FILTER (WHERE price <= 0 OR price > 1), count(*),
               avg((t_in_window_ms BETWEEN 0 AND {d * 1000 - 1})::INT),
               avg((t_in_window_ms < 0)::INT), avg((t_in_window_ms >= {d * 1000})::INT),
               quantile_cont(t_in_window_ms / 1000, [0.001, 0.999])
        FROM tr
    """)[0]
    ok4 = b[0] == 0
    sections.append("## 4. Bounds\n\n"
                    f"- Prices outside ]0, 1]: **{b[0]}** / {b[1]:,}\n"
                    f"- Share of trades before / inside / after the window: {b[3]:.1%} / {b[2]:.1%} / {b[4]:.1%}\n"
                    f"- t_in_window p0.1 % / p99.9 %: {[round(x) for x in b[5]]} s\n")
    summary.append(("4. Bounds", _verdict(ok4), f"{b[0]} prices out of bounds"))

    # 5. Complementarity
    c = q("""
        WITH s AS (SELECT market_id, ts_ms // 1000 ts_s, outcome, avg(price) p FROM tr GROUP BY ALL)
        SELECT count(*), quantile_cont(u.p + w.p, [0.05, 0.5, 0.95]), avg((abs(u.p + w.p - 1) > 0.05)::INT)
        FROM s u JOIN s w ON u.market_id = w.market_id AND u.ts_s = w.ts_s AND u.outcome = 'up' AND w.outcome = 'down'
    """)[0]
    ok5 = abs(c[1][1] - 1) < 0.02
    sections.append("## 5. Complementarity\n\n"
                    f"Seconds with an Up trade and a Down trade: {c[0]:,}. p_up + p_down p5/p50/p95 = "
                    f"{[round(x, 3) for x in c[1]]}; share with a gap > 0.05: {c[2]:.1%}.\n")
    summary.append(("5. Complementarity", _verdict(ok5), f"median {c[1][1]:.3f}"))

    # 6. Duplicates
    dd = q("SELECT count(*) - count(DISTINCT trade_key), count(*) - count(DISTINCT tx_hash) FROM tr")[0]
    ok6 = dd[0] == 0
    sections.append(f"## 6. Duplicates\n\n- Duplicated `trade_key`: **{dd[0]}**\n- Repeated tx_hash (0 expected in taker mode): {dd[1]}\n")
    summary.append(("6. Duplicates", _verdict(ok6, dd[1] == 0), f"{dd[0]} duplicates"))

    # 7. Binance
    bm = Manifest(layout, "binance").latest()
    bst: dict[str, int] = {}
    for r in bm.values():
        bst[r["status"]] = bst.get(r["status"], 0) + 1
    runs = q(f"""
        WITH r AS (SELECT ts_s, is_filled, sum((NOT is_filled)::INT) OVER (ORDER BY ts_s) grp FROM read_parquet('{btc}', hive_partitioning = false))
        SELECT max(c), count(*) FILTER (WHERE c > 60) FROM (SELECT grp, count(*) FILTER (WHERE is_filled) c FROM r GROUP BY grp)
    """)[0]
    cover = q(f"""
        SELECT count(*) FILTER (WHERE b.ts_s IS NULL) FROM mk
        LEFT JOIN read_parquet('{btc}', hive_partitioning = false) b ON b.ts_s = mk.end_ts_ms // 1000 - 1
    """)[0][0]
    ok7 = bst.get("ok", 0) == sum(bst.values()) and runs[1] == 0
    sections.append("## 7. Binance\n\n"
                    f"- Days: `{bst}` (SHA-256 checksum verified for each ok day)\n"
                    f"- Longest stretch without a trade: {runs[0]} s; stretches > 60 s: **{runs[1]}**\n"
                    f"- Markets without a Binance price at the end of their window: {cover} (file of the day not published yet)\n")
    summary.append(("7. Binance", _verdict(ok7), f"{bst.get('ok', 0)} days, max {runs[0]} s without a trade, {cover} markets not covered"))

    # 8. Outcome: Binance proxy (60 s TWAP) against the market API / Chainlink
    L = 60
    o = q(f"""
        WITH b AS (SELECT ts_s, close FROM read_parquet('{btc}', hive_partitioning = false)),
        w0 AS (
            SELECT mk.market_id, avg(b.close) AS tw0 FROM mk JOIN b
              ON b.ts_s >= mk.start_ts_ms // 1000 - {L} AND b.ts_s < mk.start_ts_ms // 1000 GROUP BY 1
        ),
        w1 AS (
            SELECT mk.market_id, avg(b.close) AS tw1 FROM mk JOIN b
              ON b.ts_s >= mk.end_ts_ms // 1000 - {L} AND b.ts_s < mk.end_ts_ms // 1000 GROUP BY 1
        ),
        tw AS (SELECT mk.market_id, mk.outcome, mk.price_start, mk.price_end, w0.tw0, w1.tw1
               FROM mk JOIN w0 USING (market_id) JOIN w1 USING (market_id))
        SELECT count(*), avg(((tw1 >= tw0) = (outcome = 'up'))::INT),
               quantile_cont(abs(tw1 - price_end), [0.5, 0.9, 0.99]),
               avg(tw1 - price_end),
               avg(((tw1 >= tw0) = (outcome = 'up'))::INT) FILTER (WHERE abs(price_end - price_start) < 10),
               count(*) FILTER (WHERE abs(price_end - price_start) < 10)
        FROM tw WHERE tw0 IS NOT NULL AND tw1 IS NOT NULL AND price_end IS NOT NULL
    """)[0]
    ok8 = o[1] is not None and o[1] >= 0.95
    sections.append("## 8. Outcome: Binance proxy against the market API\n\n"
                    f"Proxy = {L} s TWAP of the 1-second Binance closes, end against start (same rule as Chainlink).\n\n"
                    f"- Agreement with the API outcome: **{o[1]:.2%}** on {o[0]:,} markets\n"
                    f"- On windows where Chainlink moved by less than 10 $: {o[4] if o[4] is None else f'{o[4]:.2%}'} ({o[5]} markets)\n"
                    f"- |Binance TWAP − Chainlink finalPrice| p50/p90/p99 = {[round(x, 2) for x in o[2]]} $; "
                    f"mean gap (Binance − Chainlink) = {o[3]:.2f} $\n")
    summary.append(("8. Outcome", _verdict(ok8, (o[1] or 0) >= 0.9), f"agreement {o[1]:.1%}"))

    # market_1s
    if Path(m1s.split("/*")[0]).exists():
        f = q(f"""
            SELECT count(*), count(DISTINCT market_id),
                   avg((p_up_last IS NULL)::INT), avg((btc_close IS NULL)::INT),
                   quantile_cont(up_age_s, [0.5, 0.9]) FILTER (WHERE t >= 30)
            FROM read_parquet('{m1s}', hive_partitioning = false)
        """)[0]
        sections.append("## Table market_1s\n\n"
                        f"{f[0]:,} rows, {f[1]:,} markets. p_up_last missing: {f[2]:.2%}; btc_close missing: {f[3]:.2%}. "
                        f"Age of the last Up trade (t ≥ 30 s) p50/p90: {f[4]} s.\n")

    now = datetime.now(timezone.utc)
    head = (f"# QA report — {now:%Y-%m-%d %H:%M} UTC\n\n"
            + _md_table(summary, ["Check", "Status", "Detail"]) + "\n\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"qa_{now:%Y%m%d}.md"
    path.write_text(head + "\n".join(sections), encoding="utf-8")
    log(logger, "qa report", pipeline="qa", path=str(path), summary=json.dumps([s[:2] for s in summary], ensure_ascii=False))
    return path
