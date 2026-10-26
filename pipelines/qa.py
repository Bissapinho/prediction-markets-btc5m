"""Contrôles qualité (spec 6). Produit `reports/qa_YYYYMMDD.md`.

Écarts à la spec, justifiés dans docs/data_notes.md :
- n°3 : le `volume` Gamma n'est pas fiable (null sur ~7 %, parfois faux) et la Data API taker ne donne pas
  le notionnel des makers. Le contrôle vérifie donc la complétude autrement (marchés au plafond de page,
  premier trade après l'ouverture, marchés sans trade dans la fenêtre) ; le ratio au volume Gamma est indicatif.
- n°6 : la clé est `trade_key` (pas de log_index dans la Data API), et un tx_hash doit être unique en mode taker.
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

    # 1. Continuité
    mm = Manifest(layout, "markets").latest()
    st = {}
    for r in mm.values():
        st[r["status"]] = st.get(r["status"], 0) + 1
    missing = sorted(k for k, r in mm.items() if r["status"] != "ok")
    ok1 = st.get("ok", 0) >= 0.95 * max(1, sum(st.values()))
    sections.append(f"## 1. Continuité des marchés\n\nStatuts du manifeste : `{st}`.\n\n"
                    + (f"Marchés non ok : {', '.join(missing[:50])}{' …' if len(missing) > 50 else ''}\n" if missing else "Aucun marché manquant.\n"))
    summary.append(("1. Continuité", _verdict(ok1), f"{st.get('ok', 0)} ok / {sum(st.values())}"))

    # 2. Couverture des trades
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
    sections.append(f"## 2. Couverture des trades\n\nManifeste `trades` : `{tst}`.\n\n"
                    f"Marchés sans aucun trade dans [0, {d}) s : **{len(no_trade)}** / {n_mk}.\n\n"
                    + (_md_table(no_trade[:60], ["début (UTC)", "slug"]) + "\n" if no_trade else ""))
    summary.append(("2. Couverture", _verdict(ok2, len(no_trade) <= 0.02 * n_mk), f"{len(no_trade)} marchés sans trade"))

    # 3. Complétude / volume
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
    sections.append("## 3. Complétude des trades\n\n"
                    f"- Marchés au plafond de page (troncature possible) : **{len(trunc)}** {trunc[:20]}\n"
                    f"- Marchés dont le premier trade est après l'ouverture : **{late_first}**\n"
                    f"- Trades taker par marché : p1/p50/p99 = {[round(x) for x in per_mkt[0]]}, max = {per_mkt[1]}\n"
                    f"- Indicatif : notionnel taker / volume Gamma, p10/p50/p90 = {[round(x, 3) for x in per_mkt[2]]} "
                    "(Gamma compte aussi les makers, le ratio est < 1 par construction)\n")
    summary.append(("3. Complétude", _verdict(ok3, len(trunc) <= 5), f"{len(trunc)} tronqués, {late_first} premier trade tardif"))

    # 4. Bornes
    b = q(f"""
        SELECT count(*) FILTER (WHERE price <= 0 OR price > 1), count(*),
               avg((t_in_window_ms BETWEEN 0 AND {d * 1000 - 1})::INT),
               avg((t_in_window_ms < 0)::INT), avg((t_in_window_ms >= {d * 1000})::INT),
               quantile_cont(t_in_window_ms / 1000, [0.001, 0.999])
        FROM tr
    """)[0]
    ok4 = b[0] == 0
    sections.append("## 4. Bornes\n\n"
                    f"- Prix hors ]0, 1] : **{b[0]}** / {b[1]:,}\n"
                    f"- Part des trades avant / dans / après la fenêtre : {b[3]:.1%} / {b[2]:.1%} / {b[4]:.1%}\n"
                    f"- t_in_window p0,1 % / p99,9 % : {[round(x) for x in b[5]]} s\n")
    summary.append(("4. Bornes", _verdict(ok4), f"{b[0]} prix hors bornes"))

    # 5. Complémentarité
    c = q("""
        WITH s AS (SELECT market_id, ts_ms // 1000 ts_s, outcome, avg(price) p FROM tr GROUP BY ALL)
        SELECT count(*), quantile_cont(u.p + w.p, [0.05, 0.5, 0.95]), avg((abs(u.p + w.p - 1) > 0.05)::INT)
        FROM s u JOIN s w ON u.market_id = w.market_id AND u.ts_s = w.ts_s AND u.outcome = 'up' AND w.outcome = 'down'
    """)[0]
    ok5 = abs(c[1][1] - 1) < 0.02
    sections.append("## 5. Complémentarité\n\n"
                    f"Secondes avec un trade Up et un trade Down : {c[0]:,}. p_up + p_down p5/p50/p95 = "
                    f"{[round(x, 3) for x in c[1]]} ; part avec un écart > 0,05 : {c[2]:.1%}.\n")
    summary.append(("5. Complémentarité", _verdict(ok5), f"médiane {c[1][1]:.3f}"))

    # 6. Doublons
    dd = q("SELECT count(*) - count(DISTINCT trade_key), count(*) - count(DISTINCT tx_hash) FROM tr")[0]
    ok6 = dd[0] == 0
    sections.append(f"## 6. Doublons\n\n- Doublons de `trade_key` : **{dd[0]}**\n- tx_hash répétés (attendu 0 en mode taker) : {dd[1]}\n")
    summary.append(("6. Doublons", _verdict(ok6, dd[1] == 0), f"{dd[0]} doublons"))

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
                    f"- Jours : `{bst}` (checksum SHA-256 vérifié pour chaque jour ok)\n"
                    f"- Plus long passage sans trade : {runs[0]} s ; passages > 60 s : **{runs[1]}**\n"
                    f"- Marchés sans prix Binance à la fin de leur fenêtre : {cover} (fichier du jour pas encore publié)\n")
    summary.append(("7. Binance", _verdict(ok7), f"{bst.get('ok', 0)} jours, max {runs[0]} s sans trade, {cover} marchés non couverts"))

    # 8. Outcome : proxy Binance (TWAP 60 s) contre Gamma / Chainlink
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
    sections.append("## 8. Outcome : proxy Binance contre Gamma\n\n"
                    f"Proxy = TWAP {L} s des close Binance de 1 s, à la fin contre au début (même règle que Chainlink).\n\n"
                    f"- Accord avec l'outcome Gamma : **{o[1]:.2%}** sur {o[0]:,} marchés\n"
                    f"- Sur les fenêtres où Chainlink a bougé de moins de 10 $ : {o[4] if o[4] is None else f'{o[4]:.2%}'} ({o[5]} marchés)\n"
                    f"- |TWAP Binance − finalPrice Chainlink| p50/p90/p99 = {[round(x, 2) for x in o[2]]} $ ; "
                    f"écart moyen (Binance − Chainlink) = {o[3]:.2f} $\n")
    summary.append(("8. Outcome", _verdict(ok8, (o[1] or 0) >= 0.9), f"accord {o[1]:.1%}"))

    # market_1s
    if Path(m1s.split("/*")[0]).exists():
        f = q(f"""
            SELECT count(*), count(DISTINCT market_id),
                   avg((p_up_last IS NULL)::INT), avg((btc_close IS NULL)::INT),
                   quantile_cont(up_age_s, [0.5, 0.9]) FILTER (WHERE t >= 30)
            FROM read_parquet('{m1s}', hive_partitioning = false)
        """)[0]
        sections.append("## Table market_1s\n\n"
                        f"{f[0]:,} lignes, {f[1]:,} marchés. p_up_last manquant : {f[2]:.2%} ; btc_close manquant : {f[3]:.2%}. "
                        f"Âge du dernier trade Up (t ≥ 30 s) p50/p90 : {f[4]} s.\n")

    now = datetime.now(timezone.utc)
    head = (f"# Rapport QA — {now:%Y-%m-%d %H:%M} UTC\n\n"
            + _md_table(summary, ["Contrôle", "Statut", "Détail"]) + "\n\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"qa_{now:%Y%m%d}.md"
    path.write_text(head + "\n".join(sections), encoding="utf-8")
    log(logger, "qa report", pipeline="qa", path=str(path), summary=json.dumps([s[:2] for s in summary], ensure_ascii=False))
    return path
