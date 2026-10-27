"""Tables of the README and docs/METHODS.md, recomputed from the analysis outputs (results/*.json) and the parquet files.

Writes results/summary.json, results/hypotheses.md, results/ml.md, results/latency.md and results/key_figures.md.
Usage: python -m analysis.summary

Bootstrap units. Every test is reported with three cluster bootstraps: by market, by UTC day and by UTC hour
(analysis/blockboot.py). The bootstrap by UTC day is the reference for the verdicts: the 288 markets of a day share
one volatility regime, and some features use the previous window. The bootstrap by market is kept as a secondary
column and the bootstrap by UTC hour is a sensitivity analysis.

p-values. When an analysis stores its own bootstrap p-value (one-sided, H0: mean P&L <= 0; H2-info: c <= 0), it is
used as is. Otherwise the p-value comes from a normal approximation of the bootstrap distribution:
se = (hi - lo) / (2 * 1.96).

Multiple testing. Benjamini-Hochberg (q = 0.05) within each family, computed separately for each bootstrap unit.
A3 is not in any family: its market cluster is defined from the whole price path (look-ahead), so it is reported as an
ex-post diagnostic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
from scipy.special import ndtr

from analysis.run import bh

R = Path("results")
DATA = Path("data")
START = os.environ.get("PM_START", "2026-08-17")
Q = 0.05
Z95 = 1.959964
UNITS = ("day", "market", "hour")          # "day" is the reference
FAMILIES = {"P": "P (protocol)", "A": "A (price paths)", "I": "I (information)", "X": "ex-post diagnostic", "–": "–"}


def J(name: str) -> dict:
    with open(R / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def p_normal(est, lo, hi, h0=0.0, sided="greater"):
    se = (hi - lo) / (2 * Z95)
    z = (est - h0) / se
    return float(1 - ndtr(z)) if sided == "greater" else float(2 * (1 - ndtr(abs(z))))


def cents(x):
    return "–" if x is None else f"{x * 100:+.1f}"


def fmt_p(p):
    return "–" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")


# --------------------------------------------------------------------------------------------------------------------
# Data counts, straight from the parquet files (same filters as analysis/data.py).
# --------------------------------------------------------------------------------------------------------------------

def data_counts() -> dict:
    con = duckdb.connect()
    mk = f"read_parquet('{DATA}/clean/markets/markets.parquet')"
    tr = f"read_parquet('{DATA}/clean/trades/*/*.parquet', hive_partitioning = false)"
    m1s = f"read_parquet('{DATA}/features/market_1s/*/*.parquet', hive_partitioning = false)"
    b1s = f"read_parquet('{DATA}/features/btc_1s/*/*.parquet', hive_partitioning = false)"
    agg = f"read_parquet('{DATA}/clean/btc_aggtrades/*/*.parquet', hive_partitioning = false)"
    b0, b1 = con.execute(f"SELECT min(ts_s), max(ts_s) FROM {b1s}").fetchone()
    start_ms = f"epoch_ms(TIMESTAMP '{START}')"
    end_ms = "epoch_ms(TIMESTAMP '2026-10-18')"
    con.execute(f"""
        CREATE TEMP TABLE period AS SELECT * FROM {mk} WHERE start_ts_ms >= {start_ms} AND start_ts_ms < {end_ms};
        CREATE TEMP TABLE n_in AS SELECT market_id, count(*) n FROM {tr}
            WHERE t_in_window_ms >= 0 AND t_in_window_ms < 300000 GROUP BY 1;
        CREATE TEMP TABLE kept AS
            SELECT p.market_id, p.start_ts_ms FROM period p JOIN n_in USING (market_id)
            WHERE p.closed AND p.outcome IS NOT NULL
              AND p.start_ts_ms // 1000 - 1800 >= {b0} AND p.start_ts_ms // 1000 + 300 <= {b1};
    """)
    q1 = lambda s: con.execute(s).fetchone()
    n_windows = 1 + (int(q1(f"SELECT {end_ms}")[0]) - 300_000 - int(q1(f"SELECT {start_ms}")[0])) // 300_000
    n_period = q1("SELECT count(*) FROM period")[0]
    n_unresolved = q1("SELECT count(*) FROM period WHERE NOT closed OR outcome IS NULL")[0]
    n_no_trade = q1("SELECT count(*) FROM period p WHERE closed AND outcome IS NOT NULL AND market_id NOT IN (SELECT market_id FROM n_in)")[0]
    n_kept, first_ms, last_ms = q1("SELECT count(*), min(start_ts_ms), max(start_ts_ms) FROM kept")
    n_trades, notional = q1(f"SELECT count(*), sum(notional_usdc) FROM {tr} JOIN kept USING (market_id)")
    n_trades_window = q1(f"SELECT count(*) FROM {tr} JOIN kept USING (market_id) WHERE t_in_window_ms >= 0 AND t_in_window_ms < 300000")[0]
    rows_m1s = q1(f"SELECT count(*) FROM {m1s} JOIN kept USING (market_id)")[0]
    lo_s, hi_s = first_ms // 1000 - 1800, last_ms // 1000 + 300
    rows_b1s = q1(f"SELECT count(*) FROM {b1s} WHERE ts_s >= {lo_s} AND ts_s < {hi_s}")[0]
    n_agg = q1(f"SELECT count(*) FROM {agg} WHERE ts_ms >= {lo_s * 1000} AND ts_ms < {hi_s * 1000}")[0]
    n_days = q1("SELECT count(DISTINCT start_ts_ms // 86400000) FROM kept")[0]
    fmt = lambda ms: str(np.datetime64(int(ms), "ms"))[:16].replace("T", " ")
    return {"n_windows": n_windows, "n_markets_listed": int(n_period), "n_unresolved": int(n_unresolved),
            "n_no_trade_in_window": int(n_no_trade), "n_no_binance": int(n_period - n_unresolved - n_no_trade - n_kept),
            "n_markets": int(n_kept), "n_days": int(n_days), "first_start_utc": fmt(first_ms), "last_start_utc": fmt(last_ms),
            "n_trades": int(n_trades), "n_trades_in_window": int(n_trades_window), "notional_usdc": float(notional),
            "rows_market_1s": int(rows_m1s), "rows_btc_1s": int(rows_b1s), "rows_per_second_total": int(rows_m1s + rows_b1s),
            "n_binance_aggtrades": int(n_agg)}


# --------------------------------------------------------------------------------------------------------------------
# Hypotheses
# --------------------------------------------------------------------------------------------------------------------

def _boot(market: dict, blocks: dict) -> dict:
    """{unit: {lo, hi, p, n_groups}} from a market-level result and its block results."""
    out = {"market": {"lo": market.get("lo"), "hi": market.get("hi"), "p": market.get("p"), "n_groups": market.get("n_markets")}}
    for u in ("day", "hour"):
        b = blocks.get(u) or {}
        out[u] = {"lo": b.get("lo"), "hi": b.get("hi"), "p": b.get("p"), "n_groups": b.get("n_groups")}
    return out


def hypotheses() -> list[dict]:
    r2 = J("results_delta2")
    e2, e3, e4, e6, e8 = (J(f"edges_{s}") for s in ("E2", "E3", "E4", "E6", "E8"))
    dA, dX, dD = J("deep_A"), J("deep_C_extra"), J("deep_D")
    S = r2["strategies"]
    rows: list[dict] = []

    def pnl_row(hid, fam, title, test, execution, selected=None):
        rows.append({"id": hid, "family": fam, "title": title, "kind": "pnl", "execution": execution, "selected": selected,
                     "stat": "mean P&L per share, fees included", "est": test.get("mean"), "n": test.get("n"),
                     "boot": _boot(test, test)})

    def info_row(hid, fam, title, stat, est, n, boot, h0=0.0, sided="two"):
        for u, b in boot.items():
            if b.get("p") is None and b.get("lo") is not None:
                b["p"] = p_normal(est, b["lo"], b["hi"], h0=h0, sided=sided)
        rows.append({"id": hid, "family": fam, "title": title, "kind": "info", "execution": "–", "selected": None,
                     "stat": stat, "est": est, "n": n, "boot": boot})

    # Family P: protocol specified before the test evaluation.
    pnl_row("H1", "P", "Lead-lag: buy when Binance moves >= z0 sigma in 3 s and the market has not moved", S["S1"]["test"], "taker, 2 s", S["S1"]["selected"])
    pnl_row("H2", "P", "TWAP fair value from Binance: buy the side priced below the model", S["S2"]["test"], "taker, 2 s", S["S2"]["selected"])
    h2 = r2["H2"]["pooled_test"]
    info_row("H2-info", "P", "Model minus market predicts the outcome (logit coefficient c > 0)", "coefficient c", h2["c"], h2["n_obs"],
             {"market": {"lo": h2["c_lo"], "hi": h2["c_hi"], "p": h2["p_c"]},
              **{u: {"lo": h2[u]["c_lo"], "hi": h2[u]["c_hi"], "p": h2[u]["p_c"], "n_groups": h2[u]["n_groups"]} for u in ("day", "hour")}})
    pnl_row("H3", "P", "Fade price jumps that Binance does not justify", S["S3"]["test"], "taker, 2 s", S["S3"]["selected"])
    pnl_row("H4", "P", "Late favourites (t = 240-290 s, ask 0.80-0.98)", S["S4"]["test"], "taker, 2 s", S["S4"]["selected"])

    # Family A: price-path strategies.
    pnl_row("A1", "A", "Dip-buy: buy a side below 30 c, sell at 33 c", e2["case_30_33_taker_test"], "taker, 1 s")
    sel = e2["selected"]
    pnl_row("A2", "A", f"Dip-buy grid ({len(e2['grid'])} cells), best cell on train", sel["test"], "limit order",
            f"{sel['window']}, x = {sel['x']}, k = {sel['k']}")
    cl = max(e6["clusters"], key=lambda c: c["train"]["dip_30_33"] if c["train"]["dip_30_33"] is not None else -9)
    pnl_row("A3", "X", "Dip-buy restricted to the 'hesitant' market cluster (cluster defined from the whole price path)",
            cl["test"]["dip_30_33_boot"], "limit order", f"cluster {cl['cluster']}")
    pnl_row("A4", "A", f"ML classifier picks the dips that rebound (test AUC {e8['auc_test']:.2f})", e8["test_selected"], "limit order",
            f"top {1 - e8['selected_quantile']:.0%} scores")
    mom = next(r for r in e3["momentum_strategy"] if r["L_min"] == 1 and r["split"] == "test")
    pnl_row("A5", "A", "Pre-open momentum: buy the side of the last 1-min Binance trend", mom, "taker, 1 s", "L = 1 min")
    anc = next(r for r in e3["anchor_strategy"] if r["s"] == 0 and r["theta"] == 0)["test"]
    pnl_row("A6", "A", "TWAP anchor at the open: buy model minus opening price", anc, "taker, 1 s")
    sn = next(r for r in e4["sniping"] if r["q"] == 0.995 and r["cap"] == 0.99)["test"]
    pnl_row("A7", "A", "End-of-window sniping: buy at <= 99 c a side the model gives > 99.5 %", sn, "taker, 1 s")
    pa = e4["pair_arbitrage"]["test"]
    rows.append({"id": "A8", "family": "–", "title": "Pair arbitrage (Up + Down < 1 $)", "kind": "artefact",
                 "stat": "share of seconds with ask_up + ask_dn < 1", "est": pa["share_sum_lt_1"], "n": pa["seconds_with_both"],
                 "boot": {u: {"lo": None, "hi": None, "p": None} for u in UNITS}, "execution": "–", "selected": None})

    # Family I: information beyond the price (statistical tests, not P&L tests).
    os_ = dX["test"]["logit_y_on_price_and_streak"]
    info_row("A9", "I", "Under-reaction at the open: slope of outcome on logit(price), 10-40 s (H0: slope = 1)", "logit slope",
             os_["beta"][1], os_["n"],
             {"market": {"lo": os_["lo"][1], "hi": os_["hi"][1]},
              **{u: {"lo": os_[u]["lo"][1], "hi": os_[u]["hi"][1], "n_groups": os_[u]["n_groups"]} for u in ("day", "hour")}},
             h0=1.0)
    st = dD["D1_stacked"]["logistic"]
    info_row("A10", "I", "ML forecast of the outcome stacked on the market price", "log-loss gain vs market (nats)", st["gain"],
             dD["D1_outcome"]["n_test_rows"],
             {"market": {"lo": st["lo"], "hi": st["hi"]},
              **{u: {"lo": st[u]["lo"], "hi": st[u]["hi"], "n_groups": st[u]["n_groups"]} for u in ("day", "hour")}}, sided="greater")
    ld = dA["resid_leads_btc"]["test"]["10"]["mean_per_market"]
    info_row("A11", "I", "Price residual (not explained by Binance) leads the next 10 s Binance move", "mean per-market correlation",
             ld["est"], ld["n"],
             {"market": {"lo": ld["lo"], "hi": ld["hi"]},
              **{u: {"lo": ld[u]["lo"], "hi": ld[u]["hi"], "n_groups": ld[u]["n_groups"]} for u in ("day", "hour")}})

    for r in rows:
        r["p_adj"], r["significant"] = {u: None for u in UNITS}, {u: None for u in UNITS}
    for fam in ("P", "A", "I"):
        for u in UNITS:
            ps = {r["id"]: (r["boot"][u]["p"] if r["boot"][u]["p"] is not None else 1.0) for r in rows if r["family"] == fam}
            adj = bh(ps, Q)
            for r in rows:
                if r["id"] in adj:
                    r["p_adj"][u], r["significant"][u] = adj[r["id"]]["p_adj"], adj[r["id"]]["significant"]
    for r in rows:
        r["verdict"] = verdict(r)
    return rows


def verdict(r: dict) -> str:
    """Verdict from the bootstrap by UTC day."""
    if r["kind"] == "artefact":
        return "Artefact: the two prices come from different instants of the same second"
    if r["est"] is None:
        return "Not testable (fewer than 5 trades)"
    if r["family"] == "X":
        return ("Ex-post diagnostic: the cluster uses the whole price path (look-ahead), so this is not a tradable strategy; "
                "not in any BH family")
    sig = r["significant"]["day"]
    if r["kind"] == "pnl":
        if sig and r["est"] > 0:
            return "Positive after BH"
        if r["est"] <= 0:
            return "Rejected: negative P&L"
        return "Not significant after BH"
    if sig:
        if r["id"] == "H2-info":
            return "Information detected after BH; the taker strategy built on it (H2) loses"
        return "Information detected after BH (statistical test, not a P&L test)"
    return "No information detected after BH"


# --------------------------------------------------------------------------------------------------------------------
# ML / DL, latency, power
# --------------------------------------------------------------------------------------------------------------------

NAMES = {"logistic": "Logistic regression", "gradient boosting": "Gradient boosting (HistGB)", "MLP": "MLP (64, 32)",
         "market recalibrated": "Market price, recalibrated (logistic on logit p)"}


def ml_rows() -> list[dict]:
    dD = J("deep_D")
    rows = []
    items = [(k, v, "standalone") for k, v in dD["D1_outcome"]["models"].items()] + \
            [(k, v, "stacked on market price") for k, v in dD["D1_stacked"].items()]
    for k, v, setting in items:
        name = "GRU (30 s sequences + static features)" if k.startswith("GRU") else NAMES[k]
        boot = {"market": {"lo": v["lo"], "hi": v["hi"]}, **{u: {"lo": v[u]["lo"], "hi": v[u]["hi"]} for u in ("day", "hour")}}
        for b in boot.values():
            b["p"] = p_normal(v["gain"], b["lo"], b["hi"])
        rows.append({"model": name, "setting": setting, "gain": v["gain"], "logloss_model": v["logloss_model"],
                     "logloss_market": v["logloss_market"], "boot": boot, "p_adj": {}, "significant": {}})
    for u in UNITS:
        adj = bh({i: r["boot"][u]["p"] for i, r in enumerate(rows)}, Q)
        for i, r in enumerate(rows):
            r["p_adj"][u], r["significant"][u] = adj[i]["p_adj"], adj[i]["significant"]
    return rows


def latency() -> dict:
    out = {}
    for d in (0, 2, 3):
        r = J(f"results_delta{d}")
        lat = r["strategies"]["S1"]["latency"]
        out[str(d)] = {L: {"mean": v["mean"], "n": v["n"], "market": {k: v[k] for k in ("lo", "hi", "p")},
                           **{u: {k: v[u][k] for k in ("lo", "hi", "p")} for u in ("day", "hour")}} for L, v in lat.items()}
    return out


def mde(rows) -> dict:
    """Effect detectable with 80 % power, one-sided 5 % test: (z_0.95 + z_0.80) * se, se from the bootstrap CI."""
    out = {u: {} for u in UNITS}
    for r in rows:
        if r["family"] == "P" and r["kind"] == "pnl":
            for u in UNITS:
                b = r["boot"][u]
                out[u][r["id"]] = (1.644854 + 0.841621) * (b["hi"] - b["lo"]) / (2 * Z95)
    return out


# --------------------------------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------------------------------

def fmt_est(r):
    if r["est"] is None:
        return "–"
    if r["kind"] == "pnl":
        return f"{cents(r['est'])} c"
    if r["kind"] == "artefact":
        return f"{r['est']:.0%}"
    return f"{r['est']:.4f}" if abs(r["est"]) < 0.01 else f"{r['est']:.3f}"


def fmt_ci(r, u):
    b = r["boot"][u]
    if b["lo"] is None:
        return "–"
    if r["kind"] == "pnl":
        return f"[{cents(b['lo'])}, {cents(b['hi'])}]"
    f = (lambda x: f"{x:.4f}") if abs(r["est"]) < 0.01 else (lambda x: f"{x:.3f}")
    return f"[{f(b['lo'])}, {f(b['hi'])}]"


def md_hyp(rows) -> str:
    head = ["| ID | Hypothesis | Execution | Statistic (test) | Estimate | n | 95 % CI (day) | p (day) | p adj. (day) "
            "| 95 % CI (market) | p adj. (market) | p adj. (hour) | Family | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def line(r):
        n = "–" if r["n"] is None else f"{r['n']:,}"
        return (f"| {r['id']} | {r['title']} | {r['execution']} | {r['stat']} | {fmt_est(r)} | {n} | {fmt_ci(r, 'day')} | "
                f"{fmt_p(r['boot']['day']['p'])} | {fmt_p(r['p_adj']['day'])} | {fmt_ci(r, 'market')} | "
                f"{fmt_p(r['p_adj']['market'])} | {fmt_p(r['p_adj']['hour'])} | {FAMILIES[r['family']]} | {r['verdict']} |")

    main = [line(r) for r in rows if r["family"] != "X"]
    expost = [line(r) for r in rows if r["family"] == "X"]
    return ("## Tests (Benjamini-Hochberg within each family)\n\n" + "\n".join(head + main) + "\n\n"
            "## Ex-post diagnostics (not in any BH family; p adj. is not computed)\n\n" + "\n".join(head + expost) + "\n")


def md_ml(rows) -> str:
    out = ["| Model | Setting | Log-loss (model) | Log-loss (market) | Gain (nats) | 95 % CI (day) | p adj. (day) "
           "| 95 % CI (market) | p adj. (market) | p adj. (hour) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bd, bm = r["boot"]["day"], r["boot"]["market"]
        out.append(f"| {r['model']} | {r['setting']} | {r['logloss_model']:.4f} | {r['logloss_market']:.4f} | "
                   f"{r['gain']:+.4f} | [{bd['lo']:+.4f}, {bd['hi']:+.4f}] | {fmt_p(r['p_adj']['day'])} | "
                   f"[{bm['lo']:+.4f}, {bm['hi']:+.4f}] | {fmt_p(r['p_adj']['market'])} | {fmt_p(r['p_adj']['hour'])} |")
    return "\n".join(out) + "\n"


def md_latency(lat) -> str:
    Ls = list(lat["2"])
    out = []
    for u, title in (("day", "95 % CI, bootstrap by UTC day (reference)"), ("market", "95 % CI, bootstrap by market")):
        out += [f"### {title}", "", "| Re-alignment | " + " | ".join(f"L = {L} s" for L in Ls) + " |", "|---|" + "---|" * len(Ls)]
        for d, lab in (("0", "none (block time)"), ("2", "−2 s (reference)"), ("3", "−3 s")):
            out.append(f"| {lab} | " + " | ".join(f"{cents(v['mean'])} [{cents(v[u]['lo'])}, {cents(v[u]['hi'])}]"
                                                for v in lat[d].values()) + " |")
        out.append("")
    return "\n".join(out)


def md_key(sections) -> str:
    out = []
    for title, K in sections:
        out += [f"## {title}", "", "| Figure | Value | Source |", "|---|---|---|"]
        out += [f"| {name} | {val} | {src} |" for name, val, src in K]
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------------------------
# Key figures: every number quoted in README.md and docs/METHODS.md
# --------------------------------------------------------------------------------------------------------------------

def key_figures(dc, r2, hyp, ml, lat, lag, md, a11) -> list:
    meta, model, h2s = r2["meta"], r2["model"], r2["H2"]["scores"]
    H = {r["id"]: r for r in hyp}
    best_ml = max(ml, key=lambda r: r["gain"])
    gru = next(r for r in ml if r["model"].startswith("GRU"))
    recal = next(r for r in ml if r["model"].startswith("Market price"))
    n_models = len(ml)
    tested = [r for r in hyp if r["family"] in ("P", "A", "I")]
    pos_taker = [r for r in tested if r["kind"] == "pnl" and r["significant"]["day"] and r["est"] > 0 and r["execution"].startswith("taker")]
    pos_any = [r for r in tested if r["kind"] == "pnl" and r["significant"]["day"] and r["est"] > 0]
    r0 = J("results_delta0")
    h1_0 = r0["strategies"]["S1"]["latency"]["2"]
    src_lat = "results/results_delta{d}.json → results/latency.md"

    def ci_c(v, u):
        return f"{cents(v['mean'])} c/share [{cents(v[u]['lo'])}, {cents(v[u]['hi'])}]"

    def info(r, u):
        return f"{fmt_est(r)} {fmt_ci(r, u)}"

    data = [
        ("Markets analysed", f"{dc['n_markets']:,}", "data/clean/markets + filters of analysis/data.py → results/summary.json"),
        ("Five-minute windows in the period", f"{dc['n_windows']:,}", "results/summary.json"),
        ("Listed by the market API", f"{dc['n_markets_listed']:,}", "results/summary.json"),
        ("Markets with at least one trade in [0, 300 s)", f"{dc['n_markets'] + dc['n_no_binance']:,}", "results/summary.json"),
        ("Days analysed", f"{dc['n_days']}", "results/summary.json"),
        ("Period (market start, UTC)", f"{dc['first_start_utc']} → {dc['last_start_utc']}", "results/summary.json"),
        ("Train / test markets (split 2026-09-21)", f"{meta['n_train']:,} / {meta['n_test']:,}", "results/results_delta2.json (meta)"),
        ("Test days (UTC)", f"{H['H1']['boot']['day']['n_groups']}", "results/results_delta2.json (S1 test, day bootstrap)"),
        ("Trades on the analysed markets", f"{dc['n_trades']:,}", "data/clean/trades → results/summary.json"),
        ("of which inside the 5-min window", f"{dc['n_trades_in_window']:,}", "results/summary.json"),
        ("Traded notional", f"{dc['notional_usdc'] / 1e6:,.1f} M USDC", "results/summary.json"),
        ("Binance BTCUSDT aggregated trades", f"{dc['n_binance_aggtrades']:,}", "data/clean/btc_aggtrades → results/summary.json"),
        ("Per-second rows (market × second)", f"{dc['rows_market_1s']:,}", "data/features/market_1s → results/summary.json"),
        ("Per-second rows (Binance 1 s)", f"{dc['rows_btc_1s']:,}", "data/features/btc_1s → results/summary.json"),
        ("Per-second rows, total", f"{dc['rows_per_second_total']:,} ({dc['rows_per_second_total'] / 1e6:.1f} M)", "results/summary.json"),
    ]
    model_k = [
        ("Fair-value model, fitted on train", f"k = {model['k_star']:.2f}, τ = {model['tau_star']:.0e}", "results/results_delta2.json (model)"),
        ("Model reproduces the outcome at t = 299 s (test)", f"{model['outcome_agreement_t299']['test']:.1%}", "results/results_delta2.json (model)"),
        ("Test log-loss, market vs model, 10-90 s", f"{h2s['test|early (10-90 s)']['logloss_market']:.3f} vs {h2s['test|early (10-90 s)']['logloss_model']:.3f}", "results/results_delta2.json (H2.scores)"),
        ("Test log-loss, market vs model, 100-230 s", f"{h2s['test|middle (100-230 s)']['logloss_market']:.3f} vs {h2s['test|middle (100-230 s)']['logloss_model']:.3f}", "results/results_delta2.json (H2.scores)"),
        ("Test log-loss, market vs model, 240-290 s", f"{h2s['test|late (240-290 s)']['logloss_market']:.3f} vs {h2s['test|late (240-290 s)']['logloss_model']:.3f}", "results/results_delta2.json (H2.scores)"),
    ]
    tests = [
        ("Hypotheses in the public table", f"{len({r['id'].split('-')[0] for r in hyp})}", "results/hypotheses.md"),
        ("Tests in BH families (P, A, I)", f"{len(tested)}", "results/hypotheses.md"),
        ("Ex-post diagnostics / artefacts", f"{sum(r['family'] == 'X' for r in hyp)} (A3) / {sum(r['kind'] == 'artefact' for r in hyp)} (A8)", "results/hypotheses.md"),
        ("Strategies with a P&L test in a BH family", f"{sum(r['kind'] == 'pnl' for r in tested)}", "results/hypotheses.md"),
        ("P&L tests significant after BH (day) with positive P&L", f"{len(pos_any)}", "results/hypotheses.md"),
        ("of which taker strategies", f"{len(pos_taker)}", "results/hypotheses.md"),
        ("Detectable edge (80 % power), H1-H4, bootstrap by day", f"{min(md['day'].values()) * 100:.1f} to {max(md['day'].values()) * 100:.1f} c/share", "results/summary.json (mde)"),
        ("Detectable edge (80 % power), H1-H4, bootstrap by market", f"{min(md['market'].values()) * 100:.1f} to {max(md['market'].values()) * 100:.1f} c/share", "results/summary.json (mde)"),
    ]
    for r in hyp:
        if r["kind"] == "artefact":
            tests.append((f"{r['id']} share of seconds with ask_up + ask_dn < 1 (test)", f"{fmt_est(r)} of {r['n']:,} s", "results/edges_E4.json"))
            continue
        for u in ("day", "market"):
            pa = r["p_adj"][u]
            p_txt = f", p adj. {fmt_p(pa)}" if pa is not None else f", p {fmt_p(r['boot'][u]['p'])}"
            tests.append((f"{r['id']} ({u} bootstrap)", f"{info(r, u)}{p_txt}", "results/hypotheses.md"))
    ml_k = [
        ("Models compared with the market price", f"{n_models}", "results/ml.md"),
        ("ML test predictions (market × snapshot)", f"{J('deep_D')['D1_outcome']['n_test_rows']:,}", "results/deep_D.json"),
        ("Best ML log-loss gain vs market (best of 8 models)", f"{best_ml['gain']:+.4f} nats ({best_ml['model']}, {best_ml['setting']})", "results/deep_D.json → results/ml.md"),
        ("Best ML gain as a share of the market log-loss", f"{best_ml['gain'] / best_ml['logloss_market']:.1%}", "results/ml.md"),
        ("Recalibrated market price, gain", f"{recal['gain']:+.4f} nats", "results/ml.md"),
        ("GRU log-loss gain vs market (day CI)", f"{gru['gain']:+.4f} nats [{gru['boot']['day']['lo']:+.4f}, {gru['boot']['day']['hi']:+.4f}]", "results/ml.md"),
        ("ML models significant after BH (day / market)", f"{sum(r['significant']['day'] for r in ml)} / {sum(r['significant']['market'] for r in ml)}", "results/ml.md"),
    ]
    h1 = r2["H1"]["test"]
    lag_k = [
        ("Timestamp bias, block − execution (median)", f"{lag['median']:.2f} s (p1 {lag['q']['0.01']:.2f} s, p99 {lag['q']['0.99']:.2f} s, mean {lag['mean']:.2f} s)", "results/timestamp_lag.json"),
        ("Trades in the timestamp measurement", f"{lag['n']:,}", "data_sample/timestamp_lag_pairs.csv → results/timestamp_lag.json"),
        ("Earliest placement before execution after −2 s / −3 s", f"{2 - lag['q']['0.01']:.2f} s / {3 - lag['q']['0.01']:.2f} s (p1 of the gap)", "results/timestamp_lag.json"),
        ("Lead-lag regression: peak lag after −2 s", f"{h1['beta'].index(max(h1['beta'])) + r2['H1']['lags'][0]} s", "results/results_delta2.json (H1)"),
        ("Lead-lag regression: share of the response after 2 s", f"{h1['share_after_2s']:.0%}", "results/results_delta2.json (H1)"),
        ("Lead-lag (H1) at block time, L = 2 s", f"{ci_c(h1_0, 'day')}, p adj. (day) {fmt_p(r0['bh_day']['S1']['p_adj'])}, p adj. (market) {fmt_p(r0['bh']['S1']['p_adj'])}", "results/results_delta0.json"),
    ]
    for d in ("0", "2", "3"):
        for L, v in lat[d].items():
            lab = "none" if d == "0" else f"−{d} s"
            lag_k.append((f"Lead-lag (H1), re-alignment {lab}, L = {L} s", f"{ci_c(v, 'day')} (day); [{cents(v['market']['lo'])}, {cents(v['market']['hi'])}] (market)", src_lat.format(d=d)))
    a11_k = []
    for v in a11["variants"]:
        t = v["test"]["mean_per_market"]
        lab = "none (block time)" if v["realign_s"] == 0 else f"−{-v['realign_s']} s"
        a11_k.append((f"A11, re-alignment {lab}, Binance horizon t + {v['horizon'].replace('-', ' to t + ')} s",
                      f"{t['est']:.3f}, day CI [{t['day']['lo']:.3f}, {t['day']['hi']:.3f}], market CI [{t['lo']:.3f}, {t['hi']:.3f}], n = {t['n']:,}",
                      "results/a11_robustness.json"))
    return [("Data", data), ("Fair-value model", model_k), ("Hypothesis tests (test period)", tests), ("Machine learning", ml_k),
            ("Timestamp bias and lead-lag", lag_k), ("A11 robustness", a11_k)]


def constants(r2, e2, e8) -> list:
    """Protocol constants quoted in README.md / docs/METHODS.md, with the file that defines them."""
    from analysis import blockboot, run
    S = r2["strategies"]
    return [
        ("Five-minute window, markets per UTC day", f"300 s, {86400 // 300} markets", "config.yaml (duration), analysis/data.py (T)"),
        ("Resolution TWAP / pre-market volatility window", "60-second TWAP; 30-minute realised volatility (σ), 1,800 s", "analysis/run.py (L_TWAP, sigma_pre)"),
        ("Bootstrap draws: P&L by market (H1-H4) / blocks (day, hour)", f"{run.B:,} / {blockboot.B:,}", "analysis/run.py (B), analysis/blockboot.py (B)"),
        ("Bootstrap draws: H1 regression and H2 logit / A9 logit", "500 / 300", "analysis/run.py, analysis/deep_extra.py"),
        ("Benjamini-Hochberg level", f"q = {Q}", "analysis/summary.py (Q)"),
        ("Minimum trades on train for a strategy threshold", "200", "analysis/run.py (main)"),
        ("Validation days for early stopping and stacking", "last 5 days of train", "analysis/deep_cd.py, analysis/deep_stack.py"),
        ("Taker fee", f"{run.FEE_RATE} · p · (1 − p) per share: {run.fee(0.5) * 100:.2f} c at 50 c, {run.fee(0.95) * 100:.2f} c at 95 c", "analysis/run.py (fee), docs/data_notes.md"),
        ("Execution latency L", "2 s (H1-H4), 1 s (A family); grid 0, 1, 2, 3, 5 s", "analysis/run.py, analysis/edges.py"),
        ("Ask validity / bin of the timestamp histogram", "3 s / 0.25 s", "analysis/data.py (ASK_MAX_AGE), analysis/timestamp_lag.py"),
        ("Selected thresholds on train", f"H1 z0 = {S['S1']['selected']} σ in 3 s; H2 θ = {S['S2']['selected'] * 100:.0f} c; "
         f"H3 jump {S['S3']['selected'] * 100:.0f} c in 5 s; H4 t = {S['S4']['selected']} s, ask 0.80-0.98", "results/results_delta2.json (strategies)"),
        ("A1 / A7 thresholds", "buy ≤ 30 c, sell ≥ 33 c / ask ≤ 99 c, model > 99.5 %", "analysis/edges.py"),
        ("A2 grid / A4 test AUC", f"{len(e2['grid'])} cells / {e8['auc_test']:.2f}", "results/edges_E2.json, results/edges_E8.json"),
        ("ML features / snapshots", "28 features; t = 30, 60, ..., 270 s", "analysis/deep_cd.py (_features, SNAPS)"),
        ("ML models", "logistic regression C = 1; MLP (64, 32); GRU hidden size 32 on the last 30 s of 6 channels", "analysis/deep_cd.py"),
        ("Calibration snapshots", "every 10 s", "analysis/run.py (H2)"),
        ("On-chain check of the trade history", "9,018 / 9,018 trades on 4 markets", "docs/data_notes.md"),
        ("Venue outage (no trade, markets dropped by the trade filter)", "2026-09-19", "docs/data_notes.md"),
        ("A11 Binance horizons", "t to t + 10 s; t + 2 to t + 12 s", "analysis/a11_robustness.py (HORIZONS)"),
    ]


def main() -> None:
    dc = data_counts()
    r2 = J("results_delta2")
    hyp = hypotheses()
    ml = ml_rows()
    lat = latency()
    lag = J("timestamp_lag")
    a11 = J("a11_robustness")
    md = mde(hyp)
    K = key_figures(dc, r2, hyp, ml, lat, lag, md, a11) + [("Protocol constants", constants(r2, J("edges_E2"), J("edges_E8")))]
    out = {"data": dc, "meta": r2["meta"], "hypotheses": hyp, "ml": ml, "latency_S1": lat, "timestamp_lag": lag, "mde": md}
    with open(R / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    (R / "hypotheses.md").write_text(md_hyp(hyp), encoding="utf-8")
    (R / "ml.md").write_text(md_ml(ml), encoding="utf-8")
    (R / "latency.md").write_text(md_latency(lat), encoding="utf-8")
    tables = ("\n# Full tables (copies of results/hypotheses.md, ml.md, latency.md)\n\n" + md_hyp(hyp).replace("## ", "### ")
              + "\n### ML and DL models\n\n" + md_ml(ml) + "\n### Lead-lag strategy by re-alignment and latency\n\n"
              + md_latency(lat))
    (R / "key_figures.md").write_text("# Key figures\n\n" + md_key(K) + tables, encoding="utf-8")
    print(md_key(K))


if __name__ == "__main__":
    main()
