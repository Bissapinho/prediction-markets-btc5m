"""Tables of the README, recomputed from the analysis outputs (results/*.json) and the parquet files.

Writes results/summary.json, results/hypotheses.md, results/ml.md, results/latency.md and results/key_figures.md.
Usage : python -m analysis.summary

p-values. When an analysis stores its own cluster-bootstrap p-value (one-sided, H0: mean P&L <= 0), it is used as is.
Otherwise the p-value comes from a normal approximation of the bootstrap distribution: se = (hi - lo) / (2 * 1.96).
Multiple testing. Benjamini-Hochberg (q = 0.05) within each family; the pre-registered family of the protocol
(H2 information test + the four strategies S1-S4) keeps the adjustment computed by analysis.run.
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


def J(name: str) -> dict:
    with open(R / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def p_normal(est, lo, hi, h0=0.0, sided="greater"):
    se = (hi - lo) / (2 * Z95)
    z = (est - h0) / se
    return float(1 - ndtr(z)) if sided == "greater" else float(2 * (1 - ndtr(abs(z))))


def cents(x):
    return "–" if x is None else f"{x * 100:+.1f}"


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
    fmt = lambda ms: str(np.datetime64(int(ms), "ms"))[:16].replace("T", " ")
    return {"n_windows": n_windows, "n_markets_listed": int(n_period), "n_unresolved": int(n_unresolved),
            "n_no_trade_in_window": int(n_no_trade), "n_no_binance": int(n_period - n_unresolved - n_no_trade - n_kept),
            "n_markets": int(n_kept), "first_start_utc": fmt(first_ms), "last_start_utc": fmt(last_ms),
            "n_trades": int(n_trades), "n_trades_in_window": int(n_trades_window), "notional_usdc": float(notional),
            "rows_market_1s": int(rows_m1s), "rows_btc_1s": int(rows_b1s), "rows_per_second_total": int(rows_m1s + rows_b1s),
            "n_binance_aggtrades": int(n_agg)}


# --------------------------------------------------------------------------------------------------------------------
# Hypotheses
# --------------------------------------------------------------------------------------------------------------------

def hypotheses() -> list[dict]:
    r2 = J("results_delta2")
    e1, e2, e3, e4, e6, e8 = (J(f"edges_{s}") for s in ("E1", "E2", "E3", "E4", "E6", "E8"))
    dA, dX, dD = J("deep_A"), J("deep_C_extra"), J("deep_D")
    S = r2["strategies"]
    rows: list[dict] = []

    def pnl_row(hid, fam, title, test, execution, selected=None):
        rows.append({"id": hid, "family": fam, "title": title, "kind": "pnl", "execution": execution, "selected": selected,
                     "stat": "mean P&L per share, fees included", "est": test.get("mean"), "lo": test.get("lo"),
                     "hi": test.get("hi"), "n": test.get("n"), "p": test.get("p")})

    # Family P: pre-registered protocol (BH already computed in analysis.run).
    pnl_row("H1", "P", "Lead-lag: buy when Binance moves >= z0 sigma in 3 s and the market has not moved", S["S1"]["test"], "taker, 2 s", S["S1"]["selected"])
    pnl_row("H2", "P", "TWAP fair value from Binance: buy the side priced below the model", S["S2"]["test"], "taker, 2 s", S["S2"]["selected"])
    h2 = r2["H2"]["pooled_test"]
    rows.append({"id": "H2-info", "family": "P", "title": "Model minus market predicts the outcome (logit coefficient c > 0)",
                 "kind": "info", "stat": "coefficient c", "est": h2["c"], "lo": h2["c_lo"], "hi": h2["c_hi"], "n": h2["n_obs"],
                 "p": h2["p_c"], "execution": "–", "selected": None})
    pnl_row("H3", "P", "Fade price jumps that Binance does not justify", S["S3"]["test"], "taker, 2 s", S["S3"]["selected"])
    pnl_row("H4", "P", "Late favourites (t = 240-290 s, ask 0.80-0.98)", S["S4"]["test"], "taker, 2 s", S["S4"]["selected"])
    key = {"H1": "S1", "H2": "S2", "H3": "S3", "H4": "S4", "H2-info": "H2_c"}
    for row in rows:
        b = r2["bh"][key[row["id"]]]
        row["p"], row["p_adj"], row["significant"] = b["p"], b["p_adj"], b["significant"]

    # Family A: price-path strategies.
    pnl_row("A1", "A", "Dip-buy: buy a side below 30 c, sell at 33 c", e2["case_30_33_taker_test"], "taker, 1 s")
    sel = e2["selected"]
    pnl_row("A2", "A", f"Dip-buy grid ({len(e2['grid'])} cells), best cell on train", sel["test"], "limit order",
            f"{sel['window']}, x = {sel['x']}, k = {sel['k']}")
    cl = max(e6["clusters"], key=lambda c: c["train"]["dip_30_33"] if c["train"]["dip_30_33"] is not None else -9)
    pnl_row("A3", "A", "Dip-buy restricted to the 'hesitant' market cluster (defined ex post)", cl["test"]["dip_30_33_boot"],
            "limit order", f"cluster {cl['cluster']}")
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
                 "stat": "share of seconds with ask_up + ask_dn < 1", "est": pa["share_sum_lt_1"], "lo": None, "hi": None,
                 "n": pa["seconds_with_both"], "p": None, "p_adj": None, "significant": None, "execution": "–", "selected": None})

    # Family I: information beyond the price (statistical, not P&L).
    os_ = dX["test"]["logit_y_on_price_and_streak"]
    rows.append({"id": "A9", "family": "I", "title": "Under-reaction at the open: slope of outcome on logit(price), 10-40 s (H0: slope = 1)",
                 "kind": "info", "stat": "logit slope", "est": os_["beta"][1], "lo": os_["lo"][1], "hi": os_["hi"][1],
                 "n": None, "p": p_normal(os_["beta"][1], os_["lo"][1], os_["hi"][1], h0=1.0, sided="two"),
                 "execution": "–", "selected": None})
    st = dD["D1_stacked"]["logistique"]
    rows.append({"id": "A10", "family": "I", "title": "ML forecast of the outcome stacked on the market price",
                 "kind": "info", "stat": "log-loss gain vs market (nats)", "est": st["gain"], "lo": st["lo"], "hi": st["hi"],
                 "n": dD["D1_outcome"]["n_test_rows"], "p": p_normal(st["gain"], st["lo"], st["hi"]), "execution": "–", "selected": None})
    ld = dA["resid_leads_btc"]["test"]["10"]["mean_per_market"]
    rows.append({"id": "A11", "family": "I", "title": "Price residual (not explained by Binance) leads the next 10 s Binance move",
                 "kind": "info", "stat": "mean per-market correlation", "est": ld["est"], "lo": ld["lo"], "hi": ld["hi"],
                 "n": ld["n"], "p": p_normal(ld["est"], ld["lo"], ld["hi"], sided="two"), "execution": "–", "selected": None})

    for fam in ("A", "I"):
        ps = {r["id"]: r["p"] for r in rows if r["family"] == fam and r["p"] is not None}
        adj = bh(ps, Q)
        for r in rows:
            if r["id"] in adj:
                r["p_adj"], r["significant"] = adj[r["id"]]["p_adj"], adj[r["id"]]["significant"]
    for r in rows:
        r.setdefault("p_adj", None); r.setdefault("significant", None)
        r["verdict"] = verdict(r)
    return rows


def verdict(r: dict) -> str:
    if r["kind"] == "artefact":
        return "Artefact: the two prices come from different instants of the same second"
    if r["est"] is None:
        return "Not testable (fewer than 5 trades)"
    if r["kind"] == "pnl":
        if r["significant"] and r["est"] > 0:
            return "Positive after BH" + (" (ex-post cluster, not tradable)" if r["id"] == "A3" else "")
        if r["est"] <= 0:
            return "Rejected: negative P&L"
        return "Not significant after BH"
    if r["significant"]:
        if r["id"] == "H2-info":
            return "Information detected after BH; the taker strategy built on it (H2) loses"
        return "Information detected after BH (statistical test, not a P&L test)"
    return "No information detected after BH"


# --------------------------------------------------------------------------------------------------------------------
# ML / DL and latency
# --------------------------------------------------------------------------------------------------------------------

NAMES = {"logistique": "Logistic regression", "gradient boosting": "Gradient boosting (HistGB)", "MLP": "MLP (64, 32)",
         "marché recalibré": "Market price, recalibrated (logistic on logit p)"}


def ml_rows() -> list[dict]:
    dD = J("deep_D")
    rows = []
    for k, v in dD["D1_outcome"]["models"].items():
        name = "GRU (30 s sequences + static features)" if k.startswith("GRU") else NAMES[k]
        rows.append({"model": name, "setting": "standalone", **{x: v[x] for x in ("gain", "lo", "hi", "logloss_model", "logloss_market")}})
    for k, v in dD["D1_stacked"].items():
        rows.append({"model": NAMES[k], "setting": "stacked on market price", **{x: v[x] for x in ("gain", "lo", "hi", "logloss_model", "logloss_market")}})
    ps = {i: p_normal(r["gain"], r["lo"], r["hi"]) for i, r in enumerate(rows)}
    adj = bh(ps, Q)
    for i, r in enumerate(rows):
        r["p"], r["p_adj"], r["significant"] = ps[i], adj[i]["p_adj"], adj[i]["significant"]
    return rows


def latency() -> dict:
    out = {}
    for d in (0, 2, 3):
        lat = J(f"results_delta{d}")["strategies"]["S1"]["latency"]
        out[str(d)] = {L: {k: v[k] for k in ("mean", "lo", "hi", "n", "p")} for L, v in lat.items()}
    return out


def mde(rows) -> dict:
    """Effect detectable with 80 % power, one-sided 5 % test: (z_0.95 + z_0.80) * se."""
    out = {}
    for r in rows:
        if r["family"] == "P" and r["kind"] == "pnl":
            out[r["id"]] = (1.644854 + 0.841621) * (r["hi"] - r["lo"]) / (2 * Z95)
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


def fmt_ci(r):
    if r["lo"] is None:
        return "–"
    if r["kind"] == "pnl":
        return f"[{cents(r['lo'])}, {cents(r['hi'])}]"
    f = (lambda x: f"{x:.4f}") if abs(r["est"]) < 0.01 else (lambda x: f"{x:.3f}")
    return f"[{f(r['lo'])}, {f(r['hi'])}]"


def fmt_p(p):
    return "–" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def md_hyp(rows) -> str:
    fam = {"P": "P (pre-registered)", "A": "A (price paths)", "I": "I (information)", "–": "–"}
    out = ["| ID | Hypothesis | Execution | Statistic (test) | Estimate | 95 % CI | n | p | p adj. (BH) | Family | Verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n = "–" if r["n"] is None else f"{r['n']:,}"
        out.append(f"| {r['id']} | {r['title']} | {r['execution']} | {r['stat']} | {fmt_est(r)} | {fmt_ci(r)} | {n} | "
                   f"{fmt_p(r['p'])} | {fmt_p(r['p_adj'])} | {fam[r['family']]} | {r['verdict']} |")
    return "\n".join(out) + "\n"


def md_ml(rows) -> str:
    out = ["| Model | Setting | Log-loss (model) | Log-loss (market) | Gain (nats) | 95 % CI | p adj. (BH) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['model']} | {r['setting']} | {r['logloss_model']:.4f} | {r['logloss_market']:.4f} | "
                   f"{r['gain']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {fmt_p(r['p_adj'])} |")
    return "\n".join(out) + "\n"


def md_latency(lat) -> str:
    Ls = list(lat["2"])
    out = ["| Re-alignment | " + " | ".join(f"L = {L} s" for L in Ls) + " |", "|---|" + "---|" * len(Ls)]
    for d, lab in (("0", "none (block time)"), ("2", "−2 s (reference)"), ("3", "−3 s")):
        out.append(f"| {lab} | " + " | ".join(f"{cents(v['mean'])} c [{cents(v['lo'])}, {cents(v['hi'])}]" for v in lat[d].values()) + " |")
    return "\n".join(out) + "\n"


def md_key(K) -> str:
    out = ["| Figure | Value | Source |", "|---|---|---|"]
    for name, val, src in K:
        out.append(f"| {name} | {val} | {src} |")
    return "\n".join(out) + "\n"


def main() -> None:
    dc = data_counts()
    r2 = J("results_delta2")
    hyp = hypotheses()
    ml = ml_rows()
    lat = latency()
    lag = J("timestamp_lag")
    md = mde(hyp)
    meta = r2["meta"]
    best_ml = max(ml, key=lambda r: r["gain"])
    gru = next(r for r in ml if r["model"].startswith("GRU"))
    l1, l2, l1_3 = lat["2"]["1"], lat["2"]["2"], lat["3"]["1"]
    n_tested = sum(1 for r in hyp if r["kind"] != "artefact")
    K = [
        ("Markets analysed", f"{dc['n_markets']:,}", "data/clean/markets + filters of analysis/data.py → results/summary.json"),
        ("Period (market start, UTC)", f"{dc['first_start_utc']} → {dc['last_start_utc']}", "results/summary.json"),
        ("Train / test markets (split 2026-09-21)", f"{meta['n_train']:,} / {meta['n_test']:,}", "results/results_delta2.json (meta)"),
        ("Prediction Market trades (analysed markets)", f"{dc['n_trades']:,}", "data/clean/trades → results/summary.json"),
        ("of which inside the 5-min window", f"{dc['n_trades_in_window']:,}", "results/summary.json"),
        ("Traded notional", f"{dc['notional_usdc'] / 1e6:,.1f} M USDC", "results/summary.json"),
        ("Binance BTCUSDT aggregated trades", f"{dc['n_binance_aggtrades']:,}", "data/clean/btc_aggtrades → results/summary.json"),
        ("Per-second rows (market × second)", f"{dc['rows_market_1s']:,}", "data/features/market_1s → results/summary.json"),
        ("Per-second rows (Binance 1 s)", f"{dc['rows_btc_1s']:,}", "data/features/btc_1s → results/summary.json"),
        ("Per-second rows, total", f"{dc['rows_per_second_total']:,}", "results/summary.json"),
        ("Hypotheses / tests in the public table", f"{len({r['id'].split('-')[0] for r in hyp})} hypotheses, {n_tested} tests + 1 artefact (A8)", "results/hypotheses.md"),
        ("Hypotheses significant after BH with positive taker P&L", str(sum(1 for r in hyp if r['kind'] == 'pnl' and r['significant'] and r['est'] > 0 and r['execution'].startswith('taker'))), "results/hypotheses.md"),
        ("Best ML log-loss gain vs market", f"{best_ml['gain']:+.4f} nats ({best_ml['model']}, {best_ml['setting']})", "results/deep_D.json → results/ml.md"),
        ("GRU log-loss gain vs market", f"{gru['gain']:+.4f} nats [{gru['lo']:+.4f}, {gru['hi']:+.4f}]", "results/deep_D.json → results/ml.md"),
        ("Timestamp bias, block − execution (median)", f"{lag['median']:.2f} s (p1 {lag['q']['0.01']:.2f} s, p99 {lag['q']['0.99']:.2f} s)", "results/timestamp_lag.json"),
        ("Trades in the timestamp measurement", f"{lag['n']:,}", "data_sample/timestamp_lag_pairs.csv"),
        ("Lead-lag (H1), L = 1 s, −2 s re-alignment", f"{cents(l1['mean'])} c/share [{cents(l1['lo'])}, {cents(l1['hi'])}]", "results/results_delta2.json"),
        ("Lead-lag (H1), L = 1 s, −3 s re-alignment", f"{cents(l1_3['mean'])} c/share [{cents(l1_3['lo'])}, {cents(l1_3['hi'])}]", "results/results_delta3.json"),
        ("Lead-lag (H1), L = 2 s, −2 s re-alignment", f"{cents(l2['mean'])} c/share [{cents(l2['lo'])}, {cents(l2['hi'])}]", "results/results_delta2.json"),
        ("Detectable edge (80 % power), H1-H4", f"{min(md.values()) * 100:.1f} to {max(md.values()) * 100:.1f} c/share", "results/summary.json"),
    ]
    out = {"data": dc, "meta": meta, "hypotheses": hyp, "ml": ml, "latency_S1": lat, "timestamp_lag": lag, "mde": md}
    with open(R / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    (R / "hypotheses.md").write_text(md_hyp(hyp), encoding="utf-8")
    (R / "ml.md").write_text(md_ml(ml), encoding="utf-8")
    (R / "latency.md").write_text(md_latency(lat), encoding="utf-8")
    (R / "key_figures.md").write_text(md_key(K), encoding="utf-8")
    print(md_key(K))


if __name__ == "__main__":
    main()
