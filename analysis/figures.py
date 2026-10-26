"""Figures of the README (figures/*.png), from results/*.json. Usage : python -m analysis.figures"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R, F = Path("results"), Path("figures")
Z95 = 1.959964
INK, GRID = "#1f2933", "#d9dee3"
C = {"0": "#9aa5b1", "2": "#2563eb", "3": "#d97706", "market": "#2563eb", "model": "#d97706"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def J(name):
    with open(R / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def calibration():
    cal = J("results_delta2")["H2"]["calibration"]
    snaps = {"debut (10-90 s)": 9, "milieu (100-230 s)": 14, "fin (240-290 s)": 6}
    titles = {"debut (10-90 s)": "t = 10-90 s", "milieu (100-230 s)": "t = 100-230 s", "fin (240-290 s)": "t = 240-290 s"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, (g, d) in zip(axes, cal.items()):
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls="--")
        for src, lab in (("market", "market price"), ("model", "TWAP model (Binance)")):
            b = d[src]
            p = np.array([x["p_mean"] for x in b]); yv = np.array([x["y_mean"] for x in b])
            n_eff = np.array([x["n"] for x in b]) / snaps[g]
            half = Z95 * np.sqrt(np.clip(yv * (1 - yv), 1e-6, None) / n_eff)
            ax.errorbar(p, yv, yerr=half, fmt="o-", ms=3.5, lw=1.2, capsize=2, color=C[src], label=lab)
        ax.set_title(titles[g], fontsize=10)
        ax.set_xlabel("predicted P(Up)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("realised frequency of Up")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Calibration on the test period (7,775 markets)", x=0.07, ha="left", fontsize=11)
    fig.savefig(F / "fig1_calibration.png"); plt.close(fig)


def lead_lag():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    off = {"0": -0.18, "2": 0.0, "3": 0.18}
    for d, lab in (("0", "no re-alignment (block time)"), ("2", "re-aligned −2 s (reference)"), ("3", "re-aligned −3 s")):
        h = J(f"results_delta{d}")["H1"]
        lags = np.array(h["lags"]) + off[d]; t = h["test"]
        b, lo, hi = np.array(t["beta"]), np.array(t["lo"]), np.array(t["hi"])
        ax.errorbar(lags, b, yerr=[b - lo, hi - b], fmt="o-", ms=3.5, lw=1.2, capsize=2, color=C[d], label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=INK, lw=0.6, ls=":")
    ax.set_xlabel("lag of the Binance 1 s return (s); positive = the market moves after Binance")
    ax.set_ylabel("response of Δp (per unit of P_model move)")
    ax.set_title("Lead-lag: response of the market price to Binance returns (test, 95 % cluster-bootstrap CI)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.savefig(F / "fig2_leadlag.png"); plt.close(fig)


def latency():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    off = {"0": -0.08, "2": 0.0, "3": 0.08}
    for d, lab in (("0", "no re-alignment (block time)"), ("2", "re-aligned −2 s (reference)"), ("3", "re-aligned −3 s")):
        lat = J(f"results_delta{d}")["strategies"]["S1"]["latency"]
        L = np.array([float(k) for k in lat]) + off[d]
        m = np.array([v["mean"] for v in lat.values()]) * 100
        lo = np.array([v["lo"] for v in lat.values()]) * 100; hi = np.array([v["hi"] for v in lat.values()]) * 100
        ax.errorbar(L, m, yerr=[m - lo, hi - m], fmt="o-", ms=4, lw=1.2, capsize=2.5, color=C[d], label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("execution latency L (s): signal at t, market order filled at the ask observed at t + L")
    ax.set_ylabel("mean P&L (cents per share, fees included)")
    ax.set_title("Lead-lag strategy (H1) on the test period, by latency (95 % CI, bootstrap by market)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.savefig(F / "fig3_latency.png"); plt.close(fig)


def ml():
    rows = J("summary")["ml"]
    labels = [f"{r['model']} — {r['setting']}" for r in rows]
    g = np.array([r["gain"] for r in rows]) * 1000
    lo = np.array([r["lo"] for r in rows]) * 1000; hi = np.array([r["hi"] for r in rows]) * 1000
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    col = ["#2563eb" if r["significant"] else "#9aa5b1" for r in rows]
    for yi, gi, l_, h_, c_ in zip(y, g, lo, hi, col):
        ax.errorbar(gi, yi, xerr=[[gi - l_], [h_ - gi]], fmt="o", ms=5, capsize=3, color=c_)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("log-loss gain over the market price (10⁻³ nats per prediction; > 0 = better than the market)")
    ax.set_title("Outcome forecasts vs market price, test period (95 % CI by market; blue = BH-significant)", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.savefig(F / "fig4_ml_vs_market.png"); plt.close(fig)


def timestamp_lag():
    t = J("timestamp_lag")
    lo = [h["lo"] for h in t["hist"]]; n = [h["n"] for h in t["hist"]]
    w = t["hist"][0]["hi"] - t["hist"][0]["lo"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(lo, n, width=w, align="edge", color="#2563eb", edgecolor="white")
    ax.axvline(t["median"], color="#d97706", lw=1.5, label=f"median {t['median']:.2f} s")
    ax.set_xlabel("block timestamp − execution timestamp (s)")
    ax.set_ylabel("trades")
    ax.set_title(f"Timestamp bias of the trade history ({t['n']:,} trades matched on transaction hash)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.savefig(F / "fig5_timestamp_bias.png"); plt.close(fig)


def main():
    F.mkdir(exist_ok=True)
    for fn in (calibration, lead_lag, latency, ml, timestamp_lag):
        fn()
        print("ok", fn.__name__)


if __name__ == "__main__":
    main()
