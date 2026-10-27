"""Robustness of A11: does the price residual lead the next Binance move, or is the lead a timestamp artefact?

A11 (analysis/deep.py, section A) correlates, within each market, the 10 s change of the residual e (market logit not
explained by the Binance fair value) with the Binance log-return of the next 10 s. Trade timestamps are block times,
re-aligned by -2 s. The measured block - execution gap has p1 = 1.09 s and p99 = 3.47 s, so after a -2 s shift some
trades sit up to ~0.9 s before their real execution. Their price can then "anticipate" Binance moves that happened
before the trade, which creates a mechanical lead.

Variants (all on the same markets, same t grid, residual map refitted on train for each re-alignment):
- re-alignment of the market series: 0 s (block time: no trade is placed before its execution), -2 s (reference),
  -3 s (trades up to ~1.9 s early);
- Binance horizon: [t, t + 10] (reference) or [t + 2, t + 12] (skips the first 2 s, longer than any early placement
  under -2 s or -3 s);
- CI: bootstrap by market, by UTC day (reference) and by UTC hour.

Output: results/a11_robustness.json. Usage: python -m analysis.a11_robustness
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from analysis import blockboot as bb
from analysis.deep import Ctx, boot_stat, logit, nan_corr
from analysis.deep_data import DELTA

OUT = Path("results")
HORIZONS = {"0-10": (0, 10), "2-12": (2, 12)}
REALIGN = (2, 3, 0)


def shift(a: np.ndarray, k: int) -> np.ndarray:
    """out[:, t] = a[:, t + k] (NaN outside the window)."""
    out = np.full_like(a, np.nan, dtype=np.float64)
    if k >= 0:
        out[:, : a.shape[1] - k] = a[:, k:]
    else:
        out[:, -k:] = a[:, :k]
    return out


def residual(c: Ctx, x: np.ndarray) -> np.ndarray:
    """Market logit minus its linear map on the Binance fair-value logit, fitted on train (t >= 10)."""
    ok = np.isfinite(x) & np.isfinite(c.z) & c.train[:, None]
    ok[:, :10] = False
    mp = np.polyfit(c.z[ok], x[ok], 1)
    return x - np.polyval(mp, c.z)


def lead(c: Ctx, e: np.ndarray, h0: int, h1: int) -> dict:
    m = c.D["m"]
    res = {}
    for name, msk in c.splits():
        de = e[msk][:, 20:200] - e[msk][:, 10:190]
        fut = m[msk][:, 20 + h1:200 + h1] - m[msk][:, 20 + h0:200 + h0]
        per = np.array([nan_corr(de[i], fut[i]) if np.isfinite(de[i]).sum() > 20 else np.nan for i in range(de.shape[0])])
        row = {"pooled": nan_corr(de, fut), "mean_per_market": boot_stat(per, start_s=c.D["start_s"][msk])}
        if name == "test":
            row["placebo_past_10"] = nan_corr(de, m[msk][:, 20:200] - m[msk][:, 10:190])
        res[name] = row
    return res


def main() -> None:
    t0 = time.time()
    c = Ctx()
    pm = c.D["pm"]                       # market price re-aligned by DELTA = 2 s (analysis/deep_data.py)
    out = {"note": "pm(delta)[:, t] = pm(2)[:, t + delta - 2]; corr = mean per-market correlation", "variants": []}
    for delta in REALIGN:
        e = residual(c, logit(shift(pm, delta - DELTA)))
        for hz, (h0, h1) in HORIZONS.items():
            r = lead(c, e, h0, h1)
            out["variants"].append({"realign_s": -delta, "horizon": hz, **r})
            t = r["test"]["mean_per_market"]
            print(f"realign -{delta} s, Binance [t+{h0}, t+{h1}]: corr {t['est']:.4f} market [{t['lo']:.4f}, {t['hi']:.4f}] "
                  f"day [{t['day']['lo']:.4f}, {t['day']['hi']:.4f}]", flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    with open(OUT / "a11_robustness.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
