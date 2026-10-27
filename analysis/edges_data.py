"""Executable prices per second, pre-open included (results/edges_cache.npz).

Everything is expressed as "Up equivalent": u = price if the token is Up, 1 − price if it is Down.
- Up buying pressure: taker Up buy or taker Down sell (the maker counterparty was selling Up at u).
- Up selling pressure: taker Up sell or taker Down buy (the maker counterparty was buying Up at u).

Per second s ∈ [−300, 305) (re-aligned by 2 s):
- buy_max / buy_min: max / min price of the trades with Up buying pressure (Up ask hit);
- sell_min / sell_max: min / max price of the trades with Up selling pressure (Up bid hit).
A maker buy order for Up at b counts as filled only if a selling trade goes STRICTLY below b (sell_min < b), and a
maker sell at a only if a buying trade goes strictly above (buy_max > a): a prudent assumption that ignores the queue
at the same price.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.deep_data import load_cache

S0, S1 = -300, 305
W = S1 - S0
CACHE = Path("results/edges_cache.npz")


def build() -> dict:
    D = load_cache()
    n = len(D["y"])
    s = D["tr_s"].astype(np.int32)
    ok = (s >= S0) & (s < S1)
    i = D["tr_i"][ok]; col = s[ok] - S0
    up = D["tr_up"][ok]; buy = D["tr_buy"][ok]; p = D["tr_price"][ok].astype(np.float64)
    q = D["tr_size"][ok].astype(np.float64)
    u = np.where(up, p, 1 - p)
    press_up = up == buy                      # Up buy or Down sell
    flat = i.astype(np.int64) * W + col
    def agg(mask, fn, init):
        a = np.full(n * W, init)
        fn.at(a, flat[mask], u[mask])
        a[a == init] = np.nan
        return a.reshape(n, W)
    out = {
        "buy_max": agg(press_up, np.maximum, -1.0), "buy_min": agg(press_up, np.minimum, 9.0),
        "sell_min": agg(~press_up, np.minimum, 9.0), "sell_max": agg(~press_up, np.maximum, -1.0),
        "vol_sh": np.bincount(flat, q, minlength=n * W).reshape(n, W),
    }
    np.savez_compressed(CACHE, **out)
    return out


def load() -> dict:
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    o = build()
    for k, v in o.items():
        print(k, v.shape, float(np.isfinite(v).mean()) if v.dtype.kind == "f" else "")
