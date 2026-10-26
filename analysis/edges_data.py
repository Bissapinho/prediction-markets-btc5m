"""Prix exécutables par seconde, pré-ouverture comprise (results/edges_cache.npz).

Tout est exprimé en « équivalent Up » : u = prix si le token est Up, 1 − prix si c'est Down.
- Pression acheteuse Up : achat taker Up ou vente taker Down (la contrepartie maker vendait Up à u).
- Pression vendeuse Up : vente taker Up ou achat taker Down (la contrepartie maker achetait Up à u).

Par seconde s ∈ [−300, 305) (recalée de 2 s) :
- buy_max / buy_min : prix max / min des trades à pression acheteuse Up (ask Up touché) ;
- sell_min / sell_max : prix min / max des trades à pression vendeuse Up (bid Up touché).
Un ordre maker d'achat Up à b est considéré exécuté seulement si un trade vendeur passe STRICTEMENT sous b
(sell_min < b), et une vente maker à a si un trade acheteur passe strictement au-dessus (buy_max > a) :
hypothèse prudente qui ignore la file d'attente au même prix.
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
    press_up = up == buy                      # achat Up ou vente Down
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
