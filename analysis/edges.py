"""Price-path strategies A1-A8: conditional probabilities and executable strategies (results/edges_*.json).

Usage: python -m analysis.edges [E1 E2 ...]
Conventions: series re-aligned by 2 s; executable prices from analysis/edges_data.py;
exploration < 2026-09-21 ≤ test; parameters chosen on the exploration period; bootstrap CIs by market, UTC day and
UTC hour. Taker fee 0.07·p(1−p) per share; limit orders: no fee.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from analysis import blockboot as bb
from analysis.deep import RNG, Ctx, boot_stat, nan_corr
from analysis.edges_data import S0, load

OUT = Path("results")
W0 = -S0                       # indice de s = 0
FEE = 0.07


def fee(p):
    return FEE * p * (1 - p)


def mboot(pnl, mk, b=1000, start_s=None):
    """Mean P&L per trade, cluster-bootstrap CI by market (plus UTC day and UTC hour once X() has run)."""
    pnl = np.asarray(pnl, float); mk = np.asarray(mk)
    if len(pnl) < 5:
        return {"n": int(len(pnl)), "mean": None, "lo": None, "hi": None, "p": None}
    um, inv = np.unique(mk, return_inverse=True)
    s = np.bincount(inv, pnl); c = np.bincount(inv)
    bs = np.array([(s[j].sum() / c[j].sum()) for j in (RNG.integers(0, len(um), len(um)) for _ in range(b))])
    out = {"n": int(len(pnl)), "n_markets": int(len(um)), "mean": float(pnl.mean()), "lo": float(np.quantile(bs, .025)),
           "hi": float(np.quantile(bs, .975)), "p": float((np.sum(bs <= 0) + 1) / (b + 1)), "hit": float((pnl > 0).mean())}
    if start_s is not None:
        out.update(bb.blocks_mean(pnl, start_s[mk]))
    return out


class X:
    """Context: Ctx of the per-second analyses + executable prices per side."""

    def __init__(self):
        self.c = c = Ctx()
        self.start_s = c.D["start_s"]     # market start (Unix s), for the block bootstraps
        E = load()
        self.y = c.y
        # For each side: sl = price of the trades where the token is SOLD by a taker (a maker buys it),
        #                bh = price of the trades where it is BOUGHT by a taker (a maker sells it).
        self.side = {
            "up": {"sl": E["sell_min"], "bh": E["buy_max"], "ask": E["buy_max"], "y": c.y},
            "down": {"sl": 1 - E["buy_max"], "bh": 1 - E["sell_min"], "ask": 1 - E["sell_min"], "y": 1 - c.y},
        }
        self.lo = np.fmin(E["buy_min"], E["sell_min"]); self.hi = np.fmax(E["buy_max"], E["sell_max"])
        self.vol = E["vol_sh"]
        self.P = c.D["P"]


# =========================================================================== E1 conditional probabilities


def E1(x: X) -> dict:
    c = x.c
    res = {}
    win = slice(W0, W0 + 300)
    L, H = x.lo[:, win], x.hi[:, win]
    sides = {"up": (L, H, c.y), "down": (1 - H, 1 - L, 1 - c.y)}

    # (a) the intuition: a side goes ≤ x then back ≥ a before the end
    def rebound_table(msk, entry_max=300):
        rows = []
        for xx in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            for k in (0.01, 0.02, 0.03, 0.05, 0.10):
                a = xx + k
                emp, theo, ph_all, winr, n = [], [], [], [], 0
                for nm, (lw, hg, yy) in sides.items():
                    lw2, hg2, yy2 = lw[msk], hg[msk], yy[msk]
                    m = np.nan_to_num(lw2, nan=9) <= xx
                    touched = m.any(1); first = np.argmax(m, 1)
                    sel = touched & (first < entry_max) & (first < 299)
                    idx = np.nonzero(sel)[0]
                    ph = lw2[idx, first[idx]]
                    # max price after the first touch
                    hg_after = np.where(np.arange(300)[None, :] > first[idx][:, None], np.nan_to_num(hg2[idx], nan=-1), -1).max(1)
                    emp.append(hg_after >= a); theo.append(ph / a); ph_all.append(ph); winr.append(yy2[idx]); n += len(idx)
                emp = np.concatenate(emp); theo = np.concatenate(theo); ph_all = np.concatenate(ph_all); winr = np.concatenate(winr)
                rows.append({"x": xx, "a": round(a, 2), "n": int(n), "p_rebound": float(emp.mean()), "p_martingale": float(theo.mean()),
                             "gap": float(emp.mean() - theo.mean()), "avg_touch_price": float(ph_all.mean()), "p_win": float(winr.mean())})
        return rows

    for name, msk in c.splits():
        res[f"rebound_{name}"] = rebound_table(msk, 240)
    # frequency of the event "at least one side ≤ 0.30 then ≥ 0.33" per market
    for name, msk in c.splits():
        ev = np.zeros(msk.sum(), bool)
        for nm, (lw, hg, yy) in sides.items():
            lw2, hg2 = lw[msk], hg[msk]
            m = np.nan_to_num(lw2, nan=9) <= 0.30; touched = m.any(1); first = np.argmax(m, 1)
            after = np.where(np.arange(300)[None, :] > first[:, None], np.nan_to_num(hg2, nan=-1), -1).max(1)
            ev |= touched & (after >= 0.33)
        res[f"claim_30_33_{name}"] = float(ev.mean())

    # (b) A given B: outcome given the path
    pm = c.D["pm"]
    ab = {}
    for name, msk in c.splits():
        p = pm[msk]; yy = c.y[msk]
        cross = np.nansum(np.abs(np.diff(np.sign(p[:, 10:290] - 0.5), axis=1)) > 0, axis=1)
        lead60 = np.where(p[:, 60] >= 0.5, 1, 0); lead240 = np.where(p[:, 240] >= 0.5, 1, 0)
        r = {}
        r["P(Up | Up favourite at 60 s)"] = boot_stat(yy[lead60 == 1])
        r["P(favourite at 60 s wins)"] = boot_stat((yy == lead60).astype(float))
        r["P(favourite at 240 s wins)"] = boot_stat((yy == lead240).astype(float))
        r["mean price of the favourite at 240 s"] = boot_stat(np.where(lead240 == 1, p[:, 240], 1 - p[:, 240]))
        for k in (0, 1, 2, 4, 8):
            sel = cross >= k if k else cross == 0
            r[f"share of markets with ≥{k} crossings of 50 %" if k else "share without a crossing of 50 % (10-290 s)"] = float(sel.mean())
        # reversal: favourite ≥ 0.80 at some point, then loses
        fav80 = (np.nanmax(p[:, 10:290], 1) >= 0.8) & (yy == 0) | (np.nanmin(p[:, 10:290], 1) <= 0.2) & (yy == 1)
        r["P(a side was worth ≥ 80 c then lost)"] = float(fav80.mean())
        fav90 = (np.nanmax(p[:, 10:290], 1) >= 0.9) & (yy == 0) | (np.nanmin(p[:, 10:290], 1) <= 0.1) & (yy == 1)
        r["P(a side was worth ≥ 90 c then lost)"] = float(fav90.mean())
        # martingale theory: P(reversal after ≥ 0.9) ≤ 0.1 × P(reach 0.9); measured ratio
        reach90 = (np.nanmax(p[:, 10:290], 1) >= 0.9) | (np.nanmin(p[:, 10:290], 1) <= 0.1)
        r["P(loses | reached 90 c)"] = float(fav90.sum() / max(reach90.sum(), 1))
        ab[name] = r
    res["path_conditionals"] = ab

    # (c) implied touch probability: the price at t implies P(touch a before the end) = p/a;
    #     empirical comparison by starting time
    tt = []
    for t in (30, 90, 150, 210, 250):
        for name, msk in c.splits():
            p0 = pm[msk][:, t]; hh = x.hi[msk][:, W0 + t + 1:W0 + 300]; ll = x.lo[msk][:, W0 + t + 1:W0 + 300]
            up_t = np.nanmax(np.nan_to_num(hh, nan=-1), 1); dn_t = np.nanmin(np.nan_to_num(ll, nan=9), 1)
            for dlt in (0.05, 0.10):
                a = np.clip(p0 + dlt, 0, 1); sel = np.isfinite(p0) & (p0 + dlt < 0.98) & (p0 > 0.05)
                tt.append({"t": t, "split": name, "delta": dlt, "emp_up": float((up_t[sel] >= a[sel]).mean()),
                           "theo_up": float((p0[sel] / a[sel]).mean())})
    res["touch_implied"] = tt
    return res


# =========================================================================== E2 dip-buy (limit buy at x, exit at x + k)


def _dip_sim(x: X, msk, xx, k, e0, e1, sides=("up", "down")):
    pnl, mk = [], []
    tgrid = np.arange(W0, W0 + 300)
    for sd in sides:
        S = x.side[sd]
        sl = S["sl"][msk][:, tgrid]; bh = S["bh"][msk][:, tgrid]; yy = S["y"][msk]
        through = np.nan_to_num(sl, nan=9) < xx
        through[:, :e0] = False; through[:, e1:] = False
        filled = through.any(1); first = np.argmax(through, 1)
        idx = np.nonzero(filled)[0]
        exit_ok = np.where(np.arange(300)[None, :] > first[idx][:, None], np.nan_to_num(bh[idx], nan=-1), -1).max(1) > xx + k
        p = np.where(exit_ok, k, yy[idx] - xx)
        pnl.append(p); mk.append(np.nonzero(msk)[0][idx])
    return np.concatenate(pnl), np.concatenate(mk)


def E2(x: X) -> dict:
    c = x.c
    res = {"grid": []}
    windows = {"0-240 s": (0, 240), "0-120 s": (0, 120), "120-240 s": (120, 240), "240-290 s": (240, 290)}
    for wn, (e0, e1) in windows.items():
        for xx in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            for k in (0.01, 0.02, 0.03, 0.05, 0.10, 0.20):
                row = {"window": wn, "x": xx, "k": k}
                for name, msk in c.splits():
                    p, m = _dip_sim(x, msk, xx, k, e0, e1)
                    row[name] = {"n": int(len(p)), "mean": float(p.mean()) if len(p) else None,
                                 "exit_rate": float((p == k).mean()) if len(p) else None}
                res["grid"].append(row)
    elig = [r for r in res["grid"] if r["train"]["n"] >= 300]
    best = max(elig, key=lambda r: r["train"]["mean"])
    e0, e1 = windows[best["window"]]
    p, m = _dip_sim(x, c.test, best["x"], best["k"], e0, e1)
    res["selected"] = {"window": best["window"], "x": best["x"], "k": best["k"], "train_mean": best["train"]["mean"], "test": mboot(p, m, start_s=x.start_s)}
    # the case discussed: 0.30 → 0.33 over 0-240 s
    for name, msk in c.splits():
        p, m = _dip_sim(x, msk, 0.30, 0.03, 0, 240)
        res[f"case_30_33_{name}"] = mboot(p, m, start_s=x.start_s)
    # taker version of the same case: buy at the first ask ≤ 0.30 (paid at the ask 1 s later), sell at a bid ≥ 0.33, fees
    for name, msk in c.splits():
        pnl, mk = [], []
        for sd in ("up", "down"):
            S = x.side[sd]
            ask = S["bh"][msk][:, W0:W0 + 301]; bid = S["sl"][msk][:, W0:W0 + 301]; yy = S["y"][msk]
            sig = np.nan_to_num(ask[:, :240], nan=9) <= 0.30
            for i in np.nonzero(sig.any(1))[0]:
                t = np.argmax(sig[i]); a = ask[i, t + 1]
                if not np.isfinite(a):
                    continue
                cost = a + fee(a)
                ex = np.nonzero(np.nan_to_num(bid[i, t + 2:300], nan=-1) >= 0.33)[0]
                if len(ex):
                    b = bid[i, t + 2 + ex[0]]; pnl.append(b - fee(b) - cost)
                else:
                    pnl.append(yy[i] - cost)
                mk.append(np.nonzero(msk)[0][i])
        res[f"case_30_33_taker_{name}"] = mboot(np.array(pnl), np.array(mk), start_s=x.start_s)
    return res


# =========================================================================== E3 pre-open and BTC trend


def E3(x: X) -> dict:
    c = x.c; D = c.D
    res = {}
    pre = D["btc_pre"]; S = D["S"]
    # (a) BTC momentum: P(Up | return of the last L minutes before the open > 0)
    mom = []
    for Lm in (1, 3, 5, 15, 30, 60):
        r = np.log(pre[:, -1] / pre[:, -60 * Lm])
        for name, msk in c.splits():
            up = msk & (r > 0); dn = msk & (r < 0)
            mom.append({"L_min": Lm, "split": name, "p_up_if_up": boot_stat(c.y[up]), "p_up_if_down": boot_stat(c.y[dn]),
                        "corr": nan_corr(r[msk], c.y[msk])})
    res["momentum"] = mom
    # (b) TWAP anchor: at the open, m0 = ln(S_0 / A0) is already known (BTC moved during the reference minute)
    m0 = D["m"][:, 0]; z0 = m0 / (D["sigma"] * np.sqrt(D["V"][0]))
    zb = [-9, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 9]
    anc = []
    for name, msk in c.splits():
        b = np.digitize(z0[msk], zb) - 1
        pm0 = np.nanmean(D["pm"][msk][:, 0:5], 1)
        for kk in range(len(zb) - 1):
            s = b == kk
            if s.sum() < 30:
                continue
            anc.append({"split": name, "z_lo": zb[kk], "z_hi": zb[kk + 1], "n": int(s.sum()), "p_up": float(c.y[msk][s].mean()),
                        "model": float(np.nanmean(D["P"][msk][s, 0])), "market_open": float(np.nanmean(pm0[s]))})
    res["twap_anchor"] = anc
    # (c) strategy: before / at the open, buy the side given by the anchor if its ask is < model − θ
    #     signal at s (−20 … 5 s), execution at the ask observed at s + 1 (taker, fees)
    strat = []
    for s_sig in (-20, -10, -5, 0, 3):
        for th in (0.0, 0.02, 0.05):
            row = {"s": s_sig, "theta": th}
            for name, msk in c.splits():
                ids = np.nonzero(msk)[0]
                # model price at s: for s < 0, approximation with the known partial TWAP
                if s_sig >= 0:
                    Pm = D["P"][ids, s_sig]
                else:
                    known = pre[ids, 3600 - 60:3600 + s_sig]
                    Sx = pre[ids, 3600 + s_sig - 1]
                    A0e = (known.sum(1) + Sx * (-s_sig)) / 60
                    sd = D["sigma"][ids] * np.sqrt(D["V"][0] + (-s_sig) / 3)
                    Pm = ndtr(np.log(Sx / A0e) / sd)
                au = x.side["up"]["ask"][ids, W0 + s_sig]; ad = x.side["down"]["ask"][ids, W0 + s_sig]
                au1 = x.side["up"]["ask"][ids, W0 + s_sig + 1]; ad1 = x.side["down"]["ask"][ids, W0 + s_sig + 1]
                eu = Pm - au - fee(np.nan_to_num(au)); ed = (1 - Pm) - ad - fee(np.nan_to_num(ad))
                go_u = np.nan_to_num(eu, nan=-1) > th; go_d = (np.nan_to_num(ed, nan=-1) > th) & ~go_u
                pnl = np.concatenate([(c.y[ids] - au1 - fee(au1))[go_u & np.isfinite(au1)], ((1 - c.y[ids]) - ad1 - fee(ad1))[go_d & np.isfinite(ad1)]])
                mk = np.concatenate([ids[go_u & np.isfinite(au1)], ids[go_d & np.isfinite(ad1)]])
                row[name] = mboot(pnl, mk, 500, start_s=x.start_s)
                row[name]["coverage"] = float(np.isfinite(au).mean())
            strat.append(row)
    res["anchor_strategy"] = strat
    # (d) executable pure momentum: buy Up if BTC rose over the last L minutes, at the ask at s = 1
    mm = []
    for Lm in (1, 5, 15):
        r = np.log(pre[:, -1] / pre[:, -60 * Lm])
        for name, msk in c.splits():
            ids = np.nonzero(msk)[0]
            side_up = r[ids] > 0
            a = np.where(side_up, x.side["up"]["ask"][ids, W0 + 1], x.side["down"]["ask"][ids, W0 + 1])
            yy = np.where(side_up, c.y[ids], 1 - c.y[ids]); ok = np.isfinite(a)
            mm.append({"L_min": Lm, "split": name, **mboot((yy - a - fee(a))[ok], ids[ok], 500, start_s=x.start_s)})
    res["momentum_strategy"] = mm
    return res


# =========================================================================== E4 end of window (sniping) and pair arbitrage


def E4(x: X) -> dict:
    c = x.c; D = c.D
    res = {}
    # (a) sniping: at s ∈ [270, 297], if the TWAP model gives ≥ q to the favourite and its ask ≤ cap → taker buy (ask at s+1)
    sn = []
    for qmin in (0.95, 0.98, 0.99, 0.995):
        for cap in (0.95, 0.97, 0.99):
            row = {"q": qmin, "cap": cap}
            for name, msk in c.splits():
                pnl, mk = [], []
                for i in np.nonzero(msk)[0]:
                    for s in range(270, 298):
                        Pm = D["P"][i, s]
                        sd = "up" if Pm >= 0.5 else "down"
                        q = Pm if sd == "up" else 1 - Pm
                        a = x.side[sd]["ask"][i, W0 + s]
                        if q >= qmin and np.isfinite(a) and a <= cap:
                            a1 = x.side[sd]["ask"][i, W0 + s + 1]
                            if np.isfinite(a1):
                                pnl.append(x.side[sd]["y"][i] - a1 - fee(a1)); mk.append(i)
                            break
                row[name] = mboot(np.array(pnl), np.array(mk), 500, start_s=x.start_s)
            sn.append(row)
    res["sniping"] = sn
    # (b) pair arbitrage: within one second, a taker Up buy at a_u and a taker Down buy at a_d with a_u + a_d < 1
    au = x.side["up"]["ask"][:, W0:W0 + 300]
    ad_min = 1 - np.fmax.reduce([load()["sell_max"][:, W0:W0 + 300]])  # cheapest Down observed in the second
    au_min = load()["buy_min"][:, W0:W0 + 300]
    s_sum = au_min + ad_min
    arb = {}
    for name, msk in c.splits():
        ss = s_sum[msk]
        ok = np.isfinite(ss)
        gross = 1 - ss
        net = gross - fee(np.nan_to_num(au_min[msk])) - fee(np.nan_to_num(ad_min[msk]))
        arb[name] = {"seconds_with_both": int(ok.sum()), "share_sum_lt_1": float((ss[ok] < 1).mean()),
                     "share_net_positive": float((net[ok] > 0).mean()), "markets_with_net_positive": float(((net > 0) & ok).any(1).mean()),
                     "mean_net_when_positive": float(net[ok][net[ok] > 0].mean()) if (net[ok] > 0).any() else None}
    res["pair_arbitrage"] = arb
    return res




SECTIONS = {"E1": E1, "E2": E2, "E3": E3, "E4": E4}


def main(which):
    x = X()
    for s in which:
        t0 = time.time()
        if s in SECTIONS:
            r = SECTIONS[s](x)
        else:
            from analysis import edges_ml
            r = getattr(edges_ml, s)(x)
        with open(OUT / f"edges_{s}.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1, default=float)
        print(f"{s} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SECTIONS))
