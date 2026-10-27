"""H1-H4: applies the protocol (docs/METHODS.md) and writes results/results_delta<delta>.json.

Usage: python -m analysis.run <delta>   (re-alignment of the market series, in seconds: 0, 2 or 3)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr  # Φ

from analysis import blockboot as bb
from analysis.data import T, T_EXT, Data, load
from pipelines.common.storage import Layout

RNG = np.random.default_rng(20261018)
B = 2000
SPLIT_DAY = os.environ.get("PM_SPLIT", "2026-09-21")  # first day of the test period (YYYY-MM-DD)
FEE_RATE = 0.07
OUT = Path("results")

# Sum_{i,j=0}^{59} min(i, j) / 60²: variance of the future part of the TWAP average, seen from before t = 240.
L_TWAP = 60
C_TWAP = (L_TWAP - 1) * L_TWAP * (2 * L_TWAP - 1) / 6 / L_TWAP**2


def fee(p):
    return FEE_RATE * p * (1 - p)


# --------------------------------------------------------------------------- model


def btc_matrix(d: Data, lo: int, hi: int) -> np.ndarray:
    """Binance close for start + [lo, hi), shape (N, hi − lo)."""
    idx = (d.start_s - d.btc_ts0)[:, None] + np.arange(lo, hi)[None, :]
    return d.btc_close[idx]


def sigma_pre(d: Data) -> np.ndarray:
    """σ per second (log), causal: 5 s returns over [start − 1800, start)."""
    pre = btc_matrix(d, -1800, 1)[:, ::5]
    r = np.diff(np.log(pre), axis=1)
    return np.sqrt((r**2).mean(axis=1) / 5)


def model_inputs(d: Data):
    """Expected value of A1 (log, relative to A0) and variance V(t) in seconds, for t = 0..299."""
    S = btc_matrix(d, 0, T)                                        # close at the end of second t
    A0 = btc_matrix(d, -L_TWAP, 0).mean(axis=1)
    t = np.arange(T)
    n_f = np.clip(T - 1 - t, 0, None)                              # future seconds in [240, 299] after t
    cum = np.cumsum(np.where(t >= T - L_TWAP, S, 0.0), axis=1)     # observed sum of the closes of 240..t
    mean_late = (cum + n_f[None, :] * S) / L_TWAP
    mean = np.where(t[None, :] < T - L_TWAP, S, mean_late)
    V = np.where(t < T - L_TWAP, (T - L_TWAP - t) + C_TWAP, n_f * (n_f + 1) * (2 * n_f + 1) / 6 / L_TWAP**2)
    m = np.log(mean / A0[:, None])
    return m, V, S, A0


def p_model(m, V, sigma, k, tau=0.0):
    sd = np.sqrt((k * sigma[:, None]) ** 2 * V[None, :] + tau**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, m / np.where(sd > 0, sd, 1), np.sign(m) * 50)
    return ndtr(z)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------- stats


def boot_mean(x: np.ndarray, b: int = B) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "p": None}
    bm = x[RNG.integers(0, len(x), size=(b, len(x)))].mean(axis=1)
    return {"n": int(len(x)), "mean": float(x.mean()), "lo": float(np.quantile(bm, 0.025)), "hi": float(np.quantile(bm, 0.975)),
            "p": float((np.sum(bm <= 0) + 1) / (b + 1)), "sum": float(x.sum())}


def irls(X, y, w=None, iters=25):
    beta = np.zeros(X.shape[1])
    w = np.ones(len(y)) if w is None else w
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = w * mu * (1 - mu) + 1e-9
        z = eta + (y - mu) / (mu * (1 - mu) + 1e-9)
        beta_new = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            return beta_new
        beta = beta_new
    return beta


def cluster_boot_logit(X, y, cluster, b=500, rng=None):
    """Cluster bootstrap of a logit: resamples the clusters (markets, days or hours) and refits the regression."""
    rng = RNG if rng is None else rng
    uniq, inv = np.unique(cluster, return_inverse=True)
    beta = irls(X, y)
    draws = []
    for _ in range(b):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float)
        draws.append(irls(X, y, w))
    draws = np.array(draws)
    return beta, draws


def bh(pvals: dict[str, float], q: float = 0.05) -> dict[str, dict]:
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    out, max_ok = {}, 0
    for i, k in enumerate(keys, 1):
        if pvals[k] <= q * i / m:
            max_ok = i
    for i, k in enumerate(keys, 1):
        out[k] = {"p": pvals[k], "rank": i, "threshold": q * i / m, "significant": i <= max_ok,
                  "p_adj": min(1.0, min(pvals[kk] * m / j for j, kk in enumerate(keys, 1) if j >= i))}
    return out


# --------------------------------------------------------------------------- market price


def market_price(d: Data) -> np.ndarray:
    """p_mkt(t): mid (ask_up + 1 − ask_dn)/2 if both are fresh, else p_up_implied."""
    au, ad = d.ask_up[:, :T], d.ask_dn[:, :T]
    mid = (au + (1 - ad)) / 2
    return np.where(np.isnan(mid), d.p_implied, mid)


# --------------------------------------------------------------------------- strategies


def simulate(d: Data, idx: np.ndarray, signal_t: np.ndarray, signal_side: np.ndarray, L: int) -> dict:
    """One position per market: the first signal whose fill at t + L exists.

    signal_t / signal_side: (N, K) time and side (+1 Up, −1 Down, 0 none) of the signals, sorted by t.
    Returns the P&L per share (1{win} − ask − fee) of the markets `idx` that traded.
    """
    pnl, tt, side, price, mk = [], [], [], [], []
    for i in idx:
        for t, s in zip(signal_t[i], signal_side[i]):
            if s == 0 or t + L >= T_EXT:
                continue
            a = d.ask_up[i, t + L] if s > 0 else d.ask_dn[i, t + L]
            if np.isnan(a):
                continue
            win = d.y[i] if s > 0 else not d.y[i]
            pnl.append(float(win) - a - fee(a))
            tt.append(int(t)); side.append(int(s)); price.append(float(a)); mk.append(int(i))
            break
    return {"pnl": np.array(pnl), "t": np.array(tt), "side": np.array(side), "price": np.array(price), "market": np.array(mk)}


def first_signals(cond_up: np.ndarray, cond_dn: np.ndarray, ts: np.ndarray, kmax: int = 40):
    """(N, kmax) matrices of the first signals of each market, from boolean conditions (N, len(ts))."""
    n = cond_up.shape[0]
    st = np.zeros((n, kmax), dtype=int)
    ss = np.zeros((n, kmax), dtype=int)
    side = np.where(cond_up, 1, np.where(cond_dn, -1, 0))
    for i in range(n):
        nz = np.nonzero(side[i])[0][:kmax]
        st[i, : len(nz)] = ts[nz]
        ss[i, : len(nz)] = side[i, nz]
    return st, ss


def strategy_signals(name, param, d, P, pm, S, sigma):
    if name == "S1":
        ts = np.arange(10, 291)
        z = np.log(S[:, ts] / S[:, ts - 3]) / (sigma[:, None] * np.sqrt(3))
        dp = pm[:, ts] - pm[:, ts - 3]
        return first_signals((z >= param) & (dp < 0.02), (z <= -param) & (dp > -0.02), ts)
    if name == "S2":
        ts = np.arange(10, 291, 10)
        au, ad = d.ask_up[:, ts], d.ask_dn[:, ts]
        eu = np.nan_to_num(P[:, ts] - au - fee(au), nan=-np.inf)
        ed = np.nan_to_num((1 - P[:, ts]) - ad - fee(ad), nan=-np.inf)
        return first_signals((eu > param) & (eu >= ed), (ed > param) & (ed > eu), ts)
    if name == "S3":
        ts = np.arange(10, 291)
        dp = pm[:, ts] - pm[:, ts - 5]
        dP = np.abs(P[:, ts] - P[:, ts - 5])
        calm = dP <= 0.03
        return first_signals((dp <= -param) & calm, (dp >= param) & calm, ts)
    if name == "S4":
        t4 = int(param)
        fav_up = pm[:, t4] >= 0.5
        a = np.where(fav_up, d.ask_up[:, t4], d.ask_dn[:, t4])
        ok = (a >= 0.80) & (a <= 0.98)
        st = np.full((d.n, 1), t4)
        ss = np.where(ok, np.where(fav_up, 1, -1), 0)[:, None]
        return st, ss
    if name == "REF":
        return np.full((d.n, 1), 150), np.ones((d.n, 1), dtype=int)
    raise ValueError(name)


GRIDS = {"S1": [2, 3, 4, 5], "S2": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12], "S3": [0.08, 0.12, 0.20],
         "S4": [240, 260, 270, 280, 290], "REF": [0]}


def summarize_trades(r: dict, d: Data | None = None) -> dict:
    """Bootstrap by market; with `d`, also block bootstraps by UTC day and UTC hour (keys "day", "hour")."""
    out = boot_mean(r["pnl"])
    if d is not None and len(r["pnl"]):
        out.update(bb.blocks_mean(r["pnl"], d.start_s[r["market"]]))
    if len(r["pnl"]):
        out.update(hit=float((r["pnl"] > 0).mean()), avg_price=float(r["price"].mean()),
                   avg_fee=float(fee(r["price"]).mean()), share_up=float((r["side"] > 0).mean()),
                   t_median=float(np.median(r["t"])))
    return out


# --------------------------------------------------------------------------- main


def align(d: Data, delta: int) -> Data:
    """Shifts the market series by `delta` seconds: the trade-history timestamp is the Polygon block time,
    ~2.25 s after the execution (measured, docs/data_notes.md). After the shift, index t ≈ execution time."""
    if delta == 0:
        return d

    def sh(a):
        out = np.full_like(a, np.nan, dtype=np.float64)
        out[:, : a.shape[1] - delta] = a[:, delta:]
        return out

    d.p_implied, d.ask_up, d.ask_dn = sh(d.p_implied), sh(d.ask_up), sh(d.ask_dn)
    d.n_trades = np.nan_to_num(sh(d.n_trades.astype(np.float64))).astype(np.int32)
    return d


def main(delta: int = 0) -> None:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d = align(load(Layout("data")), delta)
    train = np.nonzero(d.day < SPLIT_DAY)[0]
    test = np.nonzero(d.day >= SPLIT_DAY)[0]
    y = d.y.astype(float)
    res: dict = {"meta": {"n_markets": d.n, "n_train": len(train), "n_test": len(test), "split_day": SPLIT_DAY,
                          "first_day": d.day[0], "last_day": d.day[-1], "bootstrap": B, "align_delta_s": delta}}

    # ---------- descriptif
    pm = market_price(d)
    spread = d.ask_up[:, :T] + d.ask_dn[:, :T] - 1
    res["descriptive"] = {
        "p_up_base_rate": {"train": float(y[train].mean()), "test": float(y[test].mean())},
        "by_t": {
            "t": list(range(T)),
            "trades_per_s": d.n_trades.mean(axis=0).round(3).tolist(),
            "spread_median": [float(np.nanmedian(spread[:, t])) if np.any(~np.isnan(spread[:, t])) else None for t in range(T)],
            "ask_both_avail": (~np.isnan(spread)).mean(axis=0).round(4).tolist(),
            "abs_p_dev": np.nanmean(np.abs(pm - 0.5), axis=0).round(4).tolist(),
        },
    }

    # ---------- model
    sigma = sigma_pre(d)
    m, V, S, A0 = model_inputs(d)
    snaps = np.arange(10, 291, 10)
    ks = np.round(np.arange(0.6, 2.01, 0.05), 2)
    taus = [0.0, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1.2e-4, 1.6e-4, 2.4e-4]
    ytr = np.repeat(y[train][:, None], len(snaps), 1)
    ll = {(k, tau): logloss(p_model(m[train][:, snaps], V[snaps], sigma[train], k, tau), ytr) for k in ks for tau in taus}
    k_star, tau_star = min(ll, key=ll.get)
    k_star, tau_star = float(k_star), float(tau_star)
    P = p_model(m, V, sigma, k_star, tau_star)
    a0_vs_chainlink = A0 - d.price_start
    agree_end = ((m[:, T - 1] >= 0) == d.y)
    res["model"] = {
        "k_star": k_star, "tau_star": tau_star, "tau_star_usd": float(tau_star * np.median(A0)),
        "fit_grid": [{"k": float(k), "tau": float(t), "logloss": float(v)} for (k, t), v in ll.items()],
        "sigma_per_s_median": float(np.median(sigma)), "sigma_5min_median_usd": float(np.median(sigma * np.sqrt(300) * A0)),
        "a0_minus_chainlink_usd": {"mean": float(np.nanmean(a0_vs_chainlink)), "median": float(np.nanmedian(a0_vs_chainlink))},
        "outcome_agreement_t299": {"train": float(agree_end[train].mean()), "test": float(agree_end[test].mean())},
    }

    # ---------- H2: calibration and predictive power
    def calib(p, yy, bins=10):
        e = np.linspace(0, 1, bins + 1)
        b = np.clip(np.digitize(p, e) - 1, 0, bins - 1)
        return [{"bin": int(k), "p_mean": float(p[b == k].mean()), "y_mean": float(yy[b == k].mean()), "n": int((b == k).sum())}
                for k in range(bins) if (b == k).sum() >= 30]

    groups = {"early (10-90 s)": np.arange(10, 91, 10), "middle (100-230 s)": np.arange(100, 231, 10),
              "late (240-290 s)": np.arange(240, 291, 10)}
    h2 = {"scores": {}, "calibration": {}, "logit_by_t": []}
    for split, idx in (("train", train), ("test", test)):
        for g, ts in groups.items():
            pmk, pmo = pm[idx][:, ts].ravel(), P[idx][:, ts].ravel()
            yy = np.repeat(y[idx][:, None], len(ts), 1).ravel()
            ok = ~np.isnan(pmk)
            h2["scores"][f"{split}|{g}"] = {
                "brier_market": float(np.mean((pmk[ok] - yy[ok]) ** 2)), "brier_model": float(np.mean((pmo[ok] - yy[ok]) ** 2)),
                "logloss_market": float(logloss(pmk[ok], yy[ok])), "logloss_model": float(logloss(pmo[ok], yy[ok])), "n": int(ok.sum())}
            if split == "test":
                h2["calibration"][g] = {"market": calib(pmk[ok], yy[ok]), "model": calib(pmo[ok], yy[ok])}

    t2 = np.arange(30, 271, 30)

    def h2_design(idx, ts):
        lm, lp = logit(pm[idx][:, ts]), logit(P[idx][:, ts])
        X = np.column_stack([np.ones(lm.size), lm.ravel(), (lp - lm).ravel()])
        yy = np.repeat(y[idx][:, None], len(ts), 1).ravel()
        cl = np.repeat(idx[:, None], len(ts), 1).ravel()
        ok = np.all(np.isfinite(X), axis=1)
        return X[ok], yy[ok], cl[ok]

    for split, idx in (("train", train), ("test", test)):
        X, yy, cl = h2_design(idx, t2)
        beta, draws = cluster_boot_logit(X, yy, cl, b=500)
        h2[f"pooled_{split}"] = {"a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]),
                                 "c_lo": float(np.quantile(draws[:, 2], 0.025)), "c_hi": float(np.quantile(draws[:, 2], 0.975)),
                                 "b_lo": float(np.quantile(draws[:, 1], 0.025)), "b_hi": float(np.quantile(draws[:, 1], 0.975)),
                                 "p_c": float((np.sum(draws[:, 2] <= 0) + 1) / (len(draws) + 1)), "n_obs": int(len(yy))}
        if split == "test":
            for unit in bb.UNITS:
                _, dr = cluster_boot_logit(X, yy, bb.block_keys(d.start_s[cl], unit), b=500, rng=bb.RNG)
                h2["pooled_test"][unit] = {"c_lo": float(np.quantile(dr[:, 2], 0.025)), "c_hi": float(np.quantile(dr[:, 2], 0.975)),
                                           "c_se": float(dr[:, 2].std(ddof=1)), "n_groups": int(len(np.unique(bb.block_keys(d.start_s[cl], unit)))),
                                           "p_c": float((np.sum(dr[:, 2] <= 0) + 1) / (len(dr) + 1))}
    for t in t2:
        for split, idx in (("train", train), ("test", test)):
            X, yy, _ = h2_design(idx, np.array([t]))
            beta = irls(X, yy)
            h2["logit_by_t"].append({"t": int(t), "split": split, "b": float(beta[1]), "c": float(beta[2])})
    res["H2"] = h2

    # ---------- H1: lead-lag (response of Δp_mkt to the changes ΔP of the model)
    lags = list(range(-3, 11))
    ts = np.arange(15, 286)
    dpm = np.diff(pm, axis=1)            # dpm[:, t-1] = pm[t] - pm[t-1]
    dP = np.diff(P, axis=1)
    h1 = {"lags": lags}
    for split, idx in (("train", train), ("test", test)):
        Y = dpm[idx][:, ts - 1]
        Xs = [dP[idx][:, ts - 1 - k] for k in lags]
        ok = np.isfinite(Y) & np.all([np.isfinite(x) for x in Xs], axis=0)
        # Sufficient statistics per market -> fast cluster bootstrap.
        XtX = np.zeros((len(idx), len(lags), len(lags)))
        Xty = np.zeros((len(idx), len(lags)))
        for j in range(len(idx)):
            o = ok[j]
            Xm = np.column_stack([x[j][o] for x in Xs])
            XtX[j] = Xm.T @ Xm
            Xty[j] = Xm.T @ Y[j][o]
        beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
        draws = []
        for _ in range(500):
            w = np.bincount(RNG.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
            draws.append(np.linalg.solve(np.tensordot(w, XtX, 1), w @ Xty))
        draws = np.array(draws)
        pos = [i for i, k in enumerate(lags) if k >= 0]
        cum = np.cumsum(beta[pos]); cumd = np.cumsum(draws[:, pos], axis=1)
        after_L = [i for i, k in enumerate(lags) if k >= 2]
        h1[split] = {"beta": beta.tolist(), "lo": np.quantile(draws, 0.025, 0).tolist(), "hi": np.quantile(draws, 0.975, 0).tolist(),
                     "cum": cum.tolist(), "cum_lo": np.quantile(cumd, 0.025, 0).tolist(), "cum_hi": np.quantile(cumd, 0.975, 0).tolist(),
                     "share_after_2s": float(beta[after_L].sum() / beta[pos].sum()),
                     "sum_0_10": float(beta[pos].sum())}
        if split == "test":
            # Block bootstraps: the per-market sufficient statistics are summed by UTC day (or hour), then the blocks
            # are resampled.
            for unit in bb.UNITS:
                _, g = np.unique(bb.block_keys(d.start_s[idx], unit), return_inverse=True)
                ng = int(g.max()) + 1
                XtX_g = np.zeros((ng, len(lags), len(lags))); Xty_g = np.zeros((ng, len(lags)))
                np.add.at(XtX_g, g, XtX); np.add.at(Xty_g, g, Xty)
                w = bb.draw_counts(ng, 500)
                dr = np.array([np.linalg.solve(np.tensordot(wk, XtX_g, 1), wk @ Xty_g) for wk in w])
                h1[split][unit] = {**bb.ci(dr), "n_groups": ng}
    res["H1"] = h1

    # ---------- H4: calibration by fine bin at the end of the window (market vs model)
    late = []
    for t in (240, 255, 270, 280, 290, 295):
        for split, idx in (("train", train), ("test", test)):
            pmk, pmo, yy = pm[idx, t], P[idx, t], y[idx]
            ok = ~np.isnan(pmk)
            fav = np.where(pmk >= 0.5, pmk, 1 - pmk)
            favwin = np.where(pmk >= 0.5, yy, 1 - yy)
            sel = ok & (fav >= 0.8) & (fav <= 0.98)
            late.append({"t": t, "split": split, "brier_market": float(np.mean((pmk[ok] - yy[ok]) ** 2)),
                         "brier_model": float(np.mean((pmo[ok] - yy[ok]) ** 2)),
                         "fav_price_mean": float(fav[sel].mean()) if sel.any() else None,
                         "fav_win_rate": float(favwin[sel].mean()) if sel.any() else None, "n_fav": int(sel.sum())})
    res["H4"] = {"late_calibration": late}

    # ---------- strategies: selection on train, evaluation on test
    strats = {}
    for name, grid in GRIDS.items():
        rows = []
        for param in grid:
            st, ss = strategy_signals(name, param, d, P, pm, S, sigma)
            r = simulate(d, train, st, ss, L=2)
            rows.append({"param": param, **summarize_trades(r)})
        eligible = [r for r in rows if r["n"] >= 200 and r["mean"] is not None] or [r for r in rows if r["mean"] is not None]
        best = max(eligible, key=lambda r: r["mean"]) if eligible else None
        entry = {"train_grid": rows, "selected": best["param"] if best else None}
        if best is not None:
            st, ss = strategy_signals(name, best["param"], d, P, pm, S, sigma)
            rt = simulate(d, test, st, ss, L=2)
            entry["test"] = summarize_trades(rt, d)
            entry["latency"] = {str(L): summarize_trades(simulate(d, test, st, ss, L=L), d) for L in (0, 1, 2, 3, 5)}
            # Cumulative P&L on the test period (markets in chronological order).
            order = np.argsort(d.start_s[rt["market"]]) if len(rt["market"]) else []
            entry["test_curve"] = {"day": d.day[rt["market"][order]].tolist() if len(order) else [],
                                   "cum_pnl": np.cumsum(rt["pnl"][order]).round(4).tolist() if len(order) else []}
            # Test period cut into two chronological halves.
            tdays = np.unique(d.day[test])
            mid = tdays[len(tdays) // 2]
            fmt = lambda s: f"{s[8:10]}/{s[5:7]}"
            halves = (f"S1 ({fmt(tdays[0])}-{fmt(tdays[len(tdays) // 2 - 1])})", f"S2 ({fmt(mid)}-{fmt(tdays[-1])})")
            weeks = {}
            for mi, p in zip(rt["market"], rt["pnl"]):
                w = halves[0] if d.day[mi] < mid else halves[1]
                weeks.setdefault(w, []).append(p)
            entry["test_by_week"] = {w: boot_mean(np.array(v)) for w, v in sorted(weeks.items())}
            # Raw edge by buy-price bin (diagnostic).
            if len(rt["pnl"]):
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                bi = np.digitize(rt["price"], bins) - 1
                entry["test_by_price"] = [{"lo": bins[k], "hi": bins[k + 1], **boot_mean(rt["pnl"][bi == k], 500)}
                                          for k in range(5) if (bi == k).sum() >= 20]
        strats[name] = entry
    res["strategies"] = strats

    # ---------- multiple testing (family P)
    pv = {"H2_c": res["H2"]["pooled_test"]["p_c"]}
    for name in ("S1", "S2", "S3", "S4"):
        tst = strats[name].get("test")
        pv[name] = tst["p"] if tst and tst["p"] is not None else 1.0
    res["bh"] = bh(pv)
    # Same family with the block-bootstrap p-values (by UTC day: reference for the verdicts; by UTC hour: sensitivity).
    for unit in bb.UNITS:
        pvu = {"H2_c": res["H2"]["pooled_test"][unit]["p_c"]}
        for name in ("S1", "S2", "S3", "S4"):
            tst = strats[name].get("test") or {}
            pvu[name] = tst[unit]["p"] if unit in tst and tst[unit]["p"] is not None else 1.0
        res[f"bh_{unit}"] = bh(pvu)
    res["meta"]["runtime_s"] = round(time.time() - t_start, 1)

    with open(OUT / f"results_delta{delta}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps({"k": k_star, "tau": tau_star, "H2_test": res["H2"]["pooled_test"], "bh": res["bh"],
                      **{n: {"sel": s["selected"], "test": {k: s.get("test", {}).get(k) for k in ("n", "mean", "lo", "hi", "p")}}
                         for n, s in strats.items()}}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
