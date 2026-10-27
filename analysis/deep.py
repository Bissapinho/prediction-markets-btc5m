"""Analyse approfondie des prix à la seconde (results/deep_*.json).

Usage : python -m analysis.deep A|B|C|D
  A : décorrélation BTC (résidu, dynamique, persistance, effets comportementaux à l'ouverture)
  B : volatilité et rendements (BTC, vol implicite, martingale / excès de vol, Hurst)
  D : ML / DL (outcome, Δp à 10 s, vol BTC)

Convention : séries Prediction Market recalées de 2 s. Exploration < 2026-09-21 ≤ test. IC par bootstrap sur les marchés.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from analysis import blockboot as bb
from analysis.deep_data import load_cache

OUT = Path("results")
SPLIT = os.environ.get("PM_SPLIT", "2026-09-21")  # début du test (AAAA-MM-JJ)
RNG = np.random.default_rng(7)
T = 300


def logit(p, eps=0.01):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def nan_corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    o = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[o], b[o])[0, 1]) if o.sum() > 10 else None


def boot_stat(per_market: np.ndarray, fn=np.nanmean, b: int = 1000, start_s=None) -> dict:
    """Statistic of per-market values + bootstrap CI (markets resampled).

    With start_s (market start of each value) and fn = mean, also adds block bootstraps by UTC day and UTC hour.
    """
    x = np.asarray(per_market, dtype=float)
    fin = np.isfinite(x)
    x = x[fin]
    if len(x) < 5:
        return {"est": None, "lo": None, "hi": None, "n": int(len(x))}
    bs = np.array([fn(x[RNG.integers(0, len(x), len(x))]) for _ in range(b)])
    out = {"est": float(fn(x)), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)), "n": int(len(x))}
    if start_s is not None:
        out.update(bb.blocks_mean(x, np.asarray(start_s)[fin]))
    return out


def boot_ratio(num: np.ndarray, den: np.ndarray, b: int = 1000) -> dict:
    o = np.isfinite(num) & np.isfinite(den)
    num, den = num[o], den[o]
    idx = RNG.integers(0, len(num), (b, len(num)))
    bs = num[idx].sum(1) / den[idx].sum(1)
    return {"est": float(num.sum() / den.sum()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)), "n": int(len(num))}


def ols(X, y):
    o = np.all(np.isfinite(X), 1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[o], y[o], rcond=None)
    r = y[o] - X[o] @ beta
    return beta, 1 - r.var() / y[o].var(), int(o.sum())


def cluster_ols(X, y, cluster, b=300):
    """OLS + IC bootstrap par grappes (statistiques suffisantes par marché)."""
    o = np.all(np.isfinite(X), 1) & np.isfinite(y)
    X, y, cl = X[o], y[o], cluster[o]
    uniq, inv = np.unique(cl, return_inverse=True)
    k = X.shape[1]
    XtX = np.zeros((len(uniq), k, k)); Xty = np.zeros((len(uniq), k))
    np.add.at(XtX, inv, X[:, :, None] * X[:, None, :])
    np.add.at(Xty, inv, X * y[:, None])
    beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
    draws = []
    for _ in range(b):
        w = np.bincount(RNG.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        draws.append(np.linalg.solve(np.tensordot(w, XtX, 1), w @ Xty))
    draws = np.array(draws)
    r = y - X @ beta
    return {"beta": beta.tolist(), "lo": np.quantile(draws, .025, 0).tolist(), "hi": np.quantile(draws, .975, 0).tolist(),
            "r2": float(1 - r.var() / y.var()), "n": int(len(y)), "n_markets": int(len(uniq))}


class Ctx:
    def __init__(self):
        t0 = time.time()
        self.D = D = load_cache()
        self.n = len(D["y"])
        # Marchés valides : σ pré-marché non dégénéré (un flux Binance figé casse le modèle ; 0,1 % des marchés).
        self.valid = D["sigma"] >= 4e-6
        self.train = (D["day"] < SPLIT) & self.valid
        self.test = (D["day"] >= SPLIT) & self.valid
        self.y = D["y"].astype(float)
        self.x = logit(D["pm"])                  # logit prix de marché
        self.z = logit(D["P"])                   # logit valeur théorique (Binance)
        ok = np.isfinite(self.x) & np.isfinite(self.z) & self.train[:, None]
        ok[:, :10] = False
        self.map = np.polyfit(self.z[ok], self.x[ok], 1)
        self.e = self.x - np.polyval(self.map, self.z)  # résidu non expliqué par BTC
        self.idx = np.arange(self.n)
        print(f"ctx {time.time() - t0:.1f}s", flush=True)

    def splits(self):
        return (("train", self.train), ("test", self.test))


# =========================================================================== A


def section_A(c: Ctx) -> dict:
    D, x, z, e = c.D, c.x, c.z, c.e
    res = {"map_slope": float(c.map[0]), "map_intercept": float(c.map[1])}

    # A1 : part du niveau expliquée par BTC, par tranche de 30 s
    lvl = []
    for s in range(0, T, 30):
        row = {"t0": s}
        for name, msk in c.splits():
            xs, zs = x[msk, s:s + 30], z[msk, s:s + 30]
            r = nan_corr(xs, zs)
            row[name] = {"r2": None if r is None else r * r, "resid_sd": float(np.nanstd(e[msk, s:s + 30]))}
        lvl.append(row)
    res["level_r2_by_t"] = lvl

    # A2 : part des variations expliquée par BTC selon l'horizon (signature de décorrélation)
    hz = []
    for h in (1, 2, 5, 10, 20, 30, 60):
        row = {"h": h}
        for name, msk in c.splits():
            dx = x[msk, 10 + h:290] - x[msk, 10:290 - h]
            # BTC : variation théorique contemporaine + 3 s de décalage résiduel (H1 : ~90 % intégré en 2 s)
            dz0 = z[msk, 10 + h:290] - z[msk, 10:290 - h]
            dz1 = z[msk, 9 + h:289] - z[msk, 9:289 - h]
            dz2 = z[msk, 8 + h:288] - z[msk, 8:288 - h]
            X = np.column_stack([dz0.ravel(), dz1.ravel(), dz2.ravel()])
            _, r2, nobs = ols(X, dx.ravel())
            row[name] = {"r2": float(r2), "n": nobs}
        hz.append(row)
    res["change_r2_by_horizon"] = hz

    # A3 : dynamique du résidu (ACF, demi-vie, queues)
    lags = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]
    acf = {}
    for name, msk in c.splits():
        eo = e[msk][:, 10:290]
        acf[name] = [nan_corr(eo[:, :-L], eo[:, L:]) for L in lags]
        flat = eo[np.isfinite(eo)]
        dev = flat - flat.mean()
        res.setdefault("resid_dist", {})[name] = {
            "sd": float(flat.std()), "kurtosis": float((dev**4).mean() / dev.var() ** 2),
            "q": {str(q): float(np.quantile(flat, q)) for q in (.01, .05, .25, .5, .75, .95, .99)},
            "share_abs_gt_1": float((np.abs(flat) > 1).mean())}
    # ajustement acf(h) = w · exp(−h / τ) sur h ≥ 2 (w < 1 = part de bruit blanc)
    for name in acf:
        a = np.array(acf[name]); L = np.array(lags)
        o = (L >= 2) & (a > 0)
        sl, ic = np.polyfit(L[o], np.log(a[o]), 1)
        acf[name] = {"lags": lags, "acf": acf[name], "tau_s": float(-1 / sl), "half_life_s": float(np.log(2) * -1 / sl),
                     "w": float(np.exp(ic))}
    res["resid_acf"] = acf

    # A4 : retour à la moyenne du résidu : Δe sur 10 s contre e (par tranche de |e|)
    mr = {}
    for name, msk in c.splits():
        e0 = e[msk][:, 20:280:10].ravel(); e1 = e[msk][:, 30:290:10].ravel()
        cl = np.repeat(c.idx[msk], 26)
        X = np.column_stack([np.ones_like(e0), e0])
        mr[name] = cluster_ols(X, e1 - e0, cl)
    res["resid_mean_reversion_10s"] = mr

    # A5 : le résidu anticipe-t-il BTC ? corr(Δe sur 10 s, rendement BTC futur) avant la phase TWAP (t < 230),
    # avec le rendement log pur (m) pour éviter la dérive mécanique de z ; placebo = rendement passé.
    m = D["m"]
    lead = {}
    for name, msk in c.splits():
        de = e[msk][:, 20:200] - e[msk][:, 10:190]
        row = {}
        for k in (2, 5, 10, 20, 30):
            fut = m[msk][:, 20 + k:200 + k] - m[msk][:, 20:200]
            per = np.array([nan_corr(de[i], fut[i]) if np.isfinite(de[i]).sum() > 20 else np.nan for i in range(de.shape[0])])
            row[str(k)] = {"pooled": nan_corr(de, fut), "mean_per_market": boot_stat(per, start_s=D["start_s"][msk])}
        row["placebo_past_10"] = nan_corr(de, m[msk][:, 20:200] - m[msk][:, 10:190])
        lead[name] = row
    res["resid_leads_btc"] = lead

    # A6 : persistance d'un marché à l'autre et effets comportementaux à l'ouverture
    st = D["start_s"]
    consec = np.r_[False, np.diff(st) == 300]
    prev_y = np.r_[np.nan, c.y[:-1]]
    prev_ret = np.r_[np.nan, np.log(D["price_end"][:-1] / D["price_start"][:-1])]
    e_end = np.nanmean(e[:, 240:290], axis=1)
    e_open = np.nanmean(e[:, 10:40], axis=1)
    p_open_gap = np.nanmean(D["pm"][:, 10:40] - D["P"][:, 10:40], axis=1)
    beh = {}
    for name, msk in c.splits():
        o = msk & consec
        prev_e = np.r_[np.nan, e_end[:-1]]
        beh[name] = {
            "corr_prev_end_resid_vs_open_resid": nan_corr(prev_e[o], e_open[o]),
            "open_gap_after_up": boot_stat(p_open_gap[o & (prev_y == 1)]),
            "open_gap_after_down": boot_stat(p_open_gap[o & (prev_y == 0)]),
            "corr_prev_ret_vs_open_gap": nan_corr(prev_ret[o], p_open_gap[o]),
        }
        # séries : 3 outcomes identiques d'affilée
        streak = np.zeros(c.n)
        for i in range(3, c.n):
            if consec[i] and consec[i - 1] and consec[i - 2] and c.y[i - 1] == c.y[i - 2] == c.y[i - 3]:
                streak[i] = 1 if c.y[i - 1] == 1 else -1
        beh[name]["open_gap_after_3_up"] = boot_stat(p_open_gap[o & (streak == 1)])
        beh[name]["open_gap_after_3_down"] = boot_stat(p_open_gap[o & (streak == -1)])
        beh[name]["p_up_after_3_up"] = boot_stat(c.y[o & (streak == 1)])
        beh[name]["p_up_after_3_down"] = boot_stat(c.y[o & (streak == -1)])
    res["behavior_open"] = beh

    # A7 : biais directionnel (le marché surpaie-t-il Up ?) et effet heure
    hour = ((st % 86400) // 3600).astype(int)
    gap = np.nanmean(D["pm"][:, 10:290] - D["P"][:, 10:290], axis=1)
    res["up_bias"] = {name: boot_stat(gap[msk]) for name, msk in c.splits()}
    res["resid_abs_by_hour"] = [{"h": h, "mean_abs_resid": float(np.nanmean(np.abs(e[hour == h][:, 10:290]))),
                                 "gap": float(np.nanmean(gap[hour == h]))} for h in range(24)]
    # |résidu| selon le régime de volatilité et le volume
    q = np.quantile(D["sigma"], [.2, .4, .6, .8])
    vb = np.digitize(D["sigma"], q)
    vol_usd = D["usd"].sum(1)
    qv = np.quantile(vol_usd, [.2, .4, .6, .8]); vv = np.digitize(vol_usd, qv)
    res["resid_abs_by_regime"] = {
        "sigma_quintile": [float(np.nanmean(np.abs(e[vb == k][:, 10:290]))) for k in range(5)],
        "volume_quintile": [float(np.nanmean(np.abs(e[vv == k][:, 10:290]))) for k in range(5)]}
    return res


# =========================================================================== B


def section_B(c: Ctx) -> dict:
    D = c.D
    res = {}
    S = D["S"]; pre = D["btc_pre"]          # pre : 3600 s avant l'ouverture
    full = np.concatenate([pre, S], axis=1)
    r1 = np.diff(np.log(full), axis=1)      # rendements 1 s, (N, 3899)

    # B1 : loi d'échelle (signature) : Var(r_h) / h selon h, et exposant de Hurst
    win = np.log(S)
    sig = []
    for h in (1, 2, 3, 5, 10, 15, 30, 60, 120, 300):
        rh = np.log(full[:, h::h]) - np.log(full[:, :-h:h][:, : full[:, h::h].shape[1]])
        sig.append({"h": h, "var_per_s": float(np.nanmean(rh**2) / h)})
    hs = np.array([s["h"] for s in sig]); vs = np.array([s["var_per_s"] * s["h"] for s in sig])
    o = hs >= 10
    slope = np.polyfit(np.log(hs[o]), np.log(vs[o]), 1)[0]
    res["btc_signature"] = {"points": sig, "hurst_10_300s": float(slope / 2)}

    # B2 : distribution des rendements, normalisés par σ pré-marché (marchés valides)
    v = c.valid
    r = r1[:, -T:]
    r5s = np.diff(np.log(S[:, ::5]), axis=1)          # rendements 5 s dans la fenêtre
    z1 = (r5s[v] / (D["sigma"][v, None] * np.sqrt(5))).ravel()
    z1 = z1[np.isfinite(z1)]
    r5 = (np.log(S[:, -1] / S[:, 0]) / (D["sigma"] * np.sqrt(T - 1)))[v]
    def mom(v):
        v = v[np.isfinite(v)]; d = v - v.mean()
        return {"sd": float(v.std()), "skew": float((d**3).mean() / v.std()**3), "kurtosis": float((d**4).mean() / v.var()**2),
                "p_abs_gt_3": float((np.abs(d / v.std()) > 3).mean()), "share_zero": float((v == 0).mean())}
    res["btc_returns"] = {"r5s_normalized": mom(z1), "r5m_normalized": mom(r5),
                          "share_zero_1s": float((r1[v] == 0).mean())}
    # PIT de l'outcome sous le modèle à t = 0 : la variable z0 = m_final / sd0 est-elle N(0, 1) ?
    m_end = D["m"][:, -1]
    sd0 = D["sigma"] * np.sqrt(D["V"][0])
    zz = (m_end / sd0)[v]
    res["btc_returns"]["twap_move_over_model_sd"] = mom(zz)
    res["btc_returns"]["twap_move_quantiles"] = {str(q): float(np.nanquantile(zz, q)) for q in (.01, .05, .25, .5, .75, .95, .99)}

    # B3 : saisonnalité et persistance de la vol (vol réalisée 5 min de la fenêtre)
    rv = np.sqrt(np.nansum(r5s**2, axis=1))            # vol réalisée de la fenêtre (rendements 5 s)
    st = D["start_s"]; hour = ((st % 86400) // 3600).astype(int)
    dow = ((st // 86400) + 4) % 7  # 0 = lundi
    res["btc_vol_season"] = {
        "by_hour_utc": [float(np.nanmedian(rv[hour == h]) * D["A0"][hour == h].mean()) for h in range(24)],
        "by_dow": [float(np.nanmedian(rv[dow == k]) * np.nanmean(D["A0"])) for k in range(7)]}
    consec = np.r_[False, np.diff(st) == 300]
    lrv = np.log(rv)
    res["btc_vol_persistence"] = {str(L): nan_corr(lrv[:-L][consec[L:]], lrv[L:][consec[L:]]) for L in (1, 2, 6, 12, 288)}
    # σ pré-marché (30 min) contre vol réalisée de la fenêtre
    res["btc_vol_pre_vs_realized"] = {"corr_log": nan_corr(np.log(D["sigma"][v]), lrv[v]),
                                     "ratio_median": float(np.nanmedian(rv[v] / (D["sigma"][v] * np.sqrt(T - 5))))}

    # B4 : volatilité implicite du marché (inversion du modèle TWAP) et rapport implicite / réalisée
    tau = 4e-5
    P = D["pm"]; m = D["m"]; V = D["V"]
    iv_rows = []
    for t in range(15, 286, 15):
        q = ndtri(np.clip(P[:, t], 1e-4, 1 - 1e-4))
        good = (np.abs(q) > 0.15) & (np.sign(q) == np.sign(m[:, t])) & (np.abs(m[:, t]) > 0) & np.isfinite(q)
        s2 = ((m[:, t] / np.where(good, q, 1)) ** 2 - tau**2) / V[t]
        iv = np.where(good & (s2 > 0), np.sqrt(np.abs(s2)), np.nan)
        # vol réalisée ex post sur [t, 300), par seconde, en rendements 5 s
        rv_after = np.sqrt(np.nanmean(r5s[:, t // 5 + 1:] ** 2, axis=1) / 5)
        row = {"t": t}
        for name, msk in c.splits():
            g = msk & np.isfinite(iv)
            row[name] = {"n": int(g.sum()), "iv_over_pre": float(np.nanmedian(iv[g] / D["sigma"][g])),
                         "iv_over_realized_after": float(np.nanmedian(iv[g] / rv_after[g])),
                         "pre_over_realized_after": float(np.nanmedian(D["sigma"][g] / rv_after[g]))}
        iv_rows.append(row)
    res["implied_vol_term"] = iv_rows
    # « smile » : iv / pre selon la distance au strike (|Φ⁻¹(p)|), à t = 120
    t = 120
    q = ndtri(np.clip(P[:, t], 1e-4, 1 - 1e-4))
    good = (np.abs(q) > 0.15) & (np.sign(q) == np.sign(m[:, t]))
    iv = np.where(good, np.sqrt(np.clip(((m[:, t] / np.where(good, q, 1)) ** 2 - tau**2) / V[t], 0, None)), np.nan)
    edges = [0.15, 0.4, 0.7, 1.0, 1.4, 2.0, 3.0]
    b = np.digitize(np.abs(q), edges) - 1
    res["implied_vol_smile_t120"] = [{"lo": edges[k], "hi": edges[k + 1], "n": int(((b == k) & np.isfinite(iv)).sum()),
                                      "iv_over_pre": float(np.nanmedian((iv / D["sigma"])[b == k]))} for k in range(len(edges) - 1)]
    # la vol implicite prévoit-elle la vol réalisée mieux que la vol historique ?
    t = 60
    q = ndtri(np.clip(P[:, t], 1e-4, 1 - 1e-4))
    good = (np.abs(q) > 0.3) & (np.sign(q) == np.sign(m[:, t]))
    iv = np.where(good, np.sqrt(np.clip(((m[:, t] / np.where(good, q, 1)) ** 2 - tau**2) / V[t], 1e-14, None)), np.nan)
    rv_after = np.sqrt(np.nanmean(r5s[:, t // 5 + 1:] ** 2, axis=1) / 5)
    enc = {}
    for name, msk in c.splits():
        g = msk & np.isfinite(iv) & (rv_after > 0)
        X = np.column_stack([np.ones(g.sum()), np.log(D["sigma"][g]), np.log(iv[g])])
        beta, r2, nobs = ols(X, np.log(rv_after[g]))
        _, r2_pre, _ = ols(X[:, :2], np.log(rv_after[g]))
        enc[name] = {"beta_pre": float(beta[1]), "beta_implied": float(beta[2]), "r2_both": float(r2), "r2_pre_only": float(r2_pre), "n": nobs}
    res["implied_vs_historical_forecast_t60"] = enc

    # B5 : propriétés de martingale du prix de marché
    #  (i) excès de volatilité : E[Σ(Δp)²] (jusqu'à l'outcome) / E[p0(1 − p0)] = 1 pour une martingale
    pm = D["pm"]
    ev = []
    for step in (1, 2, 5, 10, 30, 60):
        row = {"step_s": step}
        for name, msk in c.splits():
            pth = pm[msk][:, 10:299:step]
            path = np.column_stack([pth, c.y[msk]])
            path = np.where(np.isfinite(path), path, np.nan)
            # forward-fill des rares NaN
            for j in range(1, path.shape[1]):
                path[:, j] = np.where(np.isfinite(path[:, j]), path[:, j], path[:, j - 1])
            qv = np.nansum(np.diff(path, axis=1) ** 2, axis=1)
            p0 = path[:, 0]
            row[name] = boot_ratio(qv, p0 * (1 - p0))
        ev.append(row)
    res["excess_vol_ratio"] = ev
    #  (ii) ratio de variance des Δp (q secondes contre 1 s)
    vr = []
    for qn in (2, 5, 10, 30, 60):
        row = {"q": qn}
        for name, msk in c.splits():
            p = pm[msk][:, 10:290]
            d1 = np.diff(p, axis=1); dq = p[:, qn:] - p[:, :-qn]
            row[name] = float(np.nanvar(dq) / (qn * np.nanvar(d1)))
        vr.append(row)
    res["variance_ratio"] = vr
    #  (iii) autocorrélation des Δp à 10 s, avec et sans contrôle BTC
    ac = {}
    for name, msk in c.splits():
        p = pm[msk]; zz = c.z[msk]
        a = (p[:, 20:280:10] - p[:, 10:270:10]).ravel(); b = (p[:, 30:290:10] - p[:, 20:280:10]).ravel()
        dzb = (zz[:, 30:290:10] - zz[:, 20:280:10]).ravel()
        cl = np.repeat(c.idx[msk], 26)
        ac[name] = {"corr": nan_corr(a, b),
                    "ols_with_btc": cluster_ols(np.column_stack([np.ones_like(a), a, dzb]), b, cl)}
    res["dp_autocorr_10s"] = ac
    # (iv) vol du prix de marché contre vol BTC : élasticité
    sdx = np.nanstd(np.diff(c.x[:, 30:270:5], axis=1), axis=1)
    res["market_vol_elasticity"] = {name: ols(np.column_stack([np.ones(msk.sum()), np.log(D["sigma"][msk])]),
                                               np.log(sdx[msk]))[0][1].item() for name, msk in c.splits()}
    return res


# =========================================================================== main


SECTIONS = {"A": section_A, "B": section_B}


def main(which: str) -> None:
    c = Ctx()
    for s in which:
        t0 = time.time()
        if s not in SECTIONS:
            from analysis import deep_cd  # sections C et D
            fn = getattr(deep_cd, f"section_{s}")
        else:
            fn = SECTIONS[s]
        r = fn(c)
        with open(OUT / f"deep_{s}.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1, default=float)
        print(f"section {s} done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AB")
