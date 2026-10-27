"""Section D (ML / DL) de l'analyse approfondie."""

from __future__ import annotations

import time

import numpy as np

from analysis import blockboot as bb
from analysis.deep import RNG, T, Ctx, boot_stat, cluster_ols, logit, nan_corr
from analysis.run import irls

SNAPS = np.arange(30, 271, 30)


def _features(c: Ctx, ts=SNAPS):
    """Matrice de features (une ligne par marché × instant) ; uniquement de l'information connue à t."""
    D = c.D
    n = c.n
    x, z, e = c.x, c.z, c.e
    S = D["S"]; sig = D["sigma"]; flow = D["flow_usd"] / 1000; usd = D["usd"] / 1000; ntr = D["n_trades"]
    spread = D["ask_up"] + D["ask_dn"] - 1
    st = D["start_s"]; hour = (st % 86400) / 3600; dow = ((st // 86400) + 4) % 7
    consec = np.r_[False, np.diff(st) == 300]
    prev_y = np.where(consec, np.r_[0.5, c.y[:-1]], 0.5)
    prev_ret = np.where(consec, np.r_[0, np.log(D["price_end"][:-1] / D["price_start"][:-1])], 0) / sig
    r5 = np.diff(np.log(S[:, ::5]), axis=1)
    rows, names = [], None
    for t in ts:
        f = {
            "x": x[:, t], "z": z[:, t], "m_over_sd": D["m"][:, t] / (sig * np.sqrt(np.maximum(D["V"][t], 1e-9))),
            "t": np.full(n, t), "log_sigma": np.log(np.maximum(sig, 1e-7)), "e": e[:, t],
            "de_30": e[:, t] - e[:, t - 30], "dx_5": x[:, t] - x[:, t - 5], "dx_15": x[:, t] - x[:, t - 15], "dx_30": x[:, t] - x[:, t - 30],
            "dz_5": z[:, t] - z[:, t - 5], "dz_15": z[:, t] - z[:, t - 15], "dz_30": z[:, t] - z[:, t - 30],
            **{f"ret_{h}": np.log(S[:, t] / S[:, t - h]) / (sig * np.sqrt(h)) for h in (5, 15, 30)},
            "flow_10": flow[:, t - 9:t + 1].sum(1), "flow_30": flow[:, t - 29:t + 1].sum(1), "flow_60": flow[:, max(t - 59, 0):t + 1].sum(1),
            "usd_30": usd[:, t - 29:t + 1].sum(1), "ntr_30": ntr[:, t - 29:t + 1].sum(1), "spread": np.nan_to_num(spread[:, t], nan=0.02),
            "rv_ratio": np.sqrt(np.nanmean(r5[:, : max(t // 5, 1)] ** 2, axis=1) / 5) / sig,
            "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24), "weekend": (dow >= 5).astype(float),
            "prev_y": prev_y, "prev_ret": prev_ret,
        }
        names = list(f)
        rows.append(np.column_stack([f[k] for k in names]))
    X = np.stack(rows, 1)                       # (N, len(ts), F)
    return X, names


def _seq(c: Ctx, ts=SNAPS, L=30):
    """Séquences des 30 dernières secondes pour le GRU : Δx, Δz, flux, e, x, z (N, len(ts), L, 6)."""
    D = c.D
    dx = np.diff(c.x, axis=1, prepend=np.nan); dz = np.diff(c.z, axis=1, prepend=np.nan)
    flow = np.sign(D["flow_usd"]) * np.log1p(np.abs(D["flow_usd"]) / 100)
    ch = np.stack([dx, dz, flow, c.e, c.x, c.z], -1)
    ch = np.nan_to_num(ch, nan=0.0)
    out = np.stack([ch[:, t - L + 1:t + 1] for t in ts], 1)
    return out.astype(np.float32)


def _logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _compare(p_model, p_mkt, y, groups, b=1000, start_s=None):
    """Log-loss gain of the model over the market, CI by bootstrap of markets.

    With start_s (market start of each prediction), also adds block bootstraps by UTC day and UTC hour.
    """
    d = _logloss(p_mkt, y) - _logloss(p_model, y)
    um, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, d); cnt = np.bincount(inv)
    bs = [s[j].sum() / cnt[j].sum() for j in (RNG.integers(0, len(um), len(um)) for _ in range(b))]
    out = {"logloss_model": float(_logloss(p_model, y).mean()), "logloss_market": float(_logloss(p_mkt, y).mean()),
           "gain": float(d.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)),
           "brier_model": float(((p_model - y) ** 2).mean()), "brier_market": float(((p_mkt - y) ** 2).mean())}
    if start_s is not None:
        out.update(bb.blocks_mean(d, start_s))
    return out


def section_D(c: Ctx) -> dict:
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(0)
    res = {}
    t0 = time.time()
    X3, names = _features(c)
    N, K, F = X3.shape
    mk = np.repeat(np.arange(N)[:, None], K, 1)
    y = np.repeat(c.y[:, None], K, 1)
    pm = np.clip(c.D["pm"][:, SNAPS], 0.005, 0.995)
    days = c.D["day"]
    val_days = np.unique(days[c.train])[-5:]
    fit = c.train & ~np.isin(days, val_days)
    val = c.train & np.isin(days, val_days)
    def flat(mask):
        return X3[mask].reshape(-1, F), y[mask].ravel(), pm[mask].ravel(), mk[mask].ravel()
    Xf, yf, pf, gf = flat(fit); Xv, yv, pv, gv = flat(val); Xtr, ytr, ptr, gtr = flat(c.train); Xte, yte, pte, gte = flat(c.test)
    ok = lambda A: np.all(np.isfinite(A), 1)
    print(f"  D data {time.time() - t0:.0f}s rows fit={len(yf)} test={len(yte)}", flush=True)

    # ---- D1 : outcome
    models = {}
    models["logistique"] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    models["gradient boosting"] = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
                                                                 min_samples_leaf=200, l2_regularization=1.0,
                                                                 early_stopping=True, validation_fraction=0.15, random_state=0)
    models["MLP"] = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                                                  validation_fraction=0.15, max_iter=200, random_state=0))
    d1 = {"features": names, "n_train_rows": int(len(ytr)), "n_test_rows": int(len(yte)), "models": {}}
    preds = {}
    o_tr, o_te = ok(Xtr), ok(Xte)
    for name, mdl in models.items():
        mdl.fit(Xtr[o_tr], ytr[o_tr])
        p = mdl.predict_proba(Xte[o_te])[:, 1]
        preds[name] = p
        d1["models"][name] = _compare(p, pte[o_te], yte[o_te], gte[o_te], start_s=c.D["start_s"][gte[o_te]])
        print(f"  D1 {name} {time.time() - t0:.0f}s gain={d1['models'][name]['gain']:.4f}", flush=True)
    # marché recalibré seul (logistique sur logit(p)) : la référence honnête
    lr = LogisticRegression(max_iter=1000).fit(logit(ptr[o_tr], 0.005)[:, None], ytr[o_tr])
    d1["models"]["marché recalibré"] = _compare(lr.predict_proba(logit(pte[o_te], 0.005)[:, None])[:, 1], pte[o_te], yte[o_te], gte[o_te],
                                                start_s=c.D["start_s"][gte[o_te]])

    # GRU sur les séquences de 60 s + features statiques
    Sq = _seq(c)
    def tens(mask):
        xs = torch.tensor(Sq[mask].reshape(-1, Sq.shape[2], 6)); xf = X3[mask].reshape(-1, F)
        return xs, xf, y[mask].ravel()
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-9
    smu = Sq[c.train].reshape(-1, 6).mean(0); ssd = Sq[c.train].reshape(-1, 6).std(0) + 1e-9

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(6, 32, batch_first=True)
            self.head = torch.nn.Sequential(torch.nn.Linear(32 + F, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, 1))
        def forward(self, s, f):
            _, h = self.gru(s)
            return self.head(torch.cat([h[-1], f], 1)).squeeze(1)

    def prep(mask):
        s, f, yy = tens(mask)
        s = (s - torch.tensor(smu)) / torch.tensor(ssd)
        f = torch.tensor(np.nan_to_num((f - mu) / sd), dtype=torch.float32)
        return s.float(), f, torch.tensor(yy, dtype=torch.float32)

    sF, fF, yF = prep(fit); sV, fV, yV = prep(val); sT, fT, yT = prep(c.test)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, best_state, patience = 1e9, None, 0
    for epoch in range(30):
        net.train()
        perm = torch.randperm(len(yF))
        for j in range(0, len(perm), 512):
            b = perm[j:j + 512]
            opt.zero_grad(); l = lossf(net(sF[b], fF[b]), yF[b]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(sV, fV), yV).item()
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        p_gru = torch.sigmoid(net(sT, fT)).numpy()
    d1["models"]["GRU (séquence 30 s)"] = _compare(p_gru, pte, yte, gte, start_s=c.D["start_s"][gte])
    d1["gru_epochs"] = epoch + 1
    print(f"  D1 GRU {time.time() - t0:.0f}s", flush=True)

    # gain par instant pour le meilleur modèle tabulaire
    best_name = max(["logistique", "gradient boosting", "MLP"], key=lambda k: d1["models"][k]["gain"])
    tt = Xte[o_te][:, names.index("t")]
    d1["by_t"] = [{"t": int(t), **_compare(preds[best_name][tt == t], pte[o_te][tt == t], yte[o_te][tt == t], gte[o_te][tt == t], b=300)}
                  for t in SNAPS]
    d1["best_tabular"] = best_name
    # importance par permutation (gradient boosting, test)
    gb = models["gradient boosting"]
    sub = RNG.choice(np.nonzero(o_te)[0], size=min(8000, o_te.sum()), replace=False)
    pi = permutation_importance(gb, Xte[sub], yte[sub], scoring="neg_log_loss", n_repeats=5, random_state=0)
    d1["importance_gb"] = sorted([{"feature": n_, "drop_logloss": float(m_)} for n_, m_ in zip(names, pi.importances_mean)],
                                 key=lambda r: -r["drop_logloss"])
    res["D1_outcome"] = d1

    # ---- D2 : variation du prix à 10 s (x_{t+10} − x_t), toutes les 10 s
    ts2 = np.arange(30, 281, 10)
    X2, _ = _features(c, ts2)
    tgt = c.x[:, ts2 + 10] - c.x[:, ts2]
    def flat2(mask):
        return X2[mask].reshape(-1, F), tgt[mask].ravel(), np.repeat(np.arange(N)[mask][:, None], len(ts2), 1).ravel()
    A_tr, b_tr, _ = flat2(c.train); A_te, b_te, g_te = flat2(c.test)
    o1 = ok(A_tr) & np.isfinite(b_tr); o2 = ok(A_te) & np.isfinite(b_te)
    d2 = {}
    for name, mdl in {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                      "gradient boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=300,
                                                                         early_stopping=True, random_state=0)}.items():
        mdl.fit(A_tr[o1], b_tr[o1])
        pr = mdl.predict(A_te[o2])
        r2 = 1 - np.mean((b_te[o2] - pr) ** 2) / np.var(b_te[o2])
        hit = np.mean(np.sign(pr) == np.sign(b_te[o2])[np.newaxis][0]) if True else None
        nz = b_te[o2] != 0
        d2[name] = {"r2_test": float(r2), "direction_hit": float(np.mean(np.sign(pr[nz]) == np.sign(b_te[o2][nz]))),
                    "corr": float(np.corrcoef(pr, b_te[o2])[0, 1])}
        if name == "ridge":
            coefs = mdl[-1].coef_
            d2["ridge_top_coefs"] = sorted([{"feature": n_, "coef_std": float(v)} for n_, v in zip(names, coefs)], key=lambda r: -abs(r["coef_std"]))[:8]
    # le mouvement prévisible dépasse-t-il le demi-spread ? (en probabilité)
    pr_gb = mdl.predict(A_te[o2])
    xt = A_te[o2][:, names.index("x")]
    p0 = 1 / (1 + np.exp(-xt)); p1 = 1 / (1 + np.exp(-(xt + pr_gb)))
    d2["share_pred_move_gt_1c"] = float(np.mean(np.abs(p1 - p0) > 0.01))
    d2["median_abs_pred_move_c"] = float(np.median(np.abs(p1 - p0)) * 100)
    res["D2_price_10s"] = d2
    print(f"  D2 {time.time() - t0:.0f}s", flush=True)

    # ---- D3 : prévision de la vol réalisée BTC sur [60, 300) avec l'information connue à t = 60
    D = c.D
    S = D["S"]; sig = D["sigma"]
    r5 = np.diff(np.log(S[:, ::5]), axis=1)
    rv_after = np.log(np.sqrt(np.mean(r5[:, 12:] ** 2, axis=1) / 5) + 1e-9)
    rv_first = np.log(np.sqrt(np.mean(r5[:, :12] ** 2, axis=1) / 5) + 1e-9)
    pre = D["btc_pre"]; r5p = np.diff(np.log(pre[:, ::5]), axis=1)
    rv_5m = np.log(np.sqrt(np.mean(r5p[:, -60:] ** 2, axis=1) / 5) + 1e-9)
    rv_1h = np.log(np.sqrt(np.mean(r5p ** 2, axis=1) / 5) + 1e-9)
    st = D["start_s"]; hour = (st % 86400) / 3600; dow = ((st // 86400) + 4) % 7
    t = 60
    from scipy.special import ndtri
    q = ndtri(np.clip(D["pm"][:, t], 1e-4, 1 - 1e-4))
    good = (np.abs(q) > 0.3) & (np.sign(q) == np.sign(D["m"][:, t]))
    iv = np.where(good, 0.5 * np.log(np.clip(((D["m"][:, t] / np.where(good, q, 1)) ** 2 - (4e-5) ** 2) / D["V"][t], 1e-16, None)), np.nan)
    base = np.column_stack([np.log(sig), rv_first, rv_5m, rv_1h, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), (dow >= 5)])
    d3 = {}
    for label, Xv3 in (("HAR (vol passées + heure)", base), ("HAR + vol implicite Prediction Market", np.column_stack([base, iv]))):
        for mname, mdl in (("linéaire", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                           ("gradient boosting", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=100, random_state=0))):
            sel_tr = c.train & np.all(np.isfinite(Xv3), 1); sel_te = c.test & np.all(np.isfinite(Xv3), 1)
            # même échantillon pour les deux spécifications : marchés avec une vol implicite définie
            sel_tr &= np.isfinite(iv); sel_te &= np.isfinite(iv)
            mdl.fit(Xv3[sel_tr], rv_after[sel_tr])
            pr = mdl.predict(Xv3[sel_te])
            d3[f"{label} · {mname}"] = {"r2_test": float(1 - np.mean((rv_after[sel_te] - pr) ** 2) / np.var(rv_after[sel_te])), "n_test": int(sel_te.sum())}
    res["D3_vol_forecast"] = d3
    print(f"  D3 {time.time() - t0:.0f}s", flush=True)
    return res
