"""Sections ML de la recherche d'edges : clusters de marchés (E6) et classifieur de rebonds (E8)."""

from __future__ import annotations

import numpy as np

from analysis.deep import RNG, boot_stat
from analysis.edges import W0, X, _dip_sim, fee, mboot


def _market_features(x: X):
    c = x.c; D = c.D
    pm = D["pm"]
    dp5 = np.diff(pm[:, 10:290:5], axis=1)
    cross = np.nansum(np.abs(np.diff(np.sign(pm[:, 10:290] - 0.5), axis=1)) > 0, axis=1)
    dist = np.abs(pm - 0.5)
    t_decisive = np.argmax(np.nan_to_num(dist, nan=0) >= 0.3, axis=1).astype(float)
    t_decisive[~(np.nan_to_num(dist, nan=0) >= 0.3).any(1)] = 300
    usd = D["usd"].sum(1)
    S = D["S"]; r5 = np.diff(np.log(S[:, ::5]), axis=1)
    rv = np.sqrt((r5**2).mean(1) / 5)
    hour = (D["start_s"] % 86400) / 3600
    F = {
        "vol_prix": np.nanstd(dp5, 1), "croisements_50": cross.astype(float), "t_decisif": t_decisive,
        "log_volume_usd": np.log1p(usd), "trades": D["n_trades"].sum(1).astype(float),
        "taille_moy_usd": usd / np.maximum(D["n_trades"].sum(1), 1), "spread_moy": np.nanmean(np.where((D["ask_up"] + D["ask_dn"] - 1) > -.5, D["ask_up"] + D["ask_dn"] - 1, np.nan), 1),
        "vol_btc_sigma": np.log(np.maximum(rv / D["sigma"], 1e-3)), "log_sigma": np.log(np.maximum(D["sigma"], 1e-7)),
        "mouvement_btc": np.abs(D["m"][:, -1]) / (D["sigma"] * np.sqrt(300)), "residu_sd": np.nanstd(c.e[:, 10:290], 1),
        "heure_sin": np.sin(2 * np.pi * hour / 24), "heure_cos": np.cos(2 * np.pi * hour / 24),
    }
    names = list(F)
    return np.column_stack([F[k] for k in names]), names


def E6(x: X) -> dict:
    """Clusters de marchés (k-means) et composantes principales des trajectoires de prix."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    c = x.c; D = c.D
    F, names = _market_features(x)
    ok = np.all(np.isfinite(F), 1) & c.valid
    sc = StandardScaler().fit(F[ok & c.train])
    Z = sc.transform(np.nan_to_num(F))
    sil = {}
    for k in range(3, 8):
        km = KMeans(k, n_init=10, random_state=0).fit(Z[ok & c.train])
        sub = RNG.choice(np.nonzero(ok & c.train)[0], 3000, replace=False)
        sil[k] = float(silhouette_score(Z[sub], km.predict(Z[sub])))
    k_best = max(sil, key=sil.get)
    k_use = max(k_best, 4)
    km = KMeans(k_use, n_init=20, random_state=0).fit(Z[ok & c.train])
    lab = km.predict(Z)
    # P&L du dip 0,30/0,33 par cluster
    clusters = []
    for k in range(k_use):
        row = {"cluster": k, "share": float(((lab == k) & ok).mean() / ok.mean())}
        row["centre"] = {n_: float(v) for n_, v in zip(names, sc.inverse_transform(km.cluster_centers_)[k])}
        for name, msk in c.splits():
            s = msk & ok & (lab == k)
            dp, dm = _dip_sim(x, s, 0.30, 0.03, 0, 240)
            row[name] = {"n": int(s.sum()),
                         "dip_30_33": float(dp.mean()) if len(dp) else None, "dip_30_33_boot": mboot(dp, dm),
                         "favori_240_gagne": float(((D["pm"][s, 240] >= .5) == c.y[s].astype(bool)).mean())}
        clusters.append(row)
    # stabilité : la part de chaque cluster sur le test
    pca = PCA(6).fit(np.nan_to_num(D["pm"][ok & c.train][:, 0:300:10]))
    comps = pca.components_
    res = {"silhouette": sil, "k": k_use, "features": names, "clusters": clusters,
           "pca_var": pca.explained_variance_ratio_.tolist(), "pca_components": comps[:3].tolist(), "pca_t": list(range(0, 300, 10))}
    return res


def E8(x: X) -> dict:
    """Peut-on prédire quels creux (achat limite à 0,30) rebondissent jusqu'à 0,33 ?"""
    from sklearn.ensemble import HistGradientBoostingClassifier

    c = x.c; D = c.D
    rows, yv, pnl, mk = [], [], [], []
    tgrid = np.arange(W0, W0 + 300)
    for sd in ("up", "down"):
        S = x.side[sd]
        sl = S["sl"][:, tgrid]; bh = S["bh"][:, tgrid]
        th = np.nan_to_num(sl, nan=9) < 0.30
        th[:, 240:] = False; th[:, :5] = False
        filled = th.any(1); first = np.argmax(th, 1)
        sign = 1.0 if sd == "up" else -1.0
        for i in np.nonzero(filled & c.valid)[0]:
            t = first[i]
            ex = (np.nan_to_num(bh[i, t + 1:], nan=-1) > 0.33).any()
            yy = S["y"][i]
            rows.append([t, sign * (D["P"][i, t] - D["pm"][i, t]), sign * (c.z[i, t] - c.z[i, max(t - 5, 0)]),
                         sign * (c.e[i, t]), sign * D["flow_usd"][i, max(t - 10, 0):t + 1].sum() / 1000, np.log(D["sigma"][i]),
                         sign * (D["pm"][i, t] - D["pm"][i, max(t - 10, 0)]), D["n_trades"][i, max(t - 10, 0):t + 1].sum()])
            yv.append(ex); pnl.append(0.03 if ex else yy - 0.30); mk.append(i)
    Xf = np.array(rows); yv = np.array(yv); pnl = np.array(pnl); mk = np.array(mk)
    names = ["t", "ecart_modele", "btc_5s", "residu", "flux_10s", "log_sigma", "dp_10s", "trades_10s"]
    tr = c.train[mk]; te = c.test[mk]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=100, random_state=0).fit(Xf[tr], yv[tr])
    pr_tr = clf.predict_proba(Xf[tr])[:, 1]; pr_te = clf.predict_proba(Xf[te])[:, 1]
    from sklearn.metrics import roc_auc_score
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "auc_test": float(roc_auc_score(yv[te], pr_te)),
           "base_rebound_test": float(yv[te].mean()), "base_pnl_test": mboot(pnl[te], mk[te], 500)}
    # seuil choisi sur l'exploration : le quantile de proba prédite qui maximise le P&L moyen (≥ 200 trades)
    best = None
    for qq in np.arange(0.5, 0.96, 0.05):
        thr = np.quantile(pr_tr, qq); sel = pr_tr >= thr
        if sel.sum() >= 200 and (best is None or pnl[tr][sel].mean() > best[1]):
            best = (qq, pnl[tr][sel].mean(), thr)
    sel_te = pr_te >= best[2]
    res["selected_quantile"] = float(best[0]); res["train_pnl_selected"] = float(best[1])
    res["test_selected"] = mboot(pnl[te][sel_te], mk[te][sel_te], 500)
    # rebond attendu par la martingale pour ce sous-ensemble : prix de touche / 0,33 (≈ 0,26 / 0,33)
    res["features"] = names
    return res
