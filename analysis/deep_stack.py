"""D1 bis : modèles ML « empilés » sur le prix de marché (comparaison équitable avec le marché).

Le modèle est entraîné sur l'exploration hors 5 derniers jours ; une logistique y ~ logit(p_marché) + logit(p_ML)
est ajustée sur ces 5 jours de validation ; le tout est évalué une fois sur le test.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.deep import Ctx, logit
from analysis.deep_cd import SNAPS, _compare, _features


def main() -> None:
    c = Ctx()
    X3, names = _features(c)
    N, K, F = X3.shape
    y = np.repeat(c.y[:, None], K, 1)
    pm = np.clip(c.D["pm"][:, SNAPS], 0.005, 0.995)
    mk = np.repeat(np.arange(N)[:, None], K, 1)
    days = c.D["day"]
    val_days = np.unique(days[c.train])[-5:]
    fit = c.train & ~np.isin(days, val_days); val = c.train & np.isin(days, val_days)
    def flat(m):
        A = X3[m].reshape(-1, F); ok = np.all(np.isfinite(A), 1)
        return A[ok], y[m].ravel()[ok], pm[m].ravel()[ok], mk[m].ravel()[ok], A[ok][:, names.index("t")]
    Af, yf, pf, _, _ = flat(fit); Av, yv, pv, _, _ = flat(val); At, yt, pt, gt, tt = flat(c.test)
    out = {}
    for name, mdl in {
        "logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gradient boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, min_samples_leaf=200,
                                                            l2_regularization=1.0, random_state=0),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, random_state=0)),
    }.items():
        mdl.fit(Af, yf)
        def stack_X(A, p):
            return np.column_stack([logit(p, 0.005), logit(mdl.predict_proba(A)[:, 1], 0.005)])
        st = LogisticRegression(max_iter=1000).fit(stack_X(Av, pv), yv)
        p = st.predict_proba(stack_X(At, pt))[:, 1]
        r = _compare(p, pt, yt, gt, start_s=c.D["start_s"][gt])
        r["stack_weights"] = st.coef_[0].tolist()
        r["by_t"] = [{"t": int(t), **{k: v for k, v in _compare(p[tt == t], pt[tt == t], yt[tt == t], gt[tt == t], b=300).items()
                                       if k in ("gain", "lo", "hi")}} for t in SNAPS]
        out[name] = r
        print(name, round(r["gain"], 4), round(r["lo"], 4), round(r["hi"], 4), "w", np.round(r["stack_weights"], 3),
              [(x["t"], round(x["gain"], 4), round(x["lo"], 4)) for x in r["by_t"]][-3:], flush=True)
    d = json.load(open("results/deep_D.json", encoding="utf-8"))
    d["D1_stacked"] = out
    json.dump(d, open("results/deep_D.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
