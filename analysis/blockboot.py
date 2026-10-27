"""Cluster and block bootstraps.

Each draw resamples whole groups of observations with replacement, so the dependence inside a group is kept. Three
groupings are used:

- market: one 5-minute market (the unit of the original analysis);
- day: the UTC day of the market start (288 markets share one volatility regime, and some features use the previous
  window). This is the reference for the verdicts;
- hour: the UTC hour of the market start (12 markets), a sensitivity analysis.

The block draws use their own generator, so adding them does not change the market-bootstrap draws of the analyses.
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20261026)
B = 2000
UNITS = {"day": 86400, "hour": 3600}


def block_keys(start_s, unit: str) -> np.ndarray:
    """Group key of each observation from the market start (Unix seconds): UTC day or UTC hour."""
    return np.asarray(start_s, dtype=np.int64) // UNITS[unit]


def draw_counts(n_groups: int, b: int, rng=RNG) -> np.ndarray:
    """(b, n_groups) matrix: how many times each group is drawn in each bootstrap replicate."""
    idx = rng.integers(0, n_groups, size=(b, n_groups))
    counts = np.zeros((b, n_groups))
    np.add.at(counts, (np.repeat(np.arange(b), n_groups), idx.ravel()), 1.0)
    return counts


def cluster_mean(x, groups, b: int = B, rng=RNG, h0: float = 0.0) -> dict:
    """Mean of x with a bootstrap that resamples groups.

    The replicate statistic is sum(x in drawn groups) / count(x in drawn groups), i.e. the pooled mean of the drawn
    observations. p is one-sided (H0: mean <= h0); se is the standard deviation of the replicates.
    """
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(x)
    x, groups = x[ok], groups[ok]
    if len(x) < 5:
        return {"n": int(len(x)), "n_groups": None, "mean": None, "lo": None, "hi": None, "se": None, "p": None}
    _, inv = np.unique(groups, return_inverse=True)
    g = int(inv.max()) + 1
    s = np.bincount(inv, x, minlength=g)
    c = np.bincount(inv, minlength=g).astype(float)
    w = draw_counts(g, b, rng)
    bs = (w @ s) / (w @ c)
    return {"n": int(len(x)), "n_groups": g, "mean": float(x.mean()), "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)), "se": float(bs.std(ddof=1)), "p": float((np.sum(bs <= h0) + 1) / (b + 1))}


def blocks_mean(x, start_s, b: int = B, rng=RNG) -> dict:
    """cluster_mean by UTC day and by UTC hour; start_s is the market start of each observation."""
    return {u: cluster_mean(x, block_keys(start_s, u), b, rng) for u in UNITS}


def ci(draws: np.ndarray) -> dict:
    """Percentile 95 % CI and standard deviation of bootstrap replicates (columns = parameters)."""
    return {"lo": np.quantile(draws, 0.025, 0).tolist(), "hi": np.quantile(draws, 0.975, 0).tolist(),
            "se": draws.std(0, ddof=1).tolist()}
