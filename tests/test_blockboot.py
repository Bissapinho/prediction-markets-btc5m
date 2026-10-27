import numpy as np

from analysis import blockboot as bb


def test_block_keys_utc_day_and_hour():
    # 2026-09-21 00:00 UTC = 1789948800; 23:55 the same day; 00:00 the next day.
    s = np.array([1789948800, 1789948800 + 86100, 1789948800 + 86400])
    assert list(bb.block_keys(s, "day")) == [20717, 20717, 20718]
    assert list(bb.block_keys(s, "hour")) == [497208, 497231, 497232]


def test_draw_counts_resample_every_group_once_on_average():
    w = bb.draw_counts(27, 4000, np.random.default_rng(0))
    assert w.shape == (4000, 27)
    assert np.all(w.sum(1) == 27)              # each replicate draws 27 days
    assert abs(w.mean() - 1.0) < 0.01


def test_cluster_mean_known_variance():
    # 27 days of 288 markets; all markets of a day share the same value (perfect within-day dependence).
    # The day bootstrap of the mean is then the mean of 27 draws from the 27 day values:
    # sd = sd_pop(day values) / sqrt(27). A bootstrap by market would give a far smaller sd (~ sd_pop / sqrt(7776)).
    day_values = np.linspace(-1.0, 1.0, 27)
    x = np.repeat(day_values, 288)
    days = np.repeat(np.arange(27), 288)
    r = bb.cluster_mean(x, days, b=20000, rng=np.random.default_rng(1))
    expected_sd = day_values.std() / np.sqrt(27)
    assert r["n"] == 27 * 288 and r["n_groups"] == 27
    assert abs(r["mean"]) < 1e-12
    assert abs(r["se"] - expected_sd) / expected_sd < 0.03
    assert abs(r["lo"] + 1.96 * expected_sd) < 0.03 and abs(r["hi"] - 1.96 * expected_sd) < 0.03
    by_market = bb.cluster_mean(x, np.arange(len(x)), b=2000, rng=np.random.default_rng(2))
    assert by_market["se"] < r["se"] / 10


def test_cluster_mean_constant_and_pvalue():
    x = np.full(100, 0.5)
    r = bb.cluster_mean(x, np.repeat(np.arange(10), 10), b=999, rng=np.random.default_rng(3))
    assert r["lo"] == r["hi"] == 0.5 and r["se"] == 0.0
    assert r["p"] == 1 / 1000                   # no replicate <= 0
    r = bb.cluster_mean(-x, np.repeat(np.arange(10), 10), b=999, rng=np.random.default_rng(3))
    assert r["p"] == 1.0


def test_cluster_mean_unequal_groups_is_pooled_mean():
    # Groups of sizes 1 and 5: the estimate is the pooled mean (6 / 6 = 1), not the mean of group means (3);
    # each replicate is also a pooled mean, so it lies in {0, 1, 6}. NaN values are dropped.
    x = np.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan])
    r = bb.cluster_mean(x, np.array([0, 1, 1, 1, 1, 1, 1]), b=500, rng=np.random.default_rng(4))
    assert r["n"] == 6 and r["n_groups"] == 2 and r["mean"] == 1.0
    assert r["lo"] == 0.0 and r["hi"] == 6.0
