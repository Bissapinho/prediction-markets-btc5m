# Methods and full results

This document holds the detailed method and the full tables behind the [README](../README.md). Every number below is
in [results/key_figures.md](../results/key_figures.md) with its source file, and `scripts/check_figures.py` verifies
this at the end of `reproduce.sh`.

## Data

**Venue and contract.** The venue lists one binary "Bitcoin Up or Down – 5 min" market per 5-minute window. Up pays
1 $ if the 60-second TWAP of Chainlink BTC/USD at the end of the window is at least the TWAP at its start.

| | |
|---|---|
| Period | Markets starting 2026-08-17 00:00 → 2026-10-17 23:50 UTC (62 days) |
| Markets | **17,559** analysed, out of 17,856 five-minute windows in the period |
| Trades | **28,594,392** taker trades on these markets (26,956,412 inside the 5-minute windows), 435.7 M USDC notional |
| Reference price | Binance BTCUSDT aggregated trades: 58,105,834 trades, resampled to 1-second bars |
| Per-second panel | 5,267,700 market-second rows + 5,358,300 Binance 1-second rows = **10.6 M rows** |

**Sources.**

- Market metadata and outcomes come from the venue's public market API.
- Trades come from its public trade-history API: taker side, one row per match. They were checked against the on-chain
  fill events on Polygon for 4 markets, and every trade matched (details in [data_notes.md](data_notes.md)).
- Binance data comes from `data.binance.vision`.

**Filters** ([analysis/data.py](../analysis/data.py)):

| Step | Markets |
|---|---|
| Five-minute windows from 2026-08-17 to 2026-10-17 | 17,856 |
| Listed by the market API | 17,855 |
| At least one trade inside [0, 300 s) (drops the 2026-09-19 venue outage, among others) | 17,560 |
| Binance available from start − 1,800 s to start + 300 s | **17,559** |

The study starts on 2026-08-17 because the resolution rule changed in early August. Until about 2026-08-10 the markets
resolved on the Chainlink *spot* price; since then they resolve on the 60-second TWAP. The fair-value model only applies
to the TWAP period.

## Method

**Train/test split by date.** Exploration uses the markets before **2026-09-21** (9,784 markets), and the test uses the
markets from that date on (7,775 markets, 27 days). Every free parameter is chosen on the exploration period only:

- the fair-value model (k, τ);
- each strategy's threshold, which must give at least 200 trades on train;
- ML hyperparameters, early stopping and stacking weights (on the last 5 days of train).

**Use of the test set.** The test set was used to detect two data artefacts (stale ask, timestamp bias). After these
corrections, each configuration was evaluated once on the test period.

**Statistical unit and dependence.**

- Each strategy takes at most one position per market.
- The 288 markets of a day share one volatility regime, and some features use the previous window, so markets are not
  independent.
- Confidence intervals and p-values therefore come from a block bootstrap that resamples UTC days
  ([analysis/blockboot.py](../analysis/blockboot.py); unit test in
  [tests/test_blockboot.py](../tests/test_blockboot.py)). Each replicate draws 27 test days with replacement and
  recomputes the statistic on all observations of the drawn days.
- It is applied to every P&L, to the regressions (H1 lead-lag, H2-info, A9) and to the ML log-loss gains. The verdicts
  use it.
- The bootstrap that resamples markets is kept as a secondary column, and a bootstrap by UTC hour is a sensitivity
  analysis.
- Draws: 2,000 for the P&L and log-loss means by day and by hour, 500 for the H1 regression and the H2 logit, and 300
  for the A9 logit.
- P&L p-values are one-sided (H0: mean P&L ≤ 0). The other p-values come from a normal approximation of the bootstrap
  distribution.

**Multiple testing.** Benjamini–Hochberg at q = 0.05 is applied within each family, separately for each bootstrap
unit:

- **P**, the protocol specified before the test evaluation: H1, H2, H2-info, H3, H4;
- **A**, price-path strategies: A1, A2, A4–A7;
- **I**, information beyond the price: A9–A11;
- the 8 ML models.

A3 is not in any family. Its market cluster is defined from the whole price path, so it is reported as an ex-post
diagnostic. A8 is an artefact of the price data and is not tested. The table therefore holds 15 hypotheses: 14 tests
in families, 1 ex-post diagnostic and 1 artefact. No hypothesis is called significant unless it is significant after
this correction with the bootstrap by day.

**Power.** The smallest edge detectable with 80 % power is (z₀.₉₅ + z₀.₈₀) · se, with se taken from the bootstrap CI.
For H1–H4 it is 1.3 to 1.5 c/share with the bootstrap by day and 1.4 to 1.7 c/share with the bootstrap by market.

**Costs.** The taker fee is 0.07 · p · (1 − p) per share, verified on-chain (1.75 c at 50 c, 0.33 c at 95 c). Limit
orders pay no fee.

**Execution.**

- The decision uses only what is known at the end of second t.
- A market order is filled at the ask observed at t + L. L is 2 s for H1–H4, with a sensitivity analysis over L ∈ {0,
  1, 2, 3, 5}, and 1 s for the A family, which favours the strategy.
- There is no historical order book. The ask is the last taker buy of the last 3 s, floored by the mirror book
  (ask_up ≥ 1 − last buy of Down).
- A limit order counts as filled only when a trade goes *through* its price.
- The position is held to resolution unless the strategy defines an exit.

**Fair-value model (H2).** P(Up | Binance) = Φ(ln(E[A1 | S_t] / A0) / sd), with:

- A0 and A1 the Binance 60 s TWAPs at the start and end of the window;
- sd² = (kσ)² V(t) + τ², where V(t) is the exact variance of the remaining part of a Brownian average and σ is a causal
  30-minute realised volatility.

Fitted on train: k = 1.15, τ = 4e-05. The model reproduces the outcome at t = 299 s in 98.0 % of test markets.

## Full hypothesis table

Test period. P&L per share (1 share pays 1 $ if it wins), fees included. Source:
[results/hypotheses.md](../results/hypotheses.md).

### Tests (Benjamini-Hochberg within each family)

| ID | Hypothesis | Execution | Statistic (test) | Estimate | n | 95 % CI (day) | p (day) | p adj. (day) | 95 % CI (market) | p adj. (market) | p adj. (hour) | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Lead-lag: buy when Binance moves >= z0 sigma in 3 s and the market has not moved | taker, 2 s | mean P&L per share, fees included | +1.0 c | 2,570 | [-0.1, +2.1] | 0.034 | 0.086 | [-0.1, +2.2] | 0.111 | 0.104 | P (protocol) | Not significant after BH |
| H2 | TWAP fair value from Binance: buy the side priced below the model | taker, 2 s | mean P&L per share, fees included | -2.7 c | 6,821 | [-3.9, -1.6] | 1.000 | 1.000 | [-3.8, -1.7] | 1.000 | 1.000 | P (protocol) | Rejected: negative P&L |
| H2-info | Model minus market predicts the outcome (logit coefficient c > 0) | – | coefficient c | 0.124 | 69,975 | [0.031, 0.215] | 0.004 | 0.020 | [0.055, 0.195] | 0.010 | 0.020 | P (protocol) | Information detected after BH; the taker strategy built on it (H2) loses |
| H3 | Fade price jumps that Binance does not justify | taker, 2 s | mean P&L per share, fees included | -2.2 c | 6,974 | [-3.3, -1.2] | 1.000 | 1.000 | [-3.3, -1.1] | 1.000 | 1.000 | P (protocol) | Rejected: negative P&L |
| H4 | Late favourites (t = 240-290 s, ask 0.80-0.98) | taker, 2 s | mean P&L per share, fees included | -0.4 c | 1,313 | [-1.6, +0.7] | 0.782 | 1.000 | [-1.9, +0.9] | 1.000 | 1.000 | P (protocol) | Rejected: negative P&L |
| A1 | Dip-buy: buy a side below 30 c, sell at 33 c | taker, 1 s | mean P&L per share, fees included | -6.0 c | 4,614 | [-6.6, -5.4] | 1.000 | 1.000 | [-6.5, -5.4] | 1.000 | 1.000 | A (price paths) | Rejected: negative P&L |
| A2 | Dip-buy grid (216 cells), best cell on train | limit order | mean P&L per share, fees included | -1.0 c | 870 | [-1.1, -0.8] | 1.000 | 1.000 | [-1.1, -0.8] | 1.000 | 1.000 | A (price paths) | Rejected: negative P&L |
| A4 | ML classifier picks the dips that rebound (test AUC 0.59) | limit order | mean P&L per share, fees included | -1.4 c | 466 | [-2.5, -0.4] | 0.996 | 1.000 | [-2.5, -0.4] | 1.000 | 1.000 | A (price paths) | Rejected: negative P&L |
| A5 | Pre-open momentum: buy the side of the last 1-min Binance trend | taker, 1 s | mean P&L per share, fees included | -3.6 c | 5,393 | [-4.7, -2.5] | 1.000 | 1.000 | [-5.0, -2.3] | 1.000 | 1.000 | A (price paths) | Rejected: negative P&L |
| A6 | TWAP anchor at the open: buy model minus opening price | taker, 1 s | mean P&L per share, fees included | -3.0 c | 1,541 | [-5.8, -0.3] | 0.989 | 1.000 | [-5.2, -0.3] | 1.000 | 1.000 | A (price paths) | Rejected: negative P&L |
| A7 | End-of-window sniping: buy at <= 99 c a side the model gives > 99.5 % | taker, 1 s | mean P&L per share, fees included | -0.8 c | 23 | [-13.1, +5.7] | 0.547 | 1.000 | [-10.7, +5.8] | 1.000 | 1.000 | A (price paths) | Rejected: negative P&L |
| A8 | Pair arbitrage (Up + Down < 1 $) | – | share of seconds with ask_up + ask_dn < 1 | 52% | 919,083 | – | – | – | – | – | – | – | Artefact: the two prices come from different instants of the same second |
| A9 | Under-reaction at the open: slope of outcome on logit(price), 10-40 s (H0: slope = 1) | – | logit slope | 1.223 | 7,766 | [1.158, 1.290] | < 0.001 | < 0.001 | [1.150, 1.304] | < 0.001 | < 0.001 | I (information) | Information detected after BH (statistical test, not a P&L test) |
| A10 | ML forecast of the outcome stacked on the market price | – | log-loss gain vs market (nats) | 0.0008 | 69,894 | [0.0001, 0.0016] | 0.014 | 0.014 | [0.0001, 0.0016] | 0.018 | 0.028 | I (information) | Information detected after BH (statistical test, not a P&L test) |
| A11 | Price residual (not explained by Binance) leads the next 10 s Binance move | – | mean per-market correlation | 0.082 | 7,766 | [0.076, 0.088] | < 0.001 | < 0.001 | [0.079, 0.086] | < 0.001 | < 0.001 | I (information) | Information detected after BH (statistical test, not a P&L test) |

### Ex-post diagnostics (not in any BH family; p adj. is not computed)

| ID | Hypothesis | Execution | Statistic (test) | Estimate | n | 95 % CI (day) | p (day) | p adj. (day) | 95 % CI (market) | p adj. (market) | p adj. (hour) | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | Dip-buy restricted to the 'hesitant' market cluster (cluster defined from the whole price path) | limit order | mean P&L per share, fees included | +1.9 c | 2,236 | [+1.7, +2.2] | < 0.001 | – | [+1.7, +2.2] | – | – | ex-post diagnostic | Ex-post diagnostic: the cluster uses the whole price path (look-ahead), so this is not a tradable strategy; not in any BH family |

**Hypothesis details.**

- H1 buys when Binance moves ≥ 4 σ in 3 s while the market price has not moved.
- H2 buys the side priced at least θ = 5 c below the model.
- H3 fades jumps ≥ 8 c in 5 s.
- H4 buys the favourite at t = 260 s.
- A2 is the best cell of a 216-cell grid on train.
- A4 uses a gradient-boosting classifier with a test AUC of 0.59.

**Reading.**

- No strategy in the tested families has a positive P&L after the correction.
- The information tests find statistically detectable effects of a few probability points. These hypotheses were not
  built to turn the effects into a P&L.
- The bootstrap by day changes no verdict compared with the bootstrap by market.
- In the hour-bootstrap sensitivity analysis, the stacked logistic model loses its significance. It is A10, and also
  the stacked logistic row of the ML table.

![Calibration](../figures/fig1_calibration.png)

*Figure 1. Calibration on the test period, pooled over snapshots every 10 s.*

- x: mean predicted P(Up) in each decile; y: realised frequency of Up.
- Error bars: 95 % binomial intervals with n divided by the number of snapshots per market.
- The market price (blue) is on the diagonal. The Binance TWAP model (orange) is over-confident late in the window.
- Test log-loss of the market vs the model: 0.592 vs 0.603 (10–90 s), 0.382 vs 0.403 (100–230 s), 0.077 vs 0.101
  (240–290 s).

## Machine learning and deep learning

**Task and score.** The task is to forecast the outcome at t = 30, 60, …, 270 s from 28 features known at t:

- market price and its changes;
- Binance returns and volatility;
- model fair value;
- order flow, volume and spread;
- hour and previous window.

The score is the log-loss gain over the market price on the test period, over 69,894 market × snapshot predictions.

**Models.**

- logistic regression (standardised, C = 1);
- gradient boosting (scikit-learn HistGradientBoosting, early stopping);
- MLP (64, 32);
- GRU (PyTorch, hidden size 32) on the last 30 s of 6 channels, plus the static features;
- "stacked" variants: a logistic regression on [logit p_market, logit p_model], fitted on the last 5 training days.

Source: [results/ml.md](../results/ml.md).

| Model | Setting | Log-loss (model) | Log-loss (market) | Gain (nats) | 95 % CI (day) | p adj. (day) | 95 % CI (market) | p adj. (market) | p adj. (hour) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | standalone | 0.3864 | 0.3883 | +0.0019 | [+0.0011, +0.0027] | < 0.001 | [+0.0012, +0.0027] | < 0.001 | < 0.001 |
| Gradient boosting (HistGB) | standalone | 0.3997 | 0.3883 | -0.0114 | [-0.0144, -0.0085] | 1.000 | [-0.0146, -0.0084] | 1.000 | 1.000 |
| MLP (64, 32) | standalone | 0.3909 | 0.3883 | -0.0026 | [-0.0045, -0.0008] | 1.000 | [-0.0042, -0.0012] | 1.000 | 1.000 |
| Market price, recalibrated (logistic on logit p) | standalone | 0.3878 | 0.3883 | +0.0005 | [+0.0002, +0.0008] | 0.001 | [+0.0002, +0.0008] | 0.009 | 0.008 |
| GRU (30 s sequences + static features) | standalone | 0.3883 | 0.3885 | +0.0002 | [-0.0014, +0.0017] | 0.670 | [-0.0010, +0.0013] | 0.630 | 0.639 |
| Logistic regression | stacked on market price | 0.3874 | 0.3883 | +0.0008 | [+0.0001, +0.0016] | 0.039 | [+0.0001, +0.0016] | 0.047 | 0.075 |
| Gradient boosting (HistGB) | stacked on market price | 0.3898 | 0.3883 | -0.0016 | [-0.0028, -0.0003] | 1.000 | [-0.0029, -0.0004] | 1.000 | 1.000 |
| MLP (64, 32) | stacked on market price | 0.3881 | 0.3883 | +0.0002 | [-0.0006, +0.0009] | 0.616 | [-0.0006, +0.0010] | 0.630 | 0.639 |

**Reading.**

- The flexible models (boosting, MLP) do worse than the market price out of sample, and the GRU does not beat it.
- The best gain is +0.0019 nats (logistic regression, best of 8 models), 0.5 % of the market's log-loss. It is the
  maximum over 8 models evaluated on the test period, so it is an optimistic figure.
- A quarter of it (+0.0005) comes from recalibrating the market price.

![ML vs market](../figures/fig4_ml_vs_market.png)

*Figure 4. Log-loss gain over the market price on the test period, in 10⁻³ nats per prediction (> 0 means better than
the market). Bars: 95 % CI, bootstrap by UTC day. Blue: significant after BH across the 8 models.*

## Timestamp bias

**Measurement.**

- The trade-history API timestamps each trade with its Polygon block, not with its execution.
- Over one day, a live capture of the venue's WebSocket gave the execution time (ms) of trades, which were matched on
  transaction hash with the trade history.
- 4,479 trades matched. Block − execution has a median of 2.25 s (p1 1.09 s, p99 3.47 s, mean 2.24 s).

![Timestamp bias](../figures/fig5_timestamp_bias.png)

*Figure 5. Distribution of block timestamp − execution timestamp, in seconds (4,479 trades).*

**Consequence.**

- Read at block time, the market seems to react to Binance late, and the lead-lag strategy looks like a strong edge:
  +7.6 c/share at L = 2 s, with a BH-adjusted p of 0.002 by day and by market.
- Every series built from trades is therefore re-aligned by −2 s (the median, rounded to the 1-second grid).
- After re-alignment, the response of the market price peaks at a lag of 1 s, and 9 % of it arrives after 2 s.

![Lead-lag](../figures/fig2_leadlag.png)

*Figure 2. Response of the market price change Δp_t to the change of the Binance fair value ΔP_{t−lag}. These are the
coefficients of a single distributed-lag regression on the test period, with a 95 % CI from the bootstrap by UTC day.
Grey: block time. Blue: −2 s (reference). Orange: −3 s.*

**Lead-lag strategy (H1) by re-alignment and latency.** Test period; mean P&L in c/share. The latency grid is a
robustness check and is not part of the BH family. Source: [results/latency.md](../results/latency.md).

#### 95 % CI, bootstrap by UTC day (reference)

| Re-alignment | L = 0 s | L = 1 s | L = 2 s | L = 3 s | L = 5 s |
|---|---|---|---|---|---|
| none (block time) | +16.3 [+14.7, +17.8] | +13.0 [+11.4, +14.5] | +7.6 [+5.9, +9.1] | +1.1 [-0.5, +2.5] | -1.7 [-3.2, -0.3] |
| −2 s (reference) | +12.6 [+11.4, +13.8] | +3.8 [+2.6, +4.9] | +1.0 [-0.1, +2.1] | +0.5 [-0.6, +1.7] | +0.6 [-0.6, +1.8] |
| −3 s | +3.1 [+1.8, +4.5] | -0.6 [-1.8, +0.7] | -1.1 [-2.2, +0.1] | -1.3 [-2.6, +0.0] | -1.5 [-3.0, -0.0] |

#### 95 % CI, bootstrap by market

| Re-alignment | L = 0 s | L = 1 s | L = 2 s | L = 3 s | L = 5 s |
|---|---|---|---|---|---|
| none (block time) | +16.3 [+14.8, +17.9] | +13.0 [+11.5, +14.6] | +7.6 [+6.1, +9.1] | +1.1 [-0.3, +2.5] | -1.7 [-3.1, -0.3] |
| −2 s (reference) | +12.6 [+11.2, +13.9] | +3.8 [+2.6, +5.1] | +1.0 [-0.2, +2.2] | +0.5 [-0.7, +1.8] | +0.6 [-0.8, +1.9] |
| −3 s | +3.1 [+1.6, +4.6] | -0.6 [-2.0, +0.8] | -1.1 [-2.4, +0.3] | -1.3 [-2.8, +0.1] | -1.5 [-3.1, +0.0] |

**Reading.**

- At 1 s latency the result depends on the re-alignment: +3.8 c with −2 s, −0.6 c with −3 s. The true offset of
  2.25 s lies between the two, so this study cannot tell whether a 1-second taker earns anything.
- From 2 s of latency, neither re-alignment gives a CI above zero.

## A11 and the timestamp shift

**Why a check is needed.** After the −2 s shift, the block − execution gap (p1 1.09 s) places some trades up to 0.91 s
before their real execution. Their price can then appear to anticipate a Binance move that had already happened, which
would create a mechanical lead in A11.

**Checks.** [analysis/a11_robustness.py](../analysis/a11_robustness.py) recomputes A11 on the same 7,766 test markets
with three re-alignments and two Binance horizons. The statistic is the mean per-market correlation, with a 95 % CI by
UTC day.

| Re-alignment of the market series | Binance from t to t + 10 s | Binance from t + 2 to t + 12 s |
|---|---|---|
| none (block time: no trade placed before its execution) | 0.024 [0.015, 0.035] | 0.027 [0.018, 0.036] |
| −2 s (reference) | 0.082 [0.076, 0.088] | 0.067 [0.060, 0.073] |
| −3 s (trades up to 1.91 s early) | 0.137 [0.131, 0.143] | 0.092 [0.086, 0.097] |

**Reading.**

- The correlation stays positive in every cell, including the two settings that rule out the mechanism:
  - block time, where no trade is placed before its execution;
  - a Binance horizon that starts at t + 2 s, later than any early placement under −2 s or −3 s.
- The price residual therefore carries information about Binance moves at least 2 s ahead.
- Its size is not identified. It grows as trades are placed earlier, so part of the reference value of 0.082 is
  mechanical, and the lower bound is a correlation of 0.024 to 0.027.
- It is a statistical test on the price, not a P&L test.

## Reproducibility

**Seeds.**

- NumPy generators are seeded: `20261018` in `analysis/run.py`, `7` in `analysis/deep.py` (reused by `edges*`), and
  `20261026` in `analysis/blockboot.py` (block bootstraps).
- scikit-learn models use `random_state=0`, and PyTorch uses `torch.manual_seed(0)`.
- `reproduce.sh` sets `PYTHONHASHSEED=0`.

**Layout.**

```
pipelines/   download and build the parquet tables (market metadata, trades, Binance, 1-second panels)
analysis/    run.py (H1-H4) · deep*.py (A9-A11, ML/DL) · a11_robustness.py · edges*.py (A1-A8)
             blockboot.py (cluster bootstraps) · timestamp_lag.py · summary.py (tables) · figures.py
scripts/     check_figures.py (every number of README.md and docs/METHODS.md is in results/key_figures.md)
docs/        METHODS.md (this file) · data_notes.md (facts checked by hand on the venue and on-chain)
data_sample/ timestamp_lag_pairs.csv (execution vs block time, 4,479 trades)
results/     JSON outputs and Markdown tables
figures/     PNG figures
```
