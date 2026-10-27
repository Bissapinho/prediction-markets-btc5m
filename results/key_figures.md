# Key figures

## Data

| Figure | Value | Source |
|---|---|---|
| Markets analysed | 17,559 | data/clean/markets + filters of analysis/data.py → results/summary.json |
| Five-minute windows in the period | 17,856 | results/summary.json |
| Listed by the market API | 17,855 | results/summary.json |
| Markets with at least one trade in [0, 300 s) | 17,560 | results/summary.json |
| Days analysed | 62 | results/summary.json |
| Period (market start, UTC) | 2026-08-17 00:00 → 2026-10-17 23:50 | results/summary.json |
| Train / test markets (split 2026-09-21) | 9,784 / 7,775 | results/results_delta2.json (meta) |
| Test days (UTC) | 27 | results/results_delta2.json (S1 test, day bootstrap) |
| Trades on the analysed markets | 28,594,392 | data/clean/trades → results/summary.json |
| of which inside the 5-min window | 26,956,412 | results/summary.json |
| Traded notional | 435.7 M USDC | results/summary.json |
| Binance BTCUSDT aggregated trades | 58,105,834 | data/clean/btc_aggtrades → results/summary.json |
| Per-second rows (market × second) | 5,267,700 | data/features/market_1s → results/summary.json |
| Per-second rows (Binance 1 s) | 5,358,300 | data/features/btc_1s → results/summary.json |
| Per-second rows, total | 10,626,000 (10.6 M) | results/summary.json |

## Fair-value model

| Figure | Value | Source |
|---|---|---|
| Fair-value model, fitted on train | k = 1.15, τ = 4e-05 | results/results_delta2.json (model) |
| Model reproduces the outcome at t = 299 s (test) | 98.0% | results/results_delta2.json (model) |
| Test log-loss, market vs model, 10-90 s | 0.592 vs 0.603 | results/results_delta2.json (H2.scores) |
| Test log-loss, market vs model, 100-230 s | 0.382 vs 0.403 | results/results_delta2.json (H2.scores) |
| Test log-loss, market vs model, 240-290 s | 0.077 vs 0.101 | results/results_delta2.json (H2.scores) |

## Hypothesis tests (test period)

| Figure | Value | Source |
|---|---|---|
| Hypotheses in the public table | 15 | results/hypotheses.md |
| Tests in BH families (P, A, I) | 14 | results/hypotheses.md |
| Ex-post diagnostics / artefacts | 1 (A3) / 1 (A8) | results/hypotheses.md |
| Strategies with a P&L test in a BH family | 10 | results/hypotheses.md |
| P&L tests significant after BH (day) with positive P&L | 0 | results/hypotheses.md |
| of which taker strategies | 0 | results/hypotheses.md |
| Detectable edge (80 % power), H1-H4, bootstrap by day | 1.3 to 1.5 c/share | results/summary.json (mde) |
| Detectable edge (80 % power), H1-H4, bootstrap by market | 1.4 to 1.7 c/share | results/summary.json (mde) |
| H1 (day bootstrap) | +1.0 c [-0.1, +2.1], p adj. 0.086 | results/hypotheses.md |
| H1 (market bootstrap) | +1.0 c [-0.1, +2.2], p adj. 0.111 | results/hypotheses.md |
| H2 (day bootstrap) | -2.7 c [-3.9, -1.6], p adj. 1.000 | results/hypotheses.md |
| H2 (market bootstrap) | -2.7 c [-3.8, -1.7], p adj. 1.000 | results/hypotheses.md |
| H2-info (day bootstrap) | 0.124 [0.031, 0.215], p adj. 0.020 | results/hypotheses.md |
| H2-info (market bootstrap) | 0.124 [0.055, 0.195], p adj. 0.010 | results/hypotheses.md |
| H3 (day bootstrap) | -2.2 c [-3.3, -1.2], p adj. 1.000 | results/hypotheses.md |
| H3 (market bootstrap) | -2.2 c [-3.3, -1.1], p adj. 1.000 | results/hypotheses.md |
| H4 (day bootstrap) | -0.4 c [-1.6, +0.7], p adj. 1.000 | results/hypotheses.md |
| H4 (market bootstrap) | -0.4 c [-1.9, +0.9], p adj. 1.000 | results/hypotheses.md |
| A1 (day bootstrap) | -6.0 c [-6.6, -5.4], p adj. 1.000 | results/hypotheses.md |
| A1 (market bootstrap) | -6.0 c [-6.5, -5.4], p adj. 1.000 | results/hypotheses.md |
| A2 (day bootstrap) | -1.0 c [-1.1, -0.8], p adj. 1.000 | results/hypotheses.md |
| A2 (market bootstrap) | -1.0 c [-1.1, -0.8], p adj. 1.000 | results/hypotheses.md |
| A3 (day bootstrap) | +1.9 c [+1.7, +2.2], p < 0.001 | results/hypotheses.md |
| A3 (market bootstrap) | +1.9 c [+1.7, +2.2], p < 0.001 | results/hypotheses.md |
| A4 (day bootstrap) | -1.4 c [-2.5, -0.4], p adj. 1.000 | results/hypotheses.md |
| A4 (market bootstrap) | -1.4 c [-2.5, -0.4], p adj. 1.000 | results/hypotheses.md |
| A5 (day bootstrap) | -3.6 c [-4.7, -2.5], p adj. 1.000 | results/hypotheses.md |
| A5 (market bootstrap) | -3.6 c [-5.0, -2.3], p adj. 1.000 | results/hypotheses.md |
| A6 (day bootstrap) | -3.0 c [-5.8, -0.3], p adj. 1.000 | results/hypotheses.md |
| A6 (market bootstrap) | -3.0 c [-5.2, -0.3], p adj. 1.000 | results/hypotheses.md |
| A7 (day bootstrap) | -0.8 c [-13.1, +5.7], p adj. 1.000 | results/hypotheses.md |
| A7 (market bootstrap) | -0.8 c [-10.7, +5.8], p adj. 1.000 | results/hypotheses.md |
| A8 share of seconds with ask_up + ask_dn < 1 (test) | 52% of 919,083 s | results/edges_E4.json |
| A9 (day bootstrap) | 1.223 [1.158, 1.290], p adj. < 0.001 | results/hypotheses.md |
| A9 (market bootstrap) | 1.223 [1.150, 1.304], p adj. < 0.001 | results/hypotheses.md |
| A10 (day bootstrap) | 0.0008 [0.0001, 0.0016], p adj. 0.014 | results/hypotheses.md |
| A10 (market bootstrap) | 0.0008 [0.0001, 0.0016], p adj. 0.018 | results/hypotheses.md |
| A11 (day bootstrap) | 0.082 [0.076, 0.088], p adj. < 0.001 | results/hypotheses.md |
| A11 (market bootstrap) | 0.082 [0.079, 0.086], p adj. < 0.001 | results/hypotheses.md |

## Machine learning

| Figure | Value | Source |
|---|---|---|
| Models compared with the market price | 8 | results/ml.md |
| ML test predictions (market × snapshot) | 69,894 | results/deep_D.json |
| Best ML log-loss gain vs market (best of 8 models) | +0.0019 nats (Logistic regression, standalone) | results/deep_D.json → results/ml.md |
| Best ML gain as a share of the market log-loss | 0.5% | results/ml.md |
| Recalibrated market price, gain | +0.0005 nats | results/ml.md |
| GRU log-loss gain vs market (day CI) | +0.0002 nats [-0.0014, +0.0017] | results/ml.md |
| ML models significant after BH (day / market) | 3 / 3 | results/ml.md |

## Timestamp bias and lead-lag

| Figure | Value | Source |
|---|---|---|
| Timestamp bias, block − execution (median) | 2.25 s (p1 1.09 s, p99 3.47 s, mean 2.24 s) | results/timestamp_lag.json |
| Trades in the timestamp measurement | 4,479 | data_sample/timestamp_lag_pairs.csv → results/timestamp_lag.json |
| Earliest placement before execution after −2 s / −3 s | 0.91 s / 1.91 s (p1 of the gap) | results/timestamp_lag.json |
| Lead-lag regression: peak lag after −2 s | 1 s | results/results_delta2.json (H1) |
| Lead-lag regression: share of the response after 2 s | 9% | results/results_delta2.json (H1) |
| Lead-lag (H1) at block time, L = 2 s | +7.6 c/share [+5.9, +9.1], p adj. (day) 0.002, p adj. (market) 0.002 | results/results_delta0.json |
| Lead-lag (H1), re-alignment none, L = 0 s | +16.3 c/share [+14.7, +17.8] (day); [+14.8, +17.9] (market) | results/results_delta0.json → results/latency.md |
| Lead-lag (H1), re-alignment none, L = 1 s | +13.0 c/share [+11.4, +14.5] (day); [+11.5, +14.6] (market) | results/results_delta0.json → results/latency.md |
| Lead-lag (H1), re-alignment none, L = 2 s | +7.6 c/share [+5.9, +9.1] (day); [+6.1, +9.1] (market) | results/results_delta0.json → results/latency.md |
| Lead-lag (H1), re-alignment none, L = 3 s | +1.1 c/share [-0.5, +2.5] (day); [-0.3, +2.5] (market) | results/results_delta0.json → results/latency.md |
| Lead-lag (H1), re-alignment none, L = 5 s | -1.7 c/share [-3.2, -0.3] (day); [-3.1, -0.3] (market) | results/results_delta0.json → results/latency.md |
| Lead-lag (H1), re-alignment −2 s, L = 0 s | +12.6 c/share [+11.4, +13.8] (day); [+11.2, +13.9] (market) | results/results_delta2.json → results/latency.md |
| Lead-lag (H1), re-alignment −2 s, L = 1 s | +3.8 c/share [+2.6, +4.9] (day); [+2.6, +5.1] (market) | results/results_delta2.json → results/latency.md |
| Lead-lag (H1), re-alignment −2 s, L = 2 s | +1.0 c/share [-0.1, +2.1] (day); [-0.2, +2.2] (market) | results/results_delta2.json → results/latency.md |
| Lead-lag (H1), re-alignment −2 s, L = 3 s | +0.5 c/share [-0.6, +1.7] (day); [-0.7, +1.8] (market) | results/results_delta2.json → results/latency.md |
| Lead-lag (H1), re-alignment −2 s, L = 5 s | +0.6 c/share [-0.6, +1.8] (day); [-0.8, +1.9] (market) | results/results_delta2.json → results/latency.md |
| Lead-lag (H1), re-alignment −3 s, L = 0 s | +3.1 c/share [+1.8, +4.5] (day); [+1.6, +4.6] (market) | results/results_delta3.json → results/latency.md |
| Lead-lag (H1), re-alignment −3 s, L = 1 s | -0.6 c/share [-1.8, +0.7] (day); [-2.0, +0.8] (market) | results/results_delta3.json → results/latency.md |
| Lead-lag (H1), re-alignment −3 s, L = 2 s | -1.1 c/share [-2.2, +0.1] (day); [-2.4, +0.3] (market) | results/results_delta3.json → results/latency.md |
| Lead-lag (H1), re-alignment −3 s, L = 3 s | -1.3 c/share [-2.6, +0.0] (day); [-2.8, +0.1] (market) | results/results_delta3.json → results/latency.md |
| Lead-lag (H1), re-alignment −3 s, L = 5 s | -1.5 c/share [-3.0, -0.0] (day); [-3.1, +0.0] (market) | results/results_delta3.json → results/latency.md |

## A11 robustness

| Figure | Value | Source |
|---|---|---|
| A11, re-alignment −2 s, Binance horizon t + 0 to t + 10 s | 0.082, day CI [0.076, 0.088], market CI [0.078, 0.086], n = 7,766 | results/a11_robustness.json |
| A11, re-alignment −2 s, Binance horizon t + 2 to t + 12 s | 0.067, day CI [0.060, 0.073], market CI [0.063, 0.070], n = 7,766 | results/a11_robustness.json |
| A11, re-alignment −3 s, Binance horizon t + 0 to t + 10 s | 0.137, day CI [0.131, 0.143], market CI [0.133, 0.141], n = 7,766 | results/a11_robustness.json |
| A11, re-alignment −3 s, Binance horizon t + 2 to t + 12 s | 0.092, day CI [0.086, 0.097], market CI [0.088, 0.096], n = 7,766 | results/a11_robustness.json |
| A11, re-alignment none (block time), Binance horizon t + 0 to t + 10 s | 0.024, day CI [0.015, 0.035], market CI [0.021, 0.028], n = 7,766 | results/a11_robustness.json |
| A11, re-alignment none (block time), Binance horizon t + 2 to t + 12 s | 0.027, day CI [0.018, 0.036], market CI [0.023, 0.031], n = 7,766 | results/a11_robustness.json |

## Protocol constants

| Figure | Value | Source |
|---|---|---|
| Five-minute window, markets per UTC day | 300 s, 288 markets | config.yaml (duration), analysis/data.py (T) |
| Resolution TWAP / pre-market volatility window | 60-second TWAP; 30-minute realised volatility (σ), 1,800 s | analysis/run.py (L_TWAP, sigma_pre) |
| Bootstrap draws: P&L by market (H1-H4) / blocks (day, hour) | 2,000 / 2,000 | analysis/run.py (B), analysis/blockboot.py (B) |
| Bootstrap draws: H1 regression and H2 logit / A9 logit | 500 / 300 | analysis/run.py, analysis/deep_extra.py |
| Benjamini-Hochberg level | q = 0.05 | analysis/summary.py (Q) |
| Minimum trades on train for a strategy threshold | 200 | analysis/run.py (main) |
| Validation days for early stopping and stacking | last 5 days of train | analysis/deep_cd.py, analysis/deep_stack.py |
| Taker fee | 0.07 · p · (1 − p) per share: 1.75 c at 50 c, 0.33 c at 95 c | analysis/run.py (fee), docs/data_notes.md |
| Execution latency L | 2 s (H1-H4), 1 s (A family); grid 0, 1, 2, 3, 5 s | analysis/run.py, analysis/edges.py |
| Ask validity / bin of the timestamp histogram | 3 s / 0.25 s | analysis/data.py (ASK_MAX_AGE), analysis/timestamp_lag.py |
| Selected thresholds on train | H1 z0 = 4 σ in 3 s; H2 θ = 5 c; H3 jump 8 c in 5 s; H4 t = 260 s, ask 0.80-0.98 | results/results_delta2.json (strategies) |
| A1 / A7 thresholds | buy ≤ 30 c, sell ≥ 33 c / ask ≤ 99 c, model > 99.5 % | analysis/edges.py |
| A2 grid / A4 test AUC | 216 cells / 0.59 | results/edges_E2.json, results/edges_E8.json |
| ML features / snapshots | 28 features; t = 30, 60, ..., 270 s | analysis/deep_cd.py (_features, SNAPS) |
| ML models | logistic regression C = 1; MLP (64, 32); GRU hidden size 32 on the last 30 s of 6 channels | analysis/deep_cd.py |
| Calibration snapshots | every 10 s | analysis/run.py (H2) |
| On-chain check of the trade history | 9,018 / 9,018 trades on 4 markets | docs/data_notes.md |
| Venue outage (no trade, markets dropped by the trade filter) | 2026-09-19 | docs/data_notes.md |
| A11 Binance horizons | t to t + 10 s; t + 2 to t + 12 s | analysis/a11_robustness.py (HORIZONS) |

# Full tables (copies of results/hypotheses.md, ml.md, latency.md)

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

### ML and DL models

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

### Lead-lag strategy by re-alignment and latency

### 95 % CI, bootstrap by UTC day (reference)

| Re-alignment | L = 0 s | L = 1 s | L = 2 s | L = 3 s | L = 5 s |
|---|---|---|---|---|---|
| none (block time) | +16.3 [+14.7, +17.8] | +13.0 [+11.4, +14.5] | +7.6 [+5.9, +9.1] | +1.1 [-0.5, +2.5] | -1.7 [-3.2, -0.3] |
| −2 s (reference) | +12.6 [+11.4, +13.8] | +3.8 [+2.6, +4.9] | +1.0 [-0.1, +2.1] | +0.5 [-0.6, +1.7] | +0.6 [-0.6, +1.8] |
| −3 s | +3.1 [+1.8, +4.5] | -0.6 [-1.8, +0.7] | -1.1 [-2.2, +0.1] | -1.3 [-2.6, +0.0] | -1.5 [-3.0, -0.0] |

### 95 % CI, bootstrap by market

| Re-alignment | L = 0 s | L = 1 s | L = 2 s | L = 3 s | L = 5 s |
|---|---|---|---|---|---|
| none (block time) | +16.3 [+14.8, +17.9] | +13.0 [+11.5, +14.6] | +7.6 [+6.1, +9.1] | +1.1 [-0.3, +2.5] | -1.7 [-3.1, -0.3] |
| −2 s (reference) | +12.6 [+11.2, +13.9] | +3.8 [+2.6, +5.1] | +1.0 [-0.2, +2.2] | +0.5 [-0.7, +1.8] | +0.6 [-0.8, +1.9] |
| −3 s | +3.1 [+1.6, +4.6] | -0.6 [-2.0, +0.8] | -1.1 [-2.4, +0.3] | -1.3 [-2.8, +0.1] | -1.5 [-3.1, +0.0] |
