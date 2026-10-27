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
