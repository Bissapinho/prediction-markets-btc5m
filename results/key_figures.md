| Figure | Value | Source |
|---|---|---|
| Markets analysed | 17,559 | data/clean/markets + filters of analysis/data.py → results/summary.json |
| Period (market start, UTC) | 2026-08-17 00:00 → 2026-10-17 23:50 | results/summary.json |
| Train / test markets (split 2026-09-21) | 9,784 / 7,775 | results/results_delta2.json (meta) |
| Prediction Market trades (analysed markets) | 28,594,392 | data/clean/trades → results/summary.json |
| of which inside the 5-min window | 26,956,412 | results/summary.json |
| Traded notional | 435.7 M USDC | results/summary.json |
| Binance BTCUSDT aggregated trades | 58,105,834 | data/clean/btc_aggtrades → results/summary.json |
| Per-second rows (market × second) | 5,267,700 | data/features/market_1s → results/summary.json |
| Per-second rows (Binance 1 s) | 5,358,300 | data/features/btc_1s → results/summary.json |
| Per-second rows, total | 10,626,000 | results/summary.json |
| Hypotheses / tests in the public table | 15 hypotheses, 15 tests + 1 artefact (A8) | results/hypotheses.md |
| Hypotheses significant after BH with positive taker P&L | 0 | results/hypotheses.md |
| Best ML log-loss gain vs market | +0.0019 nats (Logistic regression, standalone) | results/deep_D.json → results/ml.md |
| GRU log-loss gain vs market | +0.0002 nats [-0.0010, +0.0013] | results/deep_D.json → results/ml.md |
| Timestamp bias, block − execution (median) | 2.25 s (p1 1.09 s, p99 3.47 s) | results/timestamp_lag.json |
| Trades in the timestamp measurement | 4,479 | data_sample/timestamp_lag_pairs.csv |
| Lead-lag (H1), L = 1 s, −2 s re-alignment | +3.8 c/share [+2.6, +5.1] | results/results_delta2.json |
| Lead-lag (H1), L = 1 s, −3 s re-alignment | -0.6 c/share [-2.0, +0.8] | results/results_delta3.json |
| Lead-lag (H1), L = 2 s, −2 s re-alignment | +1.0 c/share [-0.2, +2.2] | results/results_delta2.json |
| Detectable edge (80 % power), H1-H4 | 1.4 to 1.7 c/share | results/summary.json |
