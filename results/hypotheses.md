## Tests (Benjamini-Hochberg within each family)

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

## Ex-post diagnostics (not in any BH family; p adj. is not computed)

| ID | Hypothesis | Execution | Statistic (test) | Estimate | n | 95 % CI (day) | p (day) | p adj. (day) | 95 % CI (market) | p adj. (market) | p adj. (hour) | Family | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | Dip-buy restricted to the 'hesitant' market cluster (cluster defined from the whole price path) | limit order | mean P&L per share, fees included | +1.9 c | 2,236 | [+1.7, +2.2] | < 0.001 | – | [+1.7, +2.2] | – | – | ex-post diagnostic | Ex-post diagnostic: the cluster uses the whole price path (look-ahead), so this is not a tradable strategy; not in any BH family |
