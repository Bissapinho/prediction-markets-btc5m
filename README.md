# Price formation in 5-minute BTC prediction markets

**TL;DR**

- **Efficient at 2 s of latency or more.** On 7,775 held-out markets (27 test days), none of the 10 tested taker or
  limit-order strategies has a positive P&L after Benjamini–Hochberg correction. Confidence intervals come from a
  bootstrap that resamples UTC days.
- **A 2.25 s timestamp bias creates a false edge.** The trade history is stamped with the block time, a median 2.25 s
  after execution. Read at block time, a lead-lag strategy appears to earn +7.6 c/share (BH-adjusted p 0.002).
  After re-alignment, the market absorbs a Binance move within about 1 s.
- **Machine learning adds at most 0.5 % of the market's log-loss.** The best of 8 models gains 0.0019 nats per
  prediction over the market price.

![Latency](figures/fig3_latency.png)

*Mean P&L of the lead-lag strategy (H1, c/share, fees included) by execution latency L, for three re-alignments of the
trade timestamps. 95 % CI, bootstrap by UTC day.*

## Question

The venue lists one binary "Bitcoin Up or Down" market per 5-minute window. Up pays 1 $ if the 60-second TWAP of
Chainlink BTC/USD at the end of the window is at least the TWAP at its start.

The question is whether the price absorbs public information fast and accurately enough to leave no edge to a trader
who takes liquidity. Two kinds of public information are considered: the Binance price, and the market's own price
path.

## Data

| | |
|---|---|
| Period | Markets starting 2026-08-17 00:00 → 2026-10-17 23:50 UTC (62 days) |
| Markets | **17,559** analysed: 9,784 for exploration (before 2026-09-21), 7,775 for the test |
| Trades | **28,594,392** taker trades, 435.7 M USDC notional, from the venue's public trade-history API, checked on-chain |
| Reference price | Binance BTCUSDT: 58,105,834 aggregated trades, resampled to 1-second bars |

Raw data are not in the repository. `bash reproduce.sh --download` rebuilds them from the public APIs.
`data_sample/` holds the 4,479 (execution time, block time) pairs of the timestamp measurement, which cannot be
downloaded again. Sources, filters and the fair-value model are described in [docs/METHODS.md](docs/METHODS.md), and
facts checked by hand are in [docs/data_notes.md](docs/data_notes.md).

## Main results

**Method in short.**

- Every free parameter is chosen on the exploration period.
- The test set was used to detect two data artefacts (stale ask, timestamp bias). After these corrections, each
  configuration was evaluated once on the test period.
- CIs and p-values come from a bootstrap by UTC day (27 test days), because markets of the same day share a volatility
  regime. The bootstrap by market and the bootstrap by UTC hour are reported alongside.
- Benjamini–Hochberg is applied within each family:
  - **P**, the protocol specified before the test evaluation;
  - **A**, price-path strategies;
  - **I**, information beyond the price.

Details are in [docs/METHODS.md](docs/METHODS.md).

**Hypotheses (test period).** P&L per share (1 share pays 1 $ if it wins), fees included. 95 % CI and adjusted p by UTC
day. The full table, with the market and hour bootstraps, is in [docs/METHODS.md](docs/METHODS.md#full-hypothesis-table).

| ID | Hypothesis | Execution | Estimate | 95 % CI (day) | p adj. (day) | Verdict |
|---|---|---|---|---|---|---|
| H1 | Lead-lag: buy after a ≥ 4 σ Binance move in 3 s that the market has not followed | taker, 2 s | +1.0 c | [−0.1, +2.1] | 0.086 | Not significant |
| H2 | Buy the side priced below the Binance TWAP fair-value model | taker, 2 s | −2.7 c | [−3.9, −1.6] | 1.000 | Rejected |
| H3 | Fade price jumps that Binance does not justify | taker, 2 s | −2.2 c | [−3.3, −1.2] | 1.000 | Rejected |
| H4 | Late favourites (ask 0.80–0.98) | taker, 2 s | −0.4 c | [−1.6, +0.7] | 1.000 | Rejected |
| A1–A7 | Six price-path strategies (dip-buying, ML-selected dips, momentum, TWAP anchor, sniping); A3 excluded | taker 1 s or limit | −6.0 to −0.8 c | see full table | 1.000 | Rejected |
| H2-info | Model − market predicts the outcome (logit coefficient) | – | 0.124 | [0.031, 0.215] | 0.020 | Information detected |
| A9 | Under-reaction at the open (slope on logit price, H0: 1) | – | 1.223 | [1.158, 1.290] | < 0.001 | Information detected |
| A10 | ML forecast stacked on the market price (log-loss gain, nats) | – | 0.0008 | [0.0001, 0.0016] | 0.014 | Information detected |
| A11 | Price residual leads the next 10 s Binance move (correlation) | – | 0.082 | [0.076, 0.088] | < 0.001 | Information detected |

The information tests (H2-info, A9–A11) are statistical tests, not P&L tests. None of the tested strategies turns
these effects into a P&L that pays the spread and fees. Two rows are not shown above:

- A3 is positive (+1.9 c), but it uses a market cluster defined from the whole price path, so it is reported as an
  ex-post diagnostic outside any family.
- A8 (pair arbitrage) is an artefact of the price data and is not tested.

**Timestamp bias.**

- A one-day live capture matched 4,479 trades to the trade history. Block − execution has a median of 2.25 s
  (p1 1.09 s, p99 3.47 s).
- All market series are re-aligned by −2 s. The lead-lag P&L at 1 s latency then depends on the re-alignment:
  +3.8 c at −2 s, −0.6 c at −3 s.
- From 2 s of latency, neither re-alignment gives a CI above zero. Whether a sub-second taker earns anything cannot be
  decided with block timestamps.

**A11, the most robust signal.** Some trades end up placed before their execution after the −2 s shift, which could
create a mechanical lead. To rule this out, A11 was recomputed at block time and with a Binance horizon starting 2 s
later. The correlation stays positive in both cases:

- 0.024 [0.015, 0.035] at block time;
- 0.067 [0.060, 0.073] with the horizon from t + 2 to t + 12 s.

It grows to 0.137 at −3 s, so part of the reference value of 0.082 is mechanical. The residual carries information
about Binance moves at least 2 s ahead, but its size is not identified. See
[docs/METHODS.md](docs/METHODS.md#a11-and-the-timestamp-shift).

**Machine learning.**

- Logistic regression, gradient boosting, an MLP and a GRU forecast the outcome from 28 features known at t.
- The best gain over the market price is +0.0019 nats per prediction (logistic regression, best of 8 models), 0.5 % of
  the market's log-loss.
- A quarter of it (+0.0005) comes from simply recalibrating the market price.
- Boosting and the MLP do worse than the market out of sample.

## Limitations

- **Short period.** 62 days, of which 27 are test days, all under the TWAP resolution rule. With the bootstrap by day,
  edges below 1.3 to 1.5 c/share are not detectable with 80 % power.
- **One asset and one venue.** Only BTC 5-minute markets.
- **Timestamp re-alignment.** Every comparison with Binance depends on the −2 s shift, estimated from a single day of
  live capture. The result at 1 s of latency changes sign between −2 s and −3 s.
- **No order book.** The ask is the last taker buy of the last 3 s, floored by the mirror book. This is valid for a
  small size only, and the queue position of limit orders is not modelled.
- **No regime analysis.** There is no split by volatility regime and no test on the earlier spot-resolution period.
- **Model selection.** The ML gain is the best of 8 models evaluated on the test period.

## Reproduce

Python 3.13, CPU only.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m pytest -q
```

```bash
bash reproduce.sh --download   # download (~30-60 min, a few GB in data/), then all analyses, tables and figures
bash reproduce.sh              # analyses, tables and figures only, from data/ already built
```

`reproduce.sh` writes `results/*.json`, the tables `results/hypotheses.md`, `results/ml.md` and `results/latency.md`,
and `results/key_figures.md`, which lists every number of this README and of `docs/METHODS.md` with its source. Its last
step, `scripts/check_figures.py`, fails if a number in either document is missing from `key_figures.md`. The analyses
take about 10 minutes on a desktop CPU; each step's wall time is written to `results/runtime.log`. Seeds and layout
are in [docs/METHODS.md](docs/METHODS.md#reproducibility).

## License

Code released under the [MIT License](LICENSE).
