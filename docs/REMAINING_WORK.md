# Remaining work: block-bootstrap and README revision

Status on 2026-10-27. The changes described under "Done locally" exist in the working copy only. They are **not
committed and not pushed**; this file is the only file pushed.

## Done locally (uncommitted)

### Block A: method

- `analysis/blockboot.py` (new) contains a cluster/block bootstrap with three resampling units: market, UTC day
  (the reference) and UTC hour (sensitivity). It has its own RNG, so the market-bootstrap draws are unchanged.
- The day and hour bootstraps are wired into every test:
  - `analysis/run.py`: H1–H4 P&L, the latency grid, the H2-info logit, the H1 lead-lag regression, and `bh_day` / `bh_hour`.
  - `analysis/edges.py` and `analysis/edges_ml.py`: A1–A7 P&L. `mboot(..., start_s=x.start_s)`.
  - `analysis/deep.py`: A11 per-market correlation.
  - `analysis/deep_extra.py`: A9 logit (also stores `n`, which fixes the missing n for A9).
  - `analysis/deep_cd.py` and `analysis/deep_stack.py`: ML log-loss gains.
- `analysis/a11_robustness.py` (new) tests A11 with a re-alignment of 0, −2 and −3 s, and a Binance horizon of
  [t, t+10] or [t+2, t+12]. Output: `results/a11_robustness.json`.
- `analysis/summary.py` is rewritten:
  - Verdicts and BH use the day bootstrap; the market and hour bootstraps are extra columns.
  - A3 is moved out of family A into an "ex-post diagnostics" table.
  - The MDE is computed from the day CI.
  - Key figures are expanded, and "best of 8 models" is added.
- `analysis/figures.py`: figures 2, 3 and 4 use the day CIs.
- `analysis/timestamp_lag.py`: it now re-extracts only with `--extract`. By default it reads `data_sample/`, which
  avoids a network call during `reproduce.sh`.
- `reproduce.sh` gains the `analysis.a11_robustness` step.
- `README.md` (still in the old structure):
  - "One pass" is replaced by the exact use of the test set, and "pre-registered" by "specified before the test
    evaluation".
  - The tables show day CIs with the market CIs alongside, and A3 is in its own table.
  - A paragraph on A11 robustness is added.
- A full `bash reproduce.sh` run (601 s, no error) regenerated `results/` and `figures/` with this code.

### Block B: form (partial)

- `tests/test_blockboot.py` (new, 5 tests, passing) checks a known case: with perfect within-day dependence, the
  day-bootstrap SD equals sd(day values)/√27.
- `scripts/check_figures.py` (new, never run yet) checks that every number in README.md and docs/METHODS.md appears
  in results/key_figures.md.
- Comments, docstrings and log messages are translated to English in `pipelines/`, `tests/` and `config.yaml`.

## Results so far (full run, test period)

**No verdict changed** between the market bootstrap and the day bootstrap.

| Test | Estimate | 95 % CI by market → by day | p adj. by market → by day |
|---|---|---|---|
| H1 lead-lag, L = 2 s | +1.0 c | [−0.1, +2.2] → [−0.1, +2.1] | 0.111 → 0.086 (not significant) |
| H2-info | c = 0.124 | [0.055, 0.195] → [0.031, 0.215] | 0.010 → 0.020 |
| A9 slope | 1.223 | [1.150, 1.304] → [1.158, 1.290] | < 0.001 → < 0.001 |
| A10 stacked ML | +0.0008 nats | [+0.0001, +0.0016] → [+0.0001, +0.0016] | 0.018 → 0.014 |
| A11 correlation | 0.082 | [0.079, 0.086] → [0.076, 0.088] | < 0.001 → < 0.001 |
| Best ML (best of 8) | +0.0019 nats | [+0.0012, +0.0027] → [+0.0011, +0.0027] | < 0.001 → < 0.001 |
| A3 (ex-post, out of BH) | +1.9 c | [+1.7, +2.2] → [+1.7, +2.2] | was 0.007 in family A; now not tested |
| A4 | −1.4 c | [−2.5, −0.4] → [−2.5, −0.4] | 1.000 → 1.000 |
| MDE H1–H4 (80 % power) | | 1.4–1.7 c → 1.3–1.5 c | |

There is one fragile point in the hour-bootstrap sensitivity analysis. The stacked logistic model (ML table) has
p adj. 0.039 by day but 0.075 by hour.

**A11 robustness** (mean per-market correlation, 95 % CI by day, 7,766 test markets):

| Re-alignment | Binance t → t+10 s | Binance t+2 → t+12 s |
|---|---|---|
| none (block time) | 0.024 [0.015, 0.035] | 0.027 [0.018, 0.036] |
| −2 s (reference) | 0.082 [0.076, 0.088] | 0.067 [0.060, 0.073] |
| −3 s | 0.137 [0.131, 0.143] | 0.092 [0.086, 0.097] |

A11 resists the three requested checks (−3 s, horizon starting at t+2 s, day bootstrap). It also resists block time,
where no trade can be placed before its execution. Its size is not identified: it grows as trades are placed earlier,
so part of the 0.082 is mechanical. The lower bound is about 0.025.

## Left to do

### Finish block A, then commit it

1. In `README.md`, replace the placeholders `@A4CI@` with `[−2.5, −0.4]` and `@A3CI@` with `[+1.7, +2.2]`
   (values from `results/summary.json`).
2. Run `pytest`, then check that README numbers match `results/hypotheses.md`, `ml.md`, `latency.md` and
   `key_figures.md`.
3. Commit block A without the block B files. The pipelines/tests/config translations are already on disk and must
   be left out (stage file by file).

### Block B

4. **Translate the `analysis/` files.** Replacement lists are ready (scratch files `p_an1.py` to `p_an4.py`). Some
   JSON labels are renamed at the same time; update `summary.py` to match:
   - `logistique` → `logistic`
   - `marché recalibré` → `market recalibrated`
   - `GRU (séquence 30 s)` → `GRU (30 s sequence)`
   - `debut/milieu/fin (…)` → `early/middle/late (…)`
   - French feature names in `edges_ml.py`
   - French keys in `edges.py` E1
   - In `summary.py`: `NAMES`, `dD["D1_stacked"]["logistique"]`, and the `h2s['test|debut …']` keys.
   - Remaining French in `summary.py`/`timestamp_lag.py` docstrings, the `deep_extra.py` print lines, and
     `docs/data_notes.md` (already English).
5. **Restructure the README.**
   - Add a 3-bullet TL;DR at the top:
     - efficiency at ≥ 2 s latency;
     - the 2.25 s timestamp bias and its fake +7.6 c edge;
     - an ML gain ≤ 0.5 % of the log-loss.
   - Show `figures/fig3_latency.png` right after the TL;DR.
   - Keep: question, short data section, main results, limitations, reproduction.
   - Move the detailed method, the filter table, the full tables, the ML model list and the run-time table to a
     new `docs/METHODS.md`, linked from the README.
6. **Platform name.**
   - Replace "Prediction Market" used as a proper noun with "the venue" / "the prediction market". Places: README
     (title, question, data table, limitations), `pyproject.toml` description, `deep.py`, `deep_cd.py` labels,
     `deep_data.py`, `run.py` docstring.
   - Files that reveal the platform name (to report; do not change without a decision):
     - `config.yaml`: `gamma-api.polymarket.com`, `data-api.polymarket.com`, series_id 10684, slug template
       `{asset}-updown-{duration}-…`
     - `analysis/timestamp_lag.py`: `https://data-api.polymarket.com`
     - `pipelines/markets.py` and `pipelines/common/config.py`: `gamma` config key, raw directory `raw/gamma`, field
       `clobTokenIds`, source labels `gamma_*`
     - `tests/test_markets.py`: `clobTokenIds`, slugs `btc-updown-5m-…`; `tests/test_trades_features.py` slug
     - `docs/data_notes.md`: `/events?series_id=10684`, slug `btc-updown-5m-…`
     - `data/` raw paths `raw/gamma/…`, `raw/dataapi/…` (not in the repo)
     - Git remote and repository name: neutral.
7. **Small fixes.**
   - The duplicated `OUT.mkdir` line in `analysis/run.py` does **not exist** (only line 245, also in HEAD). Nothing
     to change.
   - The missing n for A9 is fixed (7,766).
8. Add `python scripts/check_figures.py` at the end of `reproduce.sh`. Run it, then complete
   `analysis/summary.py::key_figures` (including a "Protocol constants" section) until every README/METHODS number is
   found.
9. **Hypothesis count in the text.** The new total is 15 hypotheses: 14 tests in BH families, 1 ex-post diagnostic
   (A3) and 1 artefact (A8). It replaces "15 tests + 1 artefact" everywhere.

### Final verification, then commit block B

10. Run `pytest`, then `bash reproduce.sh` (with `PY=../.venv/Scripts/python` locally), then
    `python scripts/check_figures.py`. Everything must pass.
11. Commit block B.
12. Write the final report:
    - verdicts that changed (none so far);
    - key figures, market vs day bootstrap;
    - the A11 checks and conclusion;
    - the files that reveal the platform name;
    - items not done.
