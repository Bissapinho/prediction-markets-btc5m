#!/usr/bin/env bash
# Reproduces every number and figure of the README.
#
#   bash reproduce.sh            # analyses + summary + figures, from the parquet files already in data/
#   bash reproduce.sh --download # first downloads the raw data (~1 h, ~8 GB), then runs everything
#
# Run from the repository root. Each step's wall time is appended to results/runtime.log.
set -euo pipefail
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
[ -x "$PY" ] || PY=python
PM="$PY -m pipelines.cli"
export PM_SPLIT=2026-09-21      # first day of the test period
export PM_START=2026-08-17      # first day analysed (TWAP resolution rule)
export PM_ASK=prudent           # simulated ask with the mirror-book floor
export PYTHONHASHSEED=0
export PYTHONIOENCODING=utf-8
mkdir -p results figures
LOG=results/runtime.log
: > "$LOG"

step() {
    local t0=$SECONDS
    echo "=== $(date +%H:%M:%S) $*"
    "$@"
    echo "$(( SECONDS - t0 ))s  $*" | tee -a "$LOG"
}

if [ "${1:-}" = "--download" ]; then
    # 17 856 five-minute windows: 2026-08-17 00:00 -> 2026-10-17 23:55 UTC (start of the last market).
    step $PM ingest markets --n 17856 --end-ts 1792281300
    step $PM ingest trades
    step $PM ingest binance --start 2026-08-16 --end 2026-10-17
    step $PM build clean
    step $PM build features
fi

# H1-H4: lead-lag, TWAP fair value, fade, late favourites. Timestamp re-alignment of 0, 2 (reference) and 3 s.
step $PY -m analysis.run 0
step $PY -m analysis.run 2
step $PY -m analysis.run 3
# A9-A11 and the ML/DL comparison (logistic, gradient boosting, MLP, GRU, stacking).
step $PY -m analysis.deep_data
step $PY -m analysis.deep AB
step $PY -m analysis.deep D
step $PY -m analysis.deep_stack
step $PY -m analysis.deep_extra
# A11 robustness: re-alignment 0 / -2 / -3 s, Binance horizon [t, t+10] or [t+2, t+12], block bootstraps.
step $PY -m analysis.a11_robustness
# A1-A8: price-path strategies.
step $PY -m analysis.edges_data
step $PY -m analysis.edges E1 E2 E3 E4 E6 E8
# Timestamp bias (block time vs execution time), from data_sample/ when no live capture is present.
step $PY -m analysis.timestamp_lag
# Tables, key figures and figures of the README.
step $PY -m analysis.summary
step $PY -m analysis.figures
echo "=== done; total $SECONDS s"
echo "${SECONDS}s  total" >> "$LOG"
