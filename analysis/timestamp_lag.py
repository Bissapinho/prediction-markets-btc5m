"""Delay between the execution of a trade (venue WebSocket feed, ms) and the timestamp of its block (trade-history API, s).

The (execution time, block time) pairs were extracted once from a one-day WebSocket capture (data/raw/ws/book, not
provided) and published in data_sample/timestamp_lag_pairs.csv, without hashes or wallets.
Without a local capture, the script starts from this CSV. Output: results/timestamp_lag.json.
Usage : python -m analysis.timestamp_lag [--extract]
"""

import csv
import glob
import json
import time
from pathlib import Path

import numpy as np

PAIRS = Path("data_sample/timestamp_lag_pairs.csv")


def extract_pairs() -> np.ndarray:
    import httpx

    from pipelines.common.storage import read_zst

    live, mk = {}, set()
    for f in glob.glob("data/raw/ws/book/*/*/*.zst"):
        for line in read_zst(f).decode().splitlines():
            rec = json.loads(line)
            if "raw" not in rec:
                continue
            d = json.loads(rec["raw"])
            for it in (d if isinstance(d, list) else [d]):
                if it.get("event_type") == "last_trade_price" and it.get("transaction_hash"):
                    live[it["transaction_hash"].lower()] = int(it["timestamp"])
                    mk.add(it["market"])
    cl = httpx.Client(base_url="https://data-api.polymarket.com", timeout=60)
    blk = {}
    for m in sorted(mk):
        for attempt in range(5):
            r = cl.get("/trades", params={"market": m, "limit": 10000}).json()
            if isinstance(r, list):
                break
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"trade-history API /trades: unexpected response for {m}: {str(r)[:200]}")
        for x in r:
            blk[x["transactionHash"].lower()] = x["timestamp"] * 1000
    pairs = np.array(sorted((live[h], blk[h]) for h in live if h in blk), dtype=np.int64)
    PAIRS.parent.mkdir(exist_ok=True)
    with open(PAIRS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exec_ms", "block_ms"])
        w.writerows(pairs.tolist())
    return pairs


def main(extract: bool = False) -> None:
    # The published numbers come from the CSV; re-extracting needs the live capture and network access.
    if extract and glob.glob("data/raw/ws/book/*/*/*.zst"):
        pairs = extract_pairs()
    else:
        pairs = np.loadtxt(PAIRS, delimiter=",", skiprows=1, dtype=np.int64)
    d = (pairs[:, 1] - pairs[:, 0]) / 1000
    edges = np.arange(0, 4.01, 0.25)
    h, _ = np.histogram(d, edges)
    out = {"n": int(len(d)), "median": float(np.median(d)), "mean": float(d.mean()),
           "q": {str(q): float(np.quantile(d, q)) for q in (.01, .1, .25, .5, .75, .9, .99)},
           "hist": [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(h[i])} for i in range(len(h))]}
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/timestamp_lag.json", "w"), indent=1)
    print(out["n"], out["median"])


if __name__ == "__main__":
    import sys

    main("--extract" in sys.argv[1:])
