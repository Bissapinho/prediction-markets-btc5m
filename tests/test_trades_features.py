import json
from datetime import datetime, timezone

import duckdb
import polars as pl
import pytest

from pipelines.binance import parse_aggtrades_csv
from pipelines.common.config import load_config
from pipelines.common.storage import Layout, write_parquet, write_zst
from pipelines.features import build_btc_1s, build_market_1s
from pipelines.markets import MARKETS_SCHEMA
from pipelines.trades import build_trades_clean, fee_usdc, trade_key

START = 1792336800  # 2026-10-18T15:20:00Z
SLUG = f"btc-updown-5m-{START}"


def test_fee_formula_matches_onchain_example():
    # Fill on-chain vérifié (docs/data_notes.md) : 11 254,548 shares Up à 0,01 -> 7,7994 USDC.
    assert fee_usdc(0.01, 11254.548, 0.07, 1) == pytest.approx(7.7994, abs=1e-4)
    assert fee_usdc(0.5, 100, 0.07, 1) == pytest.approx(1.75)


def test_trade_key_is_stable_and_discriminant():
    a = trade_key("0xabc", "1", "0xw", "BUY", 10.0, 0.5, 1)
    assert a == trade_key("0xabc", "1", "0xw", "BUY", 10.0, 0.5, 1)
    assert a != trade_key("0xabc", "1", "0xw", "SELL", 10.0, 0.5, 1)


@pytest.mark.parametrize("unit_us", [True, False])
def test_parse_aggtrades_csv_units(unit_us):
    ts = 1792108800023206 if unit_us else 1792108800023
    csv = f"1,82245.93,0.02,10,11,{ts},True,True\n2,82245.92,0.5,12,12,{ts + 1},False,True\n".encode()
    df = parse_aggtrades_csv(csv)
    assert df["ts_ms"].to_list()[0] == 1792108800023
    assert df["is_buyer_maker"].to_list() == [True, False]


def test_parse_aggtrades_csv_with_header():
    csv = b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match\n1,1.0,2.0,1,1,1792108800023206,true,true\n"
    assert parse_aggtrades_csv(csv).height == 1


def _market_row():
    row = {k: None for k in MARKETS_SCHEMA}
    row.update(market_id="0xm", slug=SLUG, asset="btc", duration_s=300, start_ts_ms=START * 1000,
               end_ts_ms=(START + 300) * 1000, token_up="TU", token_down="TD", outcome="up", closed=True,
               fee_rate=0.07, fee_exponent=1.0, fetched_at_ms=1, ingested_at=datetime.now(timezone.utc))
    return row


def _api_trade(ts, outcome, price, size=10.0, tx=None):
    return {"proxyWallet": "0xw", "side": "BUY", "asset": "TU" if outcome == "Up" else "TD", "size": size,
            "price": price, "timestamp": ts, "outcome": outcome, "transactionHash": tx or f"0x{ts}{outcome}{price}"}


def test_end_to_end_market_1s(tmp_path):
    cfg = load_config("config.yaml")
    cfg.storage.root = tmp_path
    layout = Layout(tmp_path)
    write_parquet(pl.DataFrame([_market_row()], schema=MARKETS_SCHEMA), layout.clean("markets", "markets.parquet"))

    # Réponse Data API : du plus récent au plus ancien.
    api = [
        _api_trade(START + 10, "Down", 0.30),  # à t=10, Down plus récent -> implied = 0.70
        _api_trade(START + 5, "Up", 0.62),
        _api_trade(START + 5, "Up", 0.61),     # même seconde, plus ancien : p_up_last doit valoir 0.62
        _api_trade(START - 30, "Up", 0.55),    # avant l'ouverture : sert au forward-fill de t=0
    ]
    write_zst(layout.raw("dataapi", "trades", "date=2026-10-18", f"{SLUG}-1792340000000.json.zst"), json.dumps(api).encode())
    assert build_trades_clean(cfg, layout) == 4

    # Binance : un trade par seconde de start-5 à la fin, prix croissant, sauf une seconde vide (t=100).
    rows = [(i, 80000.0 + i, 0.1, i, i, (START - 5 + i) * 1_000_000 + 1, True)
            for i in range(0, 306) if i != 105]
    df = pl.DataFrame(rows, schema=["agg_id", "price", "qty", "first_trade_id", "last_trade_id", "ts_us", "is_buyer_maker"],
                      orient="row").with_columns(ts_ms=pl.col("ts_us") // 1000)
    write_parquet(df, layout.clean("btc_aggtrades", "date=2026-10-18", "aggtrades.parquet"))

    con = duckdb.connect()
    build_btc_1s(cfg, layout, con)
    assert build_market_1s(cfg, layout, con) == 300

    m = {r["t"]: r for r in duckdb.sql(
        f"SELECT * FROM read_parquet('{layout.features('market_1s').as_posix()}/*/*.parquet', hive_partitioning=false)"
    ).pl().iter_rows(named=True)}
    assert m[0]["p_up_last"] == 0.55 and m[0]["up_age_s"] == 30 and m[0]["p_down_last"] is None
    assert m[0]["p_up_implied"] == 0.55
    assert m[5]["p_up_last"] == 0.62 and m[5]["vol_up"] == 20 and m[5]["n_trades_up"] == 2
    assert m[9]["up_age_s"] == 4
    assert m[10]["p_up_implied"] == pytest.approx(0.70) and m[10]["down_age_s"] == 0
    assert m[299]["tau_s"] == 1 and m[0]["outcome_up"] is True
    # btc_s0 = close de la seconde start-1 (i=4 -> 80004), forward-fill sur la seconde vide t=100.
    assert m[0]["btc_s0"] == 80004.0
    assert m[100]["btc_close"] == m[99]["btc_close"]
