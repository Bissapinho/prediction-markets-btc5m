import json

from pipelines.markets import Window, _outcome, parse_event, slug_start_ts, target_starts, windows_for

TOK_UP = "105304638640581278732637135939292881669716160952855240878319918948914038203532"
TOK_DOWN = "71998849010298484930906687362014872655580753763787043955344514597860992737940"


def make_event(closed=True, prices=("1", "0"), outcomes=("Up", "Down"), meta=True):
    ev = {
        "id": "1205686",
        "slug": "btc-updown-5m-1792336800",
        "startTime": "2026-10-18T15:20:00Z",
        "closedTime": "2026-10-18T15:26:26Z",
        "markets": [
            {
                "conditionId": "0x63af",
                "slug": "btc-updown-5m-1792336800",
                # startDate = date de création, pas le début de fenêtre.
                "startDate": "2026-10-17T15:28:27.326418Z",
                "endDate": "2026-10-18T15:25:00Z",
                "eventStartTime": "2026-10-18T15:20:00Z",
                "closedTime": "2026-10-18 15:26:26+00",
                "outcomes": json.dumps(list(outcomes)),
                "outcomePrices": json.dumps(list(prices)),
                "clobTokenIds": json.dumps([TOK_UP, TOK_DOWN]),
                "closed": closed,
                "umaResolutionStatus": "resolved",
                "volumeNum": 31762.96,
                "orderPriceMinTickSize": 0.001,
                "orderMinSize": 5,
                "feeType": "crypto_fees_v2",
                "feeSchedule": {"exponent": 1, "rate": 0.07, "takerOnly": True},
                "cryptoMarketConfig": {"asset": "btc", "duration": "5m", "twapLookbackSeconds": 60},
            }
        ],
    }
    if meta:
        ev["eventMetadata"] = {"finalPrice": 81113.65, "priceToBeat": 81094.37}
    return ev


def test_parse_event_fields():
    (row,) = parse_event(make_event(), fetched_at_ms=1, default_asset="btc", default_duration_s=300)
    assert row["market_id"] == "0x63af"
    assert row["start_ts_ms"] == 1792336800_000
    assert row["end_ts_ms"] == 1792337100_000
    assert row["event_start_time_ms"] == row["start_ts_ms"]
    assert row["closed_time_ms"] == 1792337186_000
    assert (row["token_up"], row["token_down"]) == (TOK_UP, TOK_DOWN)
    assert row["outcome"] == "up"
    assert row["fee_rate"] == 0.07 and row["fee_exponent"] == 1
    assert row["price_start"] == 81094.37 and row["price_end"] == 81113.65
    assert row["resolution_price_source"] == "gamma"
    assert row["twap_lookback_s"] == 60


def test_parse_event_token_mapping_follows_outcome_order():
    (row,) = parse_event(make_event(outcomes=("Down", "Up"), prices=("0", "1")), 1, "btc", 300)
    assert row["token_down"] == TOK_UP and row["token_up"] == TOK_DOWN
    assert row["outcome"] == "up"


def test_parse_event_without_metadata():
    (row,) = parse_event(make_event(meta=False), 1, "btc", 300)
    assert row["price_start"] is None and row["resolution_price_source"] is None


def test_outcome_rules():
    assert _outcome(True, ["Up", "Down"], ["0", "1"]) == "down"
    assert _outcome(False, ["Up", "Down"], ["1", "0"]) is None
    assert _outcome(True, ["Up", "Down"], ["0.995", "0.005"]) is None
    assert _outcome(True, ["Up", "Down"], []) is None


def test_slug_start_ts():
    assert slug_start_ts("btc-updown-5m-1792336800") == 1792336800
    assert slug_start_ts("btc-updown-5m") is None


def test_windows_cover_all_end_dates():
    starts = target_starts(1792340100, 1000, 300)
    assert len(starts) == 1000 and starts[0] == 1792340100 and starts[-1] == 1792340100 - 999 * 300
    windows = windows_for(starts, 300, 6)
    for s in starts:
        end_ms = (s + 300) * 1000
        assert any(w.start_ms <= end_ms <= w.end_ms for w in windows)
    # Fenêtres alignées sur 6 h UTC et contiguës.
    assert all(w.start_ms % (6 * 3600 * 1000) == 0 for w in windows)
    assert all(a.end_ms == b.start_ms for a, b in zip(windows, windows[1:]))


def test_window_key():
    assert Window(1792303200_000, 1792324800_000).key == "2026-10-18T06"
