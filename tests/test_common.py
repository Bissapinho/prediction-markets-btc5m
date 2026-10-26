import pytest

from pipelines.common.http import backoff_delay
from pipelines.common.manifest import Manifest
from pipelines.common.storage import Layout
from pipelines.common.timeutil import iso_to_ms, parse_duration

T0 = 1792336800_000  # 2026-10-18T15:20:00Z


def test_parse_duration():
    assert parse_duration("5m") == 300
    assert parse_duration("15m") == 900
    assert parse_duration("1h") == 3600
    with pytest.raises(ValueError):
        parse_duration("5 min")


@pytest.mark.parametrize(
    "s",
    ["2026-10-18T15:20:00Z", "2026-10-18T15:20:00+00:00", "2026-10-18 15:20:00+00", "2026-10-18T15:20:00.000000Z"],
)
def test_iso_to_ms_formats(s):
    assert iso_to_ms(s) == T0


def test_backoff_bounds():
    for attempt in range(10):
        d = backoff_delay(attempt, base=1.0, cap=60.0)
        assert 0.5 * min(60, 2**attempt) <= d <= 1.5 * min(60, 2**attempt)


def test_manifest_latest_status_wins(tmp_path):
    layout = Layout(tmp_path)
    m = Manifest(layout, "markets")
    m.record("a", "missing")
    m.record_many([{"unit_key": "a", "status": "ok"}, {"unit_key": "b", "status": "error", "error": "x"}])
    reloaded = Manifest(layout, "markets")
    assert reloaded.status("a") == "ok"
    assert reloaded.status("b") == "error"
    assert reloaded.status("c") is None

