"""Logs: bracketed service names are parsed, and code finds silences and error bursts."""

from datetime import datetime, timedelta

from mosaic.tables.logs import log_facts
from mosaic.tables.pipeline_helpers import parse_log_lines


def test_a_silence_and_an_error_burst_are_found(tmp_path):
    t = datetime(2026, 3, 2, 8, 0)
    lines = []
    for i in range(300):
        t += timedelta(seconds=10)
        if i == 100:
            t += timedelta(minutes=30)  # an outage
        burst = 200 <= i < 230
        level, service = ("ERROR", "payment") if burst else ("INFO", ["api", "auth"][i % 2])
        lines.append(f"{t:%Y-%m-%d %H:%M:%S},000 {level} [{service}] message {i}")
    path = tmp_path / "app.log"
    path.write_text("\n".join(lines), encoding="utf-8")

    df = parse_log_lines(path)
    assert set(df["logger"]) == {"api", "auth", "payment"}
    assert not df["message"].str.startswith("[").any()

    facts = log_facts(df)
    assert facts["gaps"]["gap_1"]["minutes"] >= 30
    burst = facts["bursts"]["burst_1"]
    assert burst["main_source"] == "payment" and burst["errors"] == 30
    assert facts["errors_by_source"] == {"payment": 30}
