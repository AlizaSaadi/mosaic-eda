"""Time-based facts about a parsed log: levels, errors per source, silences, error bursts.

A log is analyzed as a table, but its most important problems are in time: an outage shows
up as a silence, an incident as a burst of errors from one service. Code finds both, so the
agents can cite them (named keys: gap_1, burst_1, ...).
"""

from __future__ import annotations

import pandas as pd

from mosaic.evidence.store import EvidenceStore

ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL", "SEVERE", "CRIT", "EMERG", "ALERT"}
MIN_GAP_MINUTES = 5  # a silence must be at least this long
GAP_FACTOR = 20  # and this many times the usual time between lines
MIN_BURST_ERRORS = 3  # errors in one minute to count toward a burst
BURST_FACTOR = 5  # and this many times the log's average errors per minute


def _minutes(delta: pd.Timedelta) -> float:
    return round(delta.total_seconds() / 60, 1)


def _when(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def log_facts(df: pd.DataFrame) -> dict | None:
    if "timestamp" not in df or "level" not in df:
        return None
    ts = pd.to_datetime(df["timestamp"].str.replace(",", ".", regex=False), errors="coerce",
                        format="mixed")  # fmt: skip
    level = df["level"].fillna("").str.upper().str.strip()
    source = df["logger"].fillna("").replace("", "(none)") if "logger" in df else None
    frame = pd.DataFrame({"ts": ts, "level": level})
    if source is not None:
        frame["source"] = source
    frame = frame.dropna(subset=["ts"]).sort_values("ts")
    if len(frame) < 10:
        return None
    is_error = frame["level"].isin(ERROR_LEVELS)
    facts: dict = {
        "records": len(df),
        "start": _when(frame["ts"].iloc[0]),
        "end": _when(frame["ts"].iloc[-1]),
        "levels": {k: int(v) for k, v in frame["level"].value_counts().items() if k},
        "error_share_pct": round(100 * float(is_error.mean()), 2),
        "multi_line_records": int((pd.to_numeric(df.get("extra_lines"), errors="coerce")
                                   .fillna(0) > 0).sum()) if "extra_lines" in df else 0,
    }  # fmt: skip
    if "source" in frame:
        errors = frame.loc[is_error, "source"].value_counts()
        facts["errors_by_source"] = {k: int(v) for k, v in errors.head(8).items()}

    steps = frame["ts"].diff().dropna()
    usual = steps.median()
    gaps = steps[
        (steps >= pd.Timedelta(minutes=MIN_GAP_MINUTES)) & (steps >= usual * GAP_FACTOR)
    ].nlargest(3)
    facts["gaps"] = {
        f"gap_{i}": {
            "after": _when(frame["ts"].loc[:idx].iloc[-2]),
            "until": _when(frame["ts"].loc[idx]),
            "minutes": _minutes(delta),
        }
        for i, (idx, delta) in enumerate(gaps.items(), 1)
    }
    facts["usual_seconds_between_lines"] = round(usual.total_seconds(), 1)

    per_minute = frame.loc[is_error].set_index("ts").resample("1min").size()
    span = max((frame["ts"].iloc[-1] - frame["ts"].iloc[0]).total_seconds() / 60, 1)
    average = float(is_error.sum()) / span  # errors per minute over the whole log
    hot = per_minute[per_minute >= max(MIN_BURST_ERRORS, BURST_FACTOR * average)]
    bursts, current = [], None
    for minute, count in hot.items():
        if current and minute - current["end"] <= pd.Timedelta(minutes=2):  # one incident
            current["end"], current["errors"] = minute, current["errors"] + int(count)
        else:
            current = {"start": minute, "end": minute, "errors": int(count)}
            bursts.append(current)
    facts["bursts"] = {}
    for i, b in enumerate(sorted(bursts, key=lambda b: -b["errors"])[:3], 1):
        window = frame[is_error & (frame["ts"] >= b["start"])
                       & (frame["ts"] < b["end"] + pd.Timedelta(minutes=1))]  # fmt: skip
        top = window["source"].value_counts() if "source" in window else pd.Series(dtype=int)
        facts["bursts"][f"burst_{i}"] = {
            "start": _when(b["start"]),
            "minutes": _minutes(b["end"] - b["start"] + pd.Timedelta(minutes=1)),
            "errors": len(window),
            "main_source": str(top.index[0]) if len(top) else "",
            "main_source_share_pct": round(100 * float(top.iloc[0]) / len(window), 1)
            if len(top) else 0.0,
        }  # fmt: skip
    return facts


def record_log_facts(store: EvidenceStore, df: pd.DataFrame, stage: str) -> str | None:
    facts = log_facts(df)
    if facts is None:
        return None
    gaps = "; ".join(
        f"{k}: {g['minutes']} min with no lines after {g['after']}"
        for k, g in facts["gaps"].items()
    )
    bursts = "; ".join(
        f"{k}: {b['errors']} errors in {b['minutes']} min from {b['start']}, "
        f"{b['main_source_share_pct']}% from '{b['main_source']}'"
        for k, b in facts["bursts"].items()
    )
    by_source = facts.get("errors_by_source") or {}
    return store.add(
        "tbl_log",
        "profile",
        "log_timeline",
        f"Log timeline ({stage}; code): {facts['records']} records from {facts['start']} to "
        f"{facts['end']}; levels {facts['levels']} (levels.<LEVEL>); errors are "
        f"{facts['error_share_pct']}% of records (error_share_pct); errors by source "
        f"{by_source} (errors_by_source.<name>); {facts['multi_line_records']} records span "
        f"several lines, such as stack traces (multi_line_records). Usually "
        f"{facts['usual_seconds_between_lines']} s between lines. Silences: "
        f"{gaps or 'none'} (gaps.gap_<n>.minutes). Error bursts: {bursts or 'none'} "
        "(bursts.burst_<n>.errors / .minutes / .main_source_share_pct).",
        facts,
        {"stage": stage},
    ).id
