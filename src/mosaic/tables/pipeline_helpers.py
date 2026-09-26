"""Helper functions used by cleaning operations.

This module's source is copied verbatim into every exported cleaning_pipeline.py,
so it must only import pandas, numpy, and re.
"""

import re

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "#n/a", "nil"}
TRUE_WORDS = {"true", "yes", "y", "t", "1"}
FALSE_WORDS = {"false", "no", "n", "f", "0"}


def is_missing(s):
    """True where a value is empty or a missing-value token such as 'N/A'."""
    return s.isna() | s.astype("str").str.strip().str.lower().isin(NULL_TOKENS)


def to_number(s, decimal_comma=False):
    """Parse numbers, currency, and percentages ('$1,204.50', '10%') to floats."""
    text = s.where(~is_missing(s), None).astype("str")
    text = text.str.replace(r"[$€£¥₹%\s]|USD|EUR|GBP", "", regex=True, flags=re.I)
    if decimal_comma:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        text = text.str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def to_bool(s):
    words = s.astype("str").str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[words.isin(TRUE_WORDS)] = True
    out[words.isin(FALSE_WORDS)] = False
    return out


def to_datetime(s, dayfirst=False):
    values = s.where(~is_missing(s), None)
    return pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=dayfirst)


def iqr_bounds(x, k=1.5):
    q1, q3 = np.nanpercentile(x.astype(float), [25, 75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


LEVELS = r"TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|SEVERE|CRITICAL|FATAL"
LOG_FORMATS = {
    "access": re.compile(
        r"^(?P<ip>\S+) (?P<identity>\S+) (?P<user>\S+) \[(?P<timestamp>[^\]]+)\] "
        r'"(?P<method>[A-Z]+) (?P<path>\S+)(?: (?P<protocol>[^"]*))?" (?P<status>\d{3}) '
        r'(?P<bytes>\S+)(?: "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)")?'
    ),
    "app": re.compile(
        r"^\[?(?P<timestamp>\d{4}-\d{2}-\d{2}[ T][\d:.,]+(?:Z|[+-]\d{2}:?\d{2})?)\]?\s+"
        r"\[?(?P<level>" + LEVELS + r")\]?\s+(?:\[?(?P<logger>[\w.$/-]+)\]?\s*[:-]\s+)?"
        r"(?P<message>.*)$",
        re.I,
    ),
    "level_first": re.compile(
        r"^\[?(?P<level>" + LEVELS + r")\]?\s+(?:\[?(?P<timestamp>\d{4}-\d{2}-\d{2}"
        r"[ T][\d:.,]+)\]?\s+)?(?P<message>.*)$",
        re.I,
    ),
    "syslog": re.compile(
        r"^(?P<timestamp>[A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2}) (?P<host>\S+) "
        r"(?P<process>[\w./-]+?)(?:\[(?P<pid>\d+)\])?: (?P<message>.*)$"
    ),
}


def parse_log_lines(path):
    """Parse a log file into a table. The format (web access, app, level-first, or syslog)
    is the one most lines match; lines that match nothing (stack traces, wrapped text) are
    appended to the previous record's message."""
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    lines = [ln.rstrip("\r") for ln in text.split("\n") if ln.strip()]
    head = lines[:500]
    name = max(LOG_FORMATS, key=lambda k: sum(bool(LOG_FORMATS[k].match(ln)) for ln in head))
    pattern = LOG_FORMATS[name]
    columns = list(pattern.groupindex)
    records = []
    for line in lines:
        m = pattern.match(line)
        if m:
            record = {k: (v or "") for k, v in m.groupdict().items()}
            record["extra_lines"] = "0"
            records.append(record)
        elif records:  # continuation of the previous record
            key = "message" if "message" in columns else columns[-1]
            records[-1][key] = (records[-1][key] + "\n" + line.strip()).strip()
            records[-1]["extra_lines"] = str(int(records[-1]["extra_lines"]) + 1)
    df = pd.DataFrame(records, columns=[*columns, "extra_lines"], dtype="str")
    df.attrs["log_format"] = name
    return df
