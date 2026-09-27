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


def uses_decimal_comma(text):
    """True when the values are written the European way ('2.167,60', '159,48'): more
    values can only be read that way than can only be read the US way ('1,192.90')."""
    text = text.dropna().astype("str")
    european = text.str.fullmatch(r"-?\d{1,3}(\.\d{3})+,\d+|-?\d+,\d{1,2}|-?\d+,\d{4,}")
    us = text.str.fullmatch(r"-?\d{1,3}(,\d{3})+\.\d+|-?\d+\.\d+|-?\d{1,3}(,\d{3}){2,}")
    return int(european.sum()) > int(us.sum())


def to_number(s, decimal_comma=None):
    """Parse numbers, currency, and percentages ('$1,204.50', '10%', '2.167,60 EUR') to
    floats. decimal_comma=None works out from the values which way they're written."""
    text = s.where(~is_missing(s), None).astype("str")
    text = text.str.replace(r"[$€£¥₹%\s]|USD|EUR|GBP", "", regex=True, flags=re.I)
    if decimal_comma is None:
        decimal_comma = uses_decimal_comma(text.where(s.notna()))
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


def day_comes_first(s):
    """True for dates like 25.12.2025 or 25/12/2025: a first part over 12, or dots
    (the European convention), with no second part over 12."""
    parts = s.dropna().astype("str").str.extract(r"^\s*(\d{1,2})([./-])(\d{1,2})[./-]\d{2,4}")
    parts = parts.dropna()
    if parts.empty:
        return False
    first, second = parts[0].astype(int), parts[2].astype(int)
    if (second > 12).any():
        return False
    return bool((first > 12).any() or (parts[1] == ".").mean() > 0.5)


def to_datetime(s, dayfirst=None):
    """Parse dates in any common format. dayfirst=None works it out from the values."""
    values = s.where(~is_missing(s), None)
    if dayfirst is None:
        dayfirst = day_comes_first(values)
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
        r"\[?(?P<level>" + LEVELS + r")\]?\s+"
        # a logger in brackets ("[payment] ...") or followed by ":" or "-" ("payment: ...")
        r"(?:(?P<logger>\[[\w.$/-]+\](?=\s)|[\w.$/-]+(?=\s*[:-]\s))\s*[:-]?\s+)?"
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
            if record.get("logger", "").startswith("["):
                record["logger"] = record["logger"].strip("[]")
            record["extra_lines"] = "0"
            records.append(record)
        elif records:  # continuation of the previous record
            key = "message" if "message" in columns else columns[-1]
            records[-1][key] = (records[-1][key] + "\n" + line.strip()).strip()
            records[-1]["extra_lines"] = str(int(records[-1]["extra_lines"]) + 1)
    df = pd.DataFrame(records, columns=[*columns, "extra_lines"], dtype="str")
    df.attrs["log_format"] = name
    return df
