"""Work out what each text column really holds, and parse values accordingly."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "#n/a", "nil"}

_NUM = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?([eE][+-]?\d+)?$|^[+-]?\.\d+$")
_NUM_EU = re.compile(r"^[+-]?(\d{1,3}(\.\d{3})+|\d+),\d+$")
_CURRENCY = re.compile(
    r"^[+-]?\s*(?:[$€£¥₹]|USD|EUR|GBP)\s*[\d.,]+\s*$|^[+-]?[\d.,]+\s*(?:[$€£¥₹]|USD|EUR|GBP)$", re.I
)
_PERCENT = re.compile(r"^[+-]?\d+([.,]\d+)?\s*%$")
_BOOL = {"true", "false", "yes", "no", "y", "n", "t", "f"}
_ID_NAME = re.compile(r"(^|_|\b)(id|uuid|guid|key|code|sku|ref)($|_|\b)", re.I)
_DATE_HINT = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+[A-Za-z]{3,}|[A-Za-z]{3,}\s+\d{1,2}"
)


@dataclass(frozen=True)
class ColumnType:
    kind: str  # empty, constant, id, numeric, currency, percent, boolean, datetime, category, text
    parse_rate: float = 1.0  # share of non-missing values that parse as this kind
    detail: str = ""


def missing_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").str.strip().str.lower().isin(NULL_TOKENS)


def hidden_missing_count(series: pd.Series) -> int:
    """Values like 'N/A' or '-' that mean missing but aren't empty."""
    stripped = series.fillna("").str.strip()
    return int((missing_mask(series) & (stripped != "")).sum())


def _strip_number(text: pd.Series, decimal_comma: bool) -> pd.Series:
    text = text.str.replace(r"[$€£¥₹%\s]|USD|EUR|GBP", "", regex=True, flags=re.I)
    if decimal_comma:
        return text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return text.str.replace(",", "", regex=False)


def parse_numbers(series: pd.Series, decimal_comma: bool = False) -> pd.Series:
    """Parse plain numbers, currency, and percentages to floats. Unparseable -> NaN."""
    values = series.where(~missing_mask(series), None).astype("str")
    return pd.to_numeric(_strip_number(values, decimal_comma), errors="coerce")


def parse_dates(series: pd.Series) -> pd.Series:
    values = series.where(~missing_mask(series), None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=False)


def _share(values: pd.Series, pattern: re.Pattern) -> float:
    return float(values.str.fullmatch(pattern).mean()) if len(values) else 0.0


def detect_type(series: pd.Series, name: str = "") -> ColumnType:
    present = series[~missing_mask(series)].str.strip()
    n = len(present)
    if n == 0:
        return ColumnType("empty", 1.0)
    distinct = present.nunique()
    if distinct == 1 and len(series) > 1:
        return ColumnType("constant", 1.0, f"always '{present.iloc[0]}'")
    sample = present.sample(min(n, 5000), random_state=0) if n > 5000 else present

    lowered = set(sample.str.lower().unique())
    if lowered <= _BOOL or (lowered <= {"0", "1"} and distinct == 2):
        return ColumnType("boolean", 1.0)

    rates = {
        "percent": _share(sample, _PERCENT),
        "currency": _share(sample, _CURRENCY),
        "numeric": _share(sample, _NUM),
        "numeric_eu": _share(sample, _NUM_EU),
    }
    if rates["percent"] >= 0.9:
        return ColumnType("percent", rates["percent"])
    if rates["currency"] >= 0.9 or (
        rates["currency"] >= 0.5 and rates["currency"] + rates["numeric"] >= 0.9
    ):
        return ColumnType("currency", rates["currency"] + rates["numeric"])
    numeric_rate = rates["numeric"] + rates["numeric_eu"]
    if numeric_rate >= 0.9:
        decimal_comma = rates["numeric_eu"] > rates["numeric"] * 0.5 and rates["numeric_eu"] > 0
        detail = "decimal comma" if decimal_comma else ""
        is_int = bool(sample.str.fullmatch(r"[+-]?\d+").all())
        # IDs repeat when a table has duplicate rows, so allow a little repetition
        if is_int and distinct / n >= 0.9 and _ID_NAME.search(name):
            return ColumnType("id", 1.0, "unique integers with an ID-like name")
        return ColumnType("numeric", min(numeric_rate, 1.0), detail)

    if float(sample.str.contains(_DATE_HINT).mean()) >= 0.8:
        parsed = parse_dates(sample)
        rate = float(parsed.notna().mean())
        if rate >= 0.8:
            return ColumnType("datetime", rate)

    unique_ratio = distinct / n
    if (
        unique_ratio >= 0.95
        and n >= 20
        and (_ID_NAME.search(name) or sample.str.fullmatch(r"[A-Za-z0-9_-]{4,40}").mean() >= 0.95)
    ):
        return ColumnType("id", 1.0, "unique codes")
    avg_len = float(sample.str.len().mean())
    if avg_len > 40 and unique_ratio > 0.5:
        return ColumnType("text", 1.0)
    return ColumnType("category", 1.0, f"{distinct} distinct values")
