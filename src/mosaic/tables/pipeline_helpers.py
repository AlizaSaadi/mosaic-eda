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
