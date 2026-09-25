"""Load CSV, TSV, JSON Lines, and Excel files as all-text tables.

Every cell is read as a string so nothing is silently converted. Column types
are worked out afterwards (see semantics.py) and applied by cleaning operations.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mosaic.ingest.models import IngestError

MAX_ROWS = 1_000_000
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


@dataclass
class LoadedTable:
    df: pd.DataFrame
    source: str
    format: str
    encoding: str = ""
    delimiter: str = ""
    sheet: str | None = None
    sheets: list[str] = field(default_factory=list)
    header_row: int = 0
    total_rows: int = 0
    notes: list[str] = field(default_factory=list)


def decode_bytes(raw: bytes) -> tuple[str, str]:
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise IngestError("bad_encoding", "The file's text encoding couldn't be read.")


def sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def find_header_row(rows: list[list[str]], look: int = 10) -> int:
    """Skip title rows above the real header: rows with at most one filled cell."""
    width = max((len(r) for r in rows[:look]), default=0)
    for i, row in enumerate(rows[:look]):
        filled = sum(bool(str(c).strip()) for c in row)
        if filled >= max(2, width // 2):
            return i
    return 0


def _unique_names(names: list[str]) -> tuple[list[str], list[str]]:
    notes, seen, out = [], {}, []
    for i, name in enumerate(names):
        name = str(name).strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            new = f"{name}_{seen[name]}"
            notes.append(f"Duplicate column name '{name}' renamed to '{new}'.")
            name = new
        else:
            seen[name] = 1
        out.append(name)
    return out, notes


def _frame_from_rows(rows: list[list[str]], table: LoadedTable) -> pd.DataFrame:
    header = find_header_row(rows)
    if header:
        table.notes.append(f"Skipped {header} title row(s) above the header.")
    table.header_row = header
    names, notes = _unique_names(rows[header] if rows else [])
    table.notes.extend(notes)
    body = [r + [""] * (len(names) - len(r)) for r in rows[header + 1 :]]
    ragged = sum(len(r) > len(names) for r in rows[header + 1 :])
    if ragged:
        table.notes.append(f"{ragged} row(s) had extra cells, which were dropped.")
    body = [r[: len(names)] for r in body]
    return pd.DataFrame(body, columns=names, dtype="str")


def _cap(table: LoadedTable) -> None:
    table.total_rows = len(table.df)
    if len(table.df) > MAX_ROWS:
        table.df = table.df.iloc[:MAX_ROWS].reset_index(drop=True)
        table.notes.append(f"Only the first {MAX_ROWS:,} of {table.total_rows:,} rows are used.")


def load_table(path: Path, fmt: str) -> LoadedTable:
    table = LoadedTable(df=pd.DataFrame(), source=path.name, format=fmt)
    if fmt in ("xlsx", "xls"):
        engine = "openpyxl" if fmt == "xlsx" else "xlrd"
        try:
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str, engine=engine)
        except Exception as exc:
            raise IngestError("bad_excel", "The Excel file couldn't be read.") from exc
        table.sheets = list(sheets)
        filled = {name: int(df.notna().sum().sum()) for name, df in sheets.items()}
        table.sheet = max(filled, key=filled.get)
        if len(sheets) > 1:
            table.notes.append(
                f"The workbook has {len(sheets)} sheets. Using '{table.sheet}', the fullest one."
            )
        raw = sheets[table.sheet].fillna("")
        rows = [[str(c) for c in row] for row in raw.itertuples(index=False)]
        table.df = _frame_from_rows(rows, table)
    elif fmt == "jsonl":
        text, table.encoding = decode_bytes(path.read_bytes())
        df = pd.read_json(io.StringIO(text), lines=True, dtype=False)
        table.df = df.astype("str").where(df.notna(), "")
    else:
        text, table.encoding = decode_bytes(path.read_bytes())
        table.delimiter = sniff_delimiter(text)
        rows = list(csv.reader(io.StringIO(text), delimiter=table.delimiter))
        rows = [r for r in rows if any(c.strip() for c in r)] or [[]]
        table.df = _frame_from_rows(rows, table)
    _cap(table)
    return table
