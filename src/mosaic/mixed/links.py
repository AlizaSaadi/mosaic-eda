"""Cross-type evidence for group mode, computed by code.

The most useful link in a mixed dataset is a table that describes files ("metadata.csv"
with an image column): which rows point to files that exist, which point to missing
files, which files no row mentions, and whether the table's label agrees with the file's
folder. Everything is saved as evidence the Cross-Type Synthesizer can cite.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from mosaic.evidence.store import EvidenceStore
from mosaic.ingest.models import FileEntry
from mosaic.tables.pipeline_helpers import is_missing

FILE_LIKE = re.compile(r"[\\/]|\.[A-Za-z0-9]{2,4}$")
MIN_MATCH_SHARE = 0.5
MIN_LABEL_SHARE = 0.6


def _norm_label(value: str) -> str:
    """'Circles ', 'circle', 'CIRCLE' -> 'circle' (case, spaces, and a plural 's')."""
    v = re.sub(r"[\s_-]+", " ", str(value).strip().lower())
    return v[:-1] if len(v) > 3 and v.endswith("s") and not v.endswith("ss") else v


class FileIndex:
    """Look files up by full path, path suffix, file name, or file name without extension."""

    def __init__(self, files: list[FileEntry]) -> None:
        self.files = files
        self.by_key: dict[str, list[str]] = defaultdict(list)
        for f in files:
            p = PurePosixPath(f.path.lower())
            for key in {p.as_posix(), p.name, p.stem}:
                self.by_key[key].append(f.path)

    def match(self, value: str) -> str | None:
        v = str(value).strip().replace("\\", "/").lower().lstrip("./")
        if not v:
            return None
        for key in (v, PurePosixPath(v).name, PurePosixPath(v).stem):
            hits = self.by_key.get(key, [])
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:  # the same file name in several folders: use the longer path
                full = [h for h in hits if h.lower().endswith(v)]
                return full[0] if len(full) == 1 else None
        return None


def find_file_column(df: pd.DataFrame, index: FileIndex) -> tuple[str | None, float]:
    best, best_share = None, 0.0
    for col in df.columns:
        values = df[col][~is_missing(df[col])].astype(str)
        if len(values) < 3:
            continue
        share = sum(index.match(v) is not None for v in values) / len(values)
        if share > best_share:
            best, best_share = col, share
    return (best, best_share) if best_share >= MIN_MATCH_SHARE else (None, best_share)


def link_table(table_name: str, df: pd.DataFrame, files: list[FileEntry]) -> dict[str, Any] | None:
    """How a table's rows line up with the dataset's files. None if no column holds files."""
    index = FileIndex(files)
    col, share = find_file_column(df, index)
    if col is None:
        return None
    present = ~is_missing(df[col])
    matches = df[col].where(present, "").astype(str).map(index.match)
    matched = matches.notna()
    looks_like_file = df[col].astype(str).str.contains(FILE_LIKE)
    missing = present & ~matched & looks_like_file
    referenced = Counter(matches.dropna())
    unreferenced = sorted({f.path for f in files} - set(referenced))
    folder = {f.path: PurePosixPath(f.path).parent.name for f in files}

    # a column that names the same classes as the folders
    labels = {_norm_label(v) for v in folder.values() if v}
    label_col, label_share = None, 0.0
    for other in df.columns:
        if other == col or not labels:
            continue
        values = df.loc[matched, other].astype(str).map(_norm_label)
        share_in = values.isin(labels).mean() if len(values) else 0.0
        if share_in > label_share:
            label_col, label_share = other, float(share_in)
    disagreements = []
    if label_col is not None and label_share >= MIN_LABEL_SHARE:
        for i in df.index[matched]:
            path = matches[i]
            table_label = str(df.at[i, label_col])
            if _norm_label(table_label) != _norm_label(folder[path]):
                disagreements.append(
                    {"row": int(i) + 2, "file": path, "table_label": table_label,
                     "folder": folder[path]}
                )  # fmt: skip
    else:
        label_col = None

    return {
        "table": table_name,
        "file_column": col,
        "rows": len(df),
        "rows_matched": int(matched.sum()),
        "match_share": round(100 * float(share), 2),
        "rows_missing_file": int(missing.sum()),
        "missing_examples": df.loc[missing, col].astype(str).head(10).tolist(),
        "files": len(files),
        "files_unreferenced": len(unreferenced),
        "unreferenced_examples": unreferenced[:10],
        "files_referenced_twice": sum(n > 1 for n in referenced.values()),
        "label_column": label_col,
        "label_disagreements": len(disagreements),
        "disagreement_examples": disagreements[:10],
    }


def record_links(store: EvidenceStore, link: dict[str, Any]) -> str:
    label_text = (
        f"The table's '{link['label_column']}' column disagrees with the file's folder for "
        f"{link['label_disagreements']} rows (label_disagreements): "
        + "; ".join(
            f"row {d['row']} {d['file']} says '{d['table_label']}', folder '{d['folder']}'"
            for d in link["disagreement_examples"][:5]
        )
        if link["label_column"]
        else "No table column names the same classes as the folders."
    )
    return store.add(
        "mix_links",
        "profile",
        "table_file_links",
        f"Table-to-file links (code): {link['table']} column '{link['file_column']}' names "
        f"files. {link['rows_matched']} of {link['rows']} rows match a file (rows_matched, "
        f"rows); {link['rows_missing_file']} rows point to files that aren't in the dataset "
        f"(rows_missing_file): {link['missing_examples'][:5]}; {link['files_unreferenced']} of "
        f"{link['files']} files have no row (files_unreferenced, files): "
        f"{link['unreferenced_examples'][:5]}; {link['files_referenced_twice']} files are "
        f"referenced by more than one row (files_referenced_twice). {label_text}",
        link,
    ).id


def record_overview(store: EvidenceStore, parts: list[dict[str, Any]]) -> str:
    """One entry per data type with its sub-run's results (keys parts.<type>.<field>)."""
    data = {
        "types": len(parts),
        "parts": {
            p["modality"]: {
                k: p[k]
                for k in (
                    "status",
                    "files",
                    "unit",
                    "items_before",
                    "items_after",
                    "quality_before",
                    "quality_after",
                    "findings",
                )
            }
            for p in parts
        },
    }
    text = "; ".join(
        f"{p['modality']}: {p['files']} files, {p['status']}"
        + (
            f", quality {p['quality_before']} -> {p['quality_after']} out of 100, "
            f"{p['unit']} {p['items_before']} -> {p['items_after']}, {p['findings']} findings"
            if p["status"] == "done"
            else f" ({p['error'][:120]})"
        )
        for p in parts
    )
    return store.add(
        "mix_overview",
        "profile",
        "group_overview",
        f"Group mode: {len(parts)} data types, each analyzed and cleaned by its own crew. "
        f"{text}. Keys parts.<type>.quality_before/quality_after/items_before/items_after/"
        "findings.",
        data,
    ).id
