"""Validate, dry-run, and apply cleaning plans; export the pipeline script and cleaned data."""

from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.evidence.store import EvidenceStore
from mosaic.security import injection
from mosaic.tables import pipeline_helpers
from mosaic.tables.load import LoadedTable
from mosaic.tables.ops import OPS, CleaningPlan, Risk, op_code, run_code

CONVERSIONS = {"strip_currency", "parse_percent", "cast_numeric", "parse_dates", "parse_boolean"}
MAX_ROW_LOSS = 0.30
MAX_NEW_MISSING = 0.10
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


@dataclass
class StepResult:
    index: int
    op: str
    risk: str
    columns: list[str]
    rationale: str
    code: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    new_missing: dict[str, int] = field(default_factory=dict)
    description: str = ""  # what the operation does, in plain words
    changes: str = ""  # what it actually changed in this data, with examples


MAX_EXAMPLES = 3
MAX_COLUMNS_LISTED = 6


def _show(value) -> str:
    try:
        if pd.isna(value):
            return "(missing)"
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d" if value == value.normalize() else "%Y-%m-%d %H:%M")
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        return str(int(number)) if number == int(number) and abs(number) < 1e15 else f"{number:g}"
    text = str(value).replace(chr(10), " ").replace(chr(13), " ")  # spaces stay visible
    return f"'{text[:40]}…'" if len(text) > 40 else f"'{text}'"


def _plain(value) -> str | None:
    """A value as it reads, so '5' and 5.0 count as the same."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return _show(value).strip("'")


def _count(n: int, unit: str) -> str:
    return f"{n:,} {unit if n != 1 else unit.removesuffix('s')}"


def _aligned(before: pd.DataFrame, after: pd.DataFrame):
    """Rows of `before` and `after` that are the same item, or None if that can't be told."""
    by_path = "path" in before.columns and "path" in after.columns
    if by_path and before["path"].is_unique and after["path"].is_unique:
        b = before.set_index("path", drop=False)
        a = after.set_index("path", drop=False)
        common = a.index.intersection(b.index)
        return b.loc[common], a.loc[common]
    renumbered = len(after) < len(before) and after.index.equals(pd.RangeIndex(len(after)))
    if before.index.is_unique and after.index.isin(before.index).all() and not renumbered:
        if len(after) == len(before) and not before.index.equals(after.index):
            return None  # reordered
        return before.loc[after.index], after
    return None


def describe_change(before: pd.DataFrame, after: pd.DataFrame, unit: str = "rows") -> str:
    """What one cleaning step changed, in plain words with a few examples."""
    parts = []
    removed = len(before) - len(after)
    if removed > 0:
        examples = ""
        if "path" in before.columns and "path" in after.columns:
            gone = [x for x in before["path"] if x not in set(after["path"])][:MAX_EXAMPLES]
            more = ", ..." if removed > len(gone) else ""
            examples = f" ({', '.join(str(g) for g in gone)}{more})"
        parts.append(f"removed {_count(removed, unit)}{examples}")
    elif removed < 0:
        parts.append(f"added {_count(-removed, unit)}")
    new_cols = [c for c in after.columns if c not in before.columns]
    gone_cols = [c for c in before.columns if c not in after.columns]
    renamed = []
    if len(before.columns) == len(after.columns) and len(before) == len(after):
        for old, new in zip(before.columns, after.columns, strict=True):
            if old != new and old in gone_cols and new in new_cols:
                renamed.append(f"{old} -> {new}")
                gone_cols.remove(old)
                new_cols.remove(new)
    if renamed:
        more = f" and {len(renamed) - 4} more" if len(renamed) > 4 else ""
        parts.append(f"renamed columns {', '.join(renamed[:4])}{more}")
    if gone_cols:
        parts.append("removed column(s) " + ", ".join(map(str, gone_cols[:MAX_COLUMNS_LISTED])))
    pair = _aligned(before, after)
    for col in new_cols[:MAX_COLUMNS_LISTED]:
        values = after[col]
        marked = int((values == True).sum()) if values.dtype == bool else int(values.notna().sum())  # noqa: E712
        parts.append(f"marked {_count(marked, unit)} in a new '{col}' column")
    if pair is not None:
        b_all, a_all = pair
        listed = 0
        for col in [c for c in after.columns if c in before.columns]:
            b, a = b_all[col], a_all[col]
            b_missing, a_missing = b.isna().to_numpy(), a.isna().to_numpy()
            same = (b_missing & a_missing) | (
                ~b_missing & ~a_missing & (b.astype(str).to_numpy() == a.astype(str).to_numpy())
            )
            changed = ~same
            if changed.any():  # a value only retyped (the text '5' to the number 5) is the same
                idx = np.flatnonzero(changed)
                pairs_ = zip(b.iloc[idx], a.iloc[idx], strict=True)
                keep = np.array([_plain(x) != _plain(y) for x, y in pairs_], dtype=bool)
                changed[idx[~keep]] = False
            n = int(changed.sum())
            was_text = not pd.api.types.is_numeric_dtype(before[col]) and not (
                pd.api.types.is_datetime64_any_dtype(before[col])
            )
            if was_text and pd.api.types.is_bool_dtype(after[col]):
                parts.append(f"converted '{col}' to true/false")
            elif was_text and pd.api.types.is_numeric_dtype(after[col]):
                parts.append(f"converted '{col}' to numbers")
            elif was_text and pd.api.types.is_datetime64_any_dtype(after[col]):
                parts.append(f"converted '{col}' to dates")
            if not n:
                continue
            listed += 1
            if listed > MAX_COLUMNS_LISTED:
                parts.append("and more columns")
                break
            if pd.api.types.is_bool_dtype(after[col]) and bool(a[changed].all()):
                parts.append(f"marked {_count(n, unit)}: {col.replace('_', ' ')}")  # a flag
                continue
            emptied = int((changed & a_missing).sum())
            pairs = []
            for x, y in zip(b[changed], a[changed], strict=True):
                shown = f"{_show(x)} → {_show(y)}"
                if shown not in pairs:
                    pairs.append(shown)
                if len(pairs) == MAX_EXAMPLES:
                    break
            note = (
                f", {emptied:,} of them now missing"
                if emptied and emptied < n
                else (", all now missing" if emptied else "")
            )
            parts.append(
                f"changed {_count(n, 'values')} in '{col}'{note} (e.g. {'; '.join(pairs)})"
            )
    if not parts:
        return "Nothing needed changing: the data already met this rule."
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


@dataclass
class PlanRun:
    df: pd.DataFrame
    steps: list[StepResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def execute_plan(
    plan: CleaningPlan,
    df: pd.DataFrame,
    store: EvidenceStore | None = None,
    *,
    max_row_loss: float = MAX_ROW_LOSS,
    max_new_missing: float = MAX_NEW_MISSING,
    catalog: dict | None = None,
    namespace: dict | None = None,
    unit: str = "rows",
) -> PlanRun:
    """Run the plan on a copy, step by step, collecting every problem with a fix hint.

    Tables use the default catalog; other data types pass their own catalog, helper
    namespace, and unit (images work on a table of files, one row per image).
    """
    catalog = catalog or OPS
    if namespace is not None and "path" in df.columns:  # file-based data: see _CHECK_FILES
        namespace = {**namespace, "ORIGINAL_PATHS": set(df["path"])}
    work = df.copy()
    run = PlanRun(df=work)
    start_rows = len(df)
    for i, op in enumerate(plan.ops, 1):
        target = f" on {op.columns}" if op.columns else ("" if catalog is not OPS else " on all")
        label = f"Step {i} ({op.op}{target})"
        try:
            code = op_code(op, list(work.columns), catalog)
        except ValueError as exc:
            run.errors.append(f"{label}: {exc}")
            continue
        spec = catalog[op.op]
        if spec.risk == Risk.DESTRUCTIVE and not op.evidence:
            run.errors.append(
                f"{label}: removes data, so cite the evidence artifact that justifies it."
            )
            continue
        if store is not None:
            unknown = [e for e in op.evidence if e not in store]
            if unknown:
                run.errors.append(f"{label}: cites evidence IDs that don't exist: {unknown}.")
                continue
        before = work
        try:
            after = run_code(code, before.copy(), namespace)
        except Exception as exc:
            text = str(exc)
            hint = ""
            if "string dtype" in text or "could not convert" in text:
                hint = (
                    " The column is still text: convert it first (cast_numeric, "
                    "strip_currency, or parse_percent)."
                )
            run.errors.append(f"{label} failed: {type(exc).__name__}: {text[:200]}.{hint}")
            continue
        new_missing: dict[str, int] = {}
        if op.op in CONVERSIONS and catalog is OPS:
            for col in op.columns or []:
                present = ~pipeline_helpers.is_missing(before[col])
                lost = int((present & after[col].isna()).sum())
                if lost:
                    new_missing[col] = lost
                    share = lost / max(int(present.sum()), 1)
                    if share > max_new_missing:
                        examples = (
                            before.loc[present & after[col].isna(), col].astype(str).unique()[:5]
                        )
                        run.errors.append(
                            f"{label}: would turn {lost} present values ({share:.0%}) of '{col}' "
                            "into "
                            f"missing, e.g. {list(examples)}. Check the format (decimal_comma? "
                            f"different operation?) or leave this column as it is."
                        )
        run.steps.append(
            StepResult(
                index=i,
                op=op.op,
                risk=str(spec.risk),
                columns=list(op.columns),
                rationale=op.rationale,
                code=code,
                rows_before=len(before),
                rows_after=len(after),
                cols_before=before.shape[1],
                cols_after=after.shape[1],
                new_missing=new_missing,
                description=spec.description,
                changes=describe_change(before, after, unit),
            )
        )
        work = after
    loss = 1 - len(work) / max(start_rows, 1)
    if loss > max_row_loss:
        removed = [
            (s, s.rows_before - s.rows_after) for s in run.steps if s.rows_after < s.rows_before
        ]
        culprit = (
            " Removed by step: "
            + ", ".join(f"step {s.index} ({s.op}) {n}" for s, n in removed)
            + ". Check the parameters of the biggest ones, or drop the least important step."
            if removed
            else ""
        )
        run.errors.append(
            f"The plan removes {loss:.0%} of {unit} ({start_rows - len(work)} of {start_rows}), "
            f"over the {max_row_loss:.0%} limit.{culprit} Use less destructive operations."
        )
    run.df = work
    return run


def helpers_source(module=pipeline_helpers) -> str:
    """A helper module's code without its docstring and top-level imports, for export."""
    source = inspect.getsource(module)
    body = source.split('"""', 2)[-1]  # drop the module docstring
    return re.sub(r"^(import|from) .*\n", "", body, flags=re.M).strip()


def _load_code(table: LoadedTable) -> str:
    if table.format in ("xlsx", "xls"):
        return (
            f"    df = pd.read_excel(path, sheet_name={table.sheet!r}, header={table.header_row}, "
            f"dtype=str)\n    return df.fillna('')"
        )
    if table.format == "log":
        return "    return parse_log_lines(path)"
    if table.format == "jsonl":
        return "    df = pd.read_json(path, lines=True, dtype=False)\n    return df.astype(str)"
    return (
        f"    return pd.read_csv(path, sep={table.delimiter!r}, dtype=str, keep_default_na=False,\n"
        f"                       skiprows={table.header_row}, "
        f"encoding={table.encoding or 'utf-8'!r})"
    )


def pipeline_script(plan: CleaningPlan, run: PlanRun, table: LoadedTable) -> str:
    steps = []
    for s in run.steps:
        comment = f"    # Step {s.index}: {s.op} [{s.risk}] - {s.rationale}".replace("\n", " ")
        steps.append(comment + "\n" + "\n".join("    " + ln for ln in s.code.splitlines()))
    body = "\n\n".join(steps) or "    pass"
    return f'''"""cleaning_pipeline.py - generated by MOSAIC EDA on {time.strftime("%Y-%m-%d")}.

Source: {table.source}
Plan: {plan.summary}

Rerun it on the full dataset:
    python cleaning_pipeline.py input_file cleaned.csv
"""

import re
import sys

import numpy as np
import pandas as pd

{helpers_source()}

{helpers_source(injection)}


def load(path):
{_load_code(table)}


def clean(df):
{body}
    return df


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else {table.source!r}
    target = sys.argv[2] if len(sys.argv) > 2 else "cleaned.csv"
    clean(load(source)).to_csv(target, index=False)
    print("Saved", target)
'''


def neutralize_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Stop spreadsheet apps from running cells like '=HYPERLINK(...)' in exported files."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            text = out[col].astype("str")
            risky = out[col].notna() & text.str.startswith(FORMULA_START)
            out[col] = out[col].where(~risky, "'" + text)
    return out


def export_clean(df: pd.DataFrame, path: Path) -> Path:
    neutralize_formulas(df).to_csv(path, index=False)
    return path


# ---- checks after cleaning (self-correction level 3) ----

VALUE_CHANGING = {"winsorize", "impute_median", "impute_mode", "impute_constant"}
MAX_KS = 0.2


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two ECDFs."""
    a, b = np.sort(a), np.sort(b)
    if not len(a) or not len(b):
        return 0.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def distribution_shifts(raw: pd.DataFrame, run: PlanRun, max_ks: float = MAX_KS) -> list[str]:
    """Flag value-changing steps that distort a numeric column's distribution."""
    problems = []
    for step in run.steps:
        if step.op not in VALUE_CHANGING:
            continue
        for col in step.columns:
            if col not in raw.columns or col not in run.df.columns:
                continue
            after = pd.to_numeric(run.df[col], errors="coerce").dropna().to_numpy(float)
            before = pipeline_helpers.to_number(raw[col]).dropna().to_numpy(float)
            if len(before) < 20 or len(after) < 20:
                continue
            ks = ks_statistic(before, after)
            if ks > max_ks:
                problems.append(
                    f"Step {step.index} ({step.op} on '{col}') shifts its distribution a lot "
                    f"(KS = {ks:.2f}, limit {max_ks}). Use a milder option (flag_outliers, "
                    "wider winsorize quantiles, add_missing_indicator) or leave it."
                )
    return problems


def conservative_plan(types: dict) -> CleaningPlan:
    """A safe-only plan built from the detected types: used when the strategist's plans fail."""
    from mosaic.tables.ops import CleaningOp  # local import keeps the module order simple

    by_kind: dict[str, list[str]] = {}
    for col, ctype in types.items():
        by_kind.setdefault(ctype.kind, []).append(col)
    ops = [
        CleaningOp(
            op="standardize_null_tokens",
            rationale="Turn missing-value tokens into real missing values.",
        )
    ]
    mapping = {
        "currency": "strip_currency",
        "percent": "parse_percent",
        "datetime": "parse_dates",
        "boolean": "parse_boolean",
        "id": "mark_as_id",
    }
    for kind, op in mapping.items():
        if by_kind.get(kind):
            ops.append(
                CleaningOp(op=op, columns=by_kind[kind], rationale=f"Detected {kind} columns.")
            )
    numeric = by_kind.get("numeric", [])
    for decimal_comma in (False, True):
        cols = [c for c in numeric if ("decimal comma" in types[c].detail) == decimal_comma]
        if cols:
            ops.append(
                CleaningOp(
                    op="cast_numeric",
                    columns=cols,
                    params={"decimal_comma": decimal_comma},
                    rationale="Numbers stored as text.",
                )
            )
    if by_kind.get("empty"):
        ops.append(CleaningOp(op="drop_empty_columns", rationale="Columns with no values."))
    return CleaningPlan(
        summary="Safe-only fallback plan: fix missing-value tokens and types, "
        "drop empty columns. Nothing else is changed.",
        ops=ops,
    )
