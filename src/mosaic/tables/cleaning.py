"""Validate, dry-run, and apply cleaning plans; export the pipeline script and cleaned data."""

from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mosaic.evidence.store import EvidenceStore
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
) -> PlanRun:
    """Run the plan on a copy, step by step, collecting every problem with a fix hint."""
    work = df.copy()
    run = PlanRun(df=work)
    start_rows = len(df)
    for i, op in enumerate(plan.ops, 1):
        label = f"Step {i} ({op.op} on {op.columns or 'all columns'})"
        try:
            code = op_code(op, list(work.columns))
        except ValueError as exc:
            run.errors.append(f"{label}: {exc}")
            continue
        spec = OPS[op.op]
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
            after = run_code(code, before.copy())
        except Exception as exc:
            run.errors.append(f"{label} failed: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        new_missing: dict[str, int] = {}
        if op.op in CONVERSIONS:
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
            )
        )
        work = after
    loss = 1 - len(work) / max(start_rows, 1)
    if loss > max_row_loss:
        run.errors.append(
            f"The plan removes {loss:.0%} of rows ({start_rows - len(work)} of {start_rows}), "
            f"over the {max_row_loss:.0%} limit. Use less destructive operations."
        )
    run.df = work
    return run


def _helpers_source() -> str:
    source = inspect.getsource(pipeline_helpers)
    body = source.split('"""', 2)[-1]  # drop the module docstring
    return re.sub(r"^(import|from) .*\n", "", body, flags=re.M).strip()


def _load_code(table: LoadedTable) -> str:
    if table.format in ("xlsx", "xls"):
        return (
            f"    df = pd.read_excel(path, sheet_name={table.sheet!r}, header={table.header_row}, "
            f"dtype=str)\n    return df.fillna('')"
        )
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

{_helpers_source()}


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
