"""The allowed cleaning operations for tables.

Each operation is a pandas code template. Applying a plan executes exactly that
code, so the exported cleaning_pipeline.py is what actually ran. Column names and
parameters are validated and inserted with repr(), so plan contents can't inject code.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic.json_schema import SkipJsonSchema

from mosaic.tables import pipeline_helpers


class Risk(StrEnum):
    SAFE = "safe"  # format or structure only; values keep their meaning
    LOSSY = "lossy"  # values change
    DESTRUCTIVE = "destructive"  # rows or columns are removed
    VIEW = "view"  # affects analysis only, not the exported data


class NoParams(BaseModel):
    model_config = {"extra": "forbid"}


class NumberParams(NoParams):
    decimal_comma: bool = False


class DateParams(NoParams):
    dayfirst: bool = False


class CaseParams(NoParams):
    case: Literal["lower", "title", "upper"] = "title"


class ConstantParams(NoParams):
    value: str | float | int


class WinsorParams(NoParams):
    lower: float = Field(0.01, ge=0, le=0.25)
    upper: float = Field(0.99, ge=0.75, le=1)


class PercentParams(NoParams):
    as_fraction: bool = False


@dataclass(frozen=True)
class OpSpec:
    name: str
    risk: Risk
    description: str
    params: type[BaseModel]
    template: Callable[[list[str], Any], str]  # (columns, params) -> pandas code
    needs_columns: bool = True
    all_columns_ok: bool = False  # an empty column list means "all columns"


def _each(cols: list[str], line: str) -> str:
    return "\n".join(line.format(c=repr(c)) for c in cols)


OPS: dict[str, OpSpec] = {}


def _op(spec: OpSpec) -> None:
    OPS[spec.name] = spec


_op(
    OpSpec(
        "standardize_null_tokens",
        Risk.SAFE,
        "Turn missing-value tokens ('N/A', '-', '?', 'null', '') into real missing values.",
        NoParams,
        lambda c, p: _each(c, "df[{c}] = df[{c}].mask(is_missing(df[{c}]))"),
        all_columns_ok=True,
    )
)
_op(
    OpSpec(
        "trim_whitespace",
        Risk.SAFE,
        "Remove spaces around text values.",
        NoParams,
        lambda c, p: _each(c, "df[{c}] = df[{c}].str.strip()"),
        all_columns_ok=True,
    )
)
_op(
    OpSpec(
        "normalize_case",
        Risk.LOSSY,
        "Make text consistent in case, so 'north', 'North', and 'NORTH' become one value.",
        CaseParams,
        lambda c, p: _each(c, "df[{c}] = df[{c}].str.strip().str.%s()" % p.case),
    )
)
_op(
    OpSpec(
        "strip_currency",
        Risk.SAFE,
        "Convert amounts like '$1,204.50' to numbers.",
        NumberParams,
        lambda c, p: _each(c, "df[{c}] = to_number(df[{c}], decimal_comma=%r)" % p.decimal_comma),
    )
)
_op(
    OpSpec(
        "parse_percent",
        Risk.SAFE,
        "Convert '10%' to 10.0 (or 0.10 with as_fraction).",
        PercentParams,
        lambda c, p: _each(c, "df[{c}] = to_number(df[{c}])" + (" / 100" if p.as_fraction else "")),
    )
)
_op(
    OpSpec(
        "cast_numeric",
        Risk.SAFE,
        "Convert text numbers to numbers (unparseable -> missing).",
        NumberParams,
        lambda c, p: _each(c, "df[{c}] = to_number(df[{c}], decimal_comma=%r)" % p.decimal_comma),
    )
)
_op(
    OpSpec(
        "parse_dates",
        Risk.SAFE,
        "Convert text dates in any common format to dates.",
        DateParams,
        lambda c, p: _each(c, "df[{c}] = to_datetime(df[{c}], dayfirst=%r)" % p.dayfirst),
    )
)
_op(
    OpSpec(
        "parse_boolean",
        Risk.SAFE,
        "Convert yes/no, true/false, 1/0 to booleans.",
        NoParams,
        lambda c, p: _each(c, "df[{c}] = to_bool(df[{c}])"),
    )
)
_op(
    OpSpec(
        "mark_as_id",
        Risk.VIEW,
        "Mark identifier columns so they're left out of statistics.",
        NoParams,
        lambda c, p: "df.attrs.setdefault('id_columns', []).extend(%r)" % (list(c),),
    )
)
_op(
    OpSpec(
        "drop_empty_columns",
        Risk.SAFE,
        "Drop columns with no values at all.",
        NoParams,
        lambda c, p: "df = df.loc[:, ~df.apply(lambda s: is_missing(s).all())]",
        needs_columns=False,
    )
)
_op(
    OpSpec(
        "drop_constant_columns",
        Risk.LOSSY,
        "Drop columns that hold a single value.",
        NoParams,
        lambda c, p: "df = df.loc[:, df.nunique(dropna=True) > 1]",
        needs_columns=False,
    )
)
_op(
    OpSpec(
        "drop_columns",
        Risk.DESTRUCTIVE,
        "Drop specific columns, for example a column that leaks the target.",
        NoParams,
        lambda c, p: "df = df.drop(columns=%r)" % (list(c),),
    )
)
_op(
    OpSpec(
        "drop_exact_duplicates",
        Risk.DESTRUCTIVE,
        "Remove rows that are exact copies.",
        NoParams,
        lambda c, p: "df = df.drop_duplicates().reset_index(drop=True)",
        needs_columns=False,
    )
)
_op(
    OpSpec(
        "drop_rows_missing_in",
        Risk.DESTRUCTIVE,
        "Remove rows where any of these columns is missing.",
        NoParams,
        lambda c, p: "df = df.dropna(subset=%r).reset_index(drop=True)" % (list(c),),
    )
)
_op(
    OpSpec(
        "impute_median",
        Risk.LOSSY,
        "Fill missing numbers with the column median.",
        NoParams,
        lambda c, p: _each(c, "df[{c}] = df[{c}].fillna(df[{c}].median())"),
    )
)
_op(
    OpSpec(
        "impute_mode",
        Risk.LOSSY,
        "Fill missing values with the most common value.",
        NoParams,
        lambda c, p: _each(c, "df[{c}] = df[{c}].fillna(df[{c}].mode(dropna=True).iloc[0])"),
    )
)
_op(
    OpSpec(
        "impute_constant",
        Risk.LOSSY,
        "Fill missing values with a fixed value.",
        ConstantParams,
        lambda c, p: _each(c, "df[{c}] = df[{c}].fillna(%r)" % (p.value,)),
    )
)
_op(
    OpSpec(
        "add_missing_indicator",
        Risk.SAFE,
        "Add a <column>_was_missing flag before imputing, so the information isn't lost.",
        NoParams,
        lambda c, p: "\n".join(f"df[{(col + '_was_missing')!r}] = df[{col!r}].isna()" for col in c),
    )
)
_op(
    OpSpec(
        "winsorize",
        Risk.LOSSY,
        "Cap extreme values at the given quantiles.",
        WinsorParams,
        lambda c, p: _each(
            c,
            "df[{c}] = df[{c}].clip(df[{c}].quantile(%r), df[{c}].quantile(%r))"
            % (p.lower, p.upper),
        ),
    )
)
_op(
    OpSpec(
        "flag_outliers",
        Risk.SAFE,
        "Add a <column>_outlier flag using the 1.5 x IQR rule.",
        NoParams,
        lambda c, p: "\n".join(
            f"_lo, _hi = iqr_bounds(df[{col!r}])\n"
            f"df[{(col + '_outlier')!r}] = (df[{col!r}] < _lo) | (df[{col!r}] > _hi)"
            for col in c
        ),
    )
)
_op(
    OpSpec(
        "standardize_column_names",
        Risk.SAFE,
        "Rename columns to snake_case. Put this last: later steps would need the new names.",
        NoParams,
        lambda c, p: (
            "df.columns = [re.sub(r'[^0-9a-z]+', '_', str(x).lower()).strip('_') "
            "for x in df.columns]"
        ),
        needs_columns=False,
    )
)


class CleaningOp(BaseModel):
    op: str = Field(description="Operation name from the catalog")
    columns: list[str] = Field(default_factory=list, description="Target columns")
    # Gemini's structured output drops free-form dict fields, so agents send parameters as
    # a JSON string; code turns it into `params`
    params_json: str = Field(
        default="{}", description='Parameters as a JSON object string, e.g. {"min_blur": 175.5}'
    )
    params: SkipJsonSchema[dict[str, Any]] = Field(default_factory=dict, exclude=True)
    rationale: str = Field(description="Why, in one sentence, citing the evidence")
    evidence: list[str] = Field(default_factory=list, description="Artifact IDs supporting it")

    @model_validator(mode="before")
    @classmethod
    def _params_from_json(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("params"):  # code and tests may pass a dict directly
            data["params_json"] = json.dumps(data["params"])
            return data
        text = (data.get("params_json") or "{}").strip() or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"params_json isn't valid JSON: {text[:80]}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("params_json must be a JSON object")
        data["params"] = parsed
        return data


class CleaningPlan(BaseModel):
    summary: str = Field(description="One or two sentences on the overall approach")
    ops: list[CleaningOp]


def _param_text(name: str, field: Any) -> str:
    """'min_blur: number (required)' or 'max_distance: integer = 6', readable for agents."""
    kind = getattr(field.annotation, "__name__", str(field.annotation))
    kind = {"float": "number", "int": "integer", "str": "text", "bool": "true/false"}.get(
        kind, kind
    )
    if kind == "list":
        kind = "list of exact file paths from the evidence"
    if field.is_required():
        return f"{name}: {kind} (required)"
    return f"{name}: {kind} = {field.default!r}"


def catalog_text(catalog: dict[str, OpSpec] | None = None) -> str:
    lines = []
    for spec in (catalog or OPS).values():
        fields = ", ".join(_param_text(n, f) for n, f in spec.params.model_fields.items())
        cols = (
            "columns required"
            if spec.needs_columns and not spec.all_columns_ok
            else ("columns optional (empty = all)" if spec.all_columns_ok else "no columns")
        )
        lines.append(
            f"- {spec.name} [{spec.risk}] {spec.description} ({cols}"
            + (f"; params: {fields}" if fields else "")
            + ")"
        )
    return "\n".join(lines)


def op_code(
    op: CleaningOp, columns_now: list[str], catalog: dict[str, OpSpec] | None = None
) -> str:
    """Validate one operation and return its pandas code. Raises ValueError with a fix hint."""
    catalog = catalog or OPS
    spec = catalog.get(op.op)
    if spec is None:
        allowed = ", ".join(catalog)
        raise ValueError(f"'{op.op}' is not an allowed operation. Choose from: {allowed}.")
    try:
        params = spec.params(**op.params)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ValueError(f"'{op.op}' has invalid params ({problems}).") from exc
    cols = list(op.columns)
    if spec.needs_columns and not cols:
        if not spec.all_columns_ok:
            raise ValueError(f"'{op.op}' needs at least one column.")
        cols = list(columns_now)
    missing = [c for c in cols if c not in columns_now]
    if missing:
        raise ValueError(
            f"'{op.op}' refers to column(s) {missing} that don't exist at this step. "
            f"Columns now: {columns_now}."
        )
    return spec.template(cols, params)


def helper_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"pd": pd, "re": re}
    ns.update({k: v for k, v in vars(pipeline_helpers).items() if not k.startswith("_")})
    return ns


def run_code(code: str, df: pd.DataFrame, namespace: dict[str, Any] | None = None) -> pd.DataFrame:
    namespace = dict(namespace or helper_namespace())
    namespace["df"] = df
    exec(compile(code, "<cleaning-op>", "exec"), namespace)  # templates only; see module docstring
    return namespace["df"]
