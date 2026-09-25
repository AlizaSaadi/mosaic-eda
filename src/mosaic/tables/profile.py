"""Profile a table in code and save every result to the evidence store.

Agents never compute statistics. They read the one-line summaries produced here
and cite the artifact IDs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mosaic.evidence.store import EvidenceStore
from mosaic.tables.load import LoadedTable
from mosaic.tables.semantics import (
    ColumnType,
    detect_type,
    hidden_missing_count,
    missing_mask,
    parse_dates,
    parse_numbers,
)
from mosaic.ui.palette import HARVEST

NUMERIC_KINDS = ("numeric", "currency", "percent")
MAX_HIST = 6
MAX_BARS = 3
TARGET_WORDS = ("target", "label", "class", "outcome", "churn", "churned", "y")


@dataclass
class ProfileResult:
    types: dict[str, ColumnType]
    artifact_ids: list[str] = field(default_factory=list)
    chart_ids: list[str] = field(default_factory=list)
    target: str | None = None
    quality: float = 0.0


def r4(x: float | None) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 4)


def typed_series(series: pd.Series, ctype: ColumnType) -> pd.Series:
    if ctype.kind in NUMERIC_KINDS:
        return parse_numbers(series, decimal_comma="decimal comma" in ctype.detail)
    if ctype.kind == "datetime":
        return parse_dates(series)
    return series.where(~missing_mask(series), None)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def guess_target(goal: str, columns: list[str]) -> str | None:
    if not goal.strip():
        return None
    goal_norm = _norm(goal)
    named = [c for c in columns if len(_norm(c)) >= 2 and _norm(c) in goal_norm]
    if named:
        return max(named, key=lambda c: len(_norm(c)))
    if re.search(r"predict|classif|forecast|target|model", goal, re.I):
        for word in TARGET_WORDS:
            for c in columns:
                if _norm(c) == word:
                    return c
    return None


def _column_stats(raw: pd.Series, ctype: ColumnType) -> dict:
    n = len(raw)
    missing = int(missing_mask(raw).sum())
    present = raw[~missing_mask(raw)]
    stats: dict = {
        "kind": ctype.kind,
        "detail": ctype.detail,
        "parse_rate": r4(ctype.parse_rate),
        "missing": missing,
        "missing_pct": r4(100 * missing / n) if n else 0.0,
        "hidden_missing": hidden_missing_count(raw),
        "unique": int(present.nunique()),
        "unique_pct": r4(100 * present.nunique() / max(len(present), 1)),
    }
    typed = typed_series(raw, ctype)
    if ctype.kind in NUMERIC_KINDS:
        values = typed.dropna()
        stats["unparseable"] = int(len(present) - len(values))
        if len(values):
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
            stats.update(
                mean=r4(values.mean()),
                std=r4(values.std()),
                min=r4(values.min()),
                p25=r4(q1),
                median=r4(values.median()),
                p75=r4(q3),
                max=r4(values.max()),
                skew=r4(values.skew()) if len(values) > 2 else None,
                outliers_iqr=outliers,
                outliers_pct=r4(100 * outliers / len(values)),
                negatives=int((values < 0).sum()),
                zeros=int((values == 0).sum()),
            )
    elif ctype.kind == "datetime":
        values = typed.dropna()
        stats["unparseable"] = int(len(present) - len(values))
        if len(values):
            stats.update(min=str(values.min().date()), max=str(values.max().date()))
    elif ctype.kind in ("category", "boolean", "constant"):
        top = present.str.strip().value_counts().head(8)
        stats["top_values"] = [[str(k), int(v)] for k, v in top.items()]
        variants = present.str.strip().str.lower().nunique()
        if variants < present.str.strip().nunique():
            stats["case_variants"] = int(present.str.strip().nunique() - variants)
    elif ctype.kind == "text":
        lengths = present.str.len()
        stats.update(avg_length=r4(lengths.mean()), max_length=int(lengths.max()))
    return stats


def _summary_line(name: str, s: dict) -> str:
    parts = [f"{name}: {s['kind']}"]
    if s["missing"]:
        parts.append(f"missing {s['missing_pct']}%")
    if s.get("hidden_missing"):
        parts.append(f"{s['hidden_missing']} hidden-missing tokens")
    if s.get("unparseable"):
        parts.append(f"{s['unparseable']} unparseable values")
    if "mean" in s:
        parts.append(f"mean {s['mean']}, median {s['median']}, min {s['min']}, max {s['max']}")
        if s.get("skew") is not None:
            parts.append(f"skew {s['skew']}")
        if s.get("outliers_iqr"):
            parts.append(f"{s['outliers_iqr']} IQR outliers ({s['outliers_pct']}%)")
    if s.get("top_values"):
        top = ", ".join(f"{v} ({c})" for v, c in s["top_values"][:4])
        parts.append(f"top: {top}")
    if s.get("case_variants"):
        parts.append(f"{s['case_variants']} case/spelling variants")
    if "min" in s and s["kind"] == "datetime":
        parts.append(f"range {s['min']} to {s['max']}")
    if s["detail"] and s["kind"] in ("constant", "id", "numeric"):
        parts.append(s["detail"])
    return "; ".join(parts)


def _quality(overview: dict, columns: dict) -> dict:
    ncols = max(len(columns), 1)
    avg_missing = sum(c["missing_pct"] or 0 for c in columns.values()) / ncols
    type_issues = sum(1 for c in columns.values() if c.get("unparseable"))
    dead = sum(1 for c in columns.values() if c["kind"] in ("empty", "constant"))
    hidden = sum(1 for c in columns.values() if c.get("hidden_missing"))
    parts = {
        "missing": -min(30.0, avg_missing * 0.6),
        "duplicates": -min(15.0, overview["duplicate_pct"] * 1.5),
        "type_issues": -min(20.0, type_issues * 4.0),
        "empty_or_constant": -min(10.0, dead * 3.0),
        "hidden_missing": -min(10.0, hidden * 2.0),
    }
    score = max(0.0, 100.0 + sum(parts.values()))
    return {"score": round(score, 1), "parts": {k: round(v, 1) for k, v in parts.items()}}


def _fig(fig: go.Figure, title: str) -> dict:
    fig.update_layout(
        title=title,
        template="plotly_white",
        colorway=HARVEST["chart"],
        font={"family": "Inter, system-ui, sans-serif", "color": HARVEST["text"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        height=340,
    )
    return json.loads(fig.to_json())


def _charts(df: pd.DataFrame, types: dict[str, ColumnType], columns: dict, store, stage):
    ids = []
    missing = {c: s["missing_pct"] for c, s in columns.items() if s["missing_pct"]}
    if missing:
        order = sorted(missing, key=missing.get, reverse=True)[:20]
        fig = go.Figure(
            go.Bar(x=order, y=[missing[c] for c in order], marker_color=HARVEST["accent"])
        )
        fig.update_yaxes(title="% missing")
        ids.append(
            store.add(
                "chart",
                "chart",
                "missing_bar",
                f"Chart: missing values by column ({stage})",
                {"figure": _fig(fig, "Missing values by column")},
            ).id
        )
    numeric = [c for c, t in types.items() if t.kind in NUMERIC_KINDS][:MAX_HIST]
    for col in numeric:
        values = typed_series(df[col], types[col]).dropna()
        if len(values) < 2:
            continue
        counts, edges = np.histogram(values, bins=min(30, max(5, int(np.sqrt(len(values))))))
        centers = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure(
            go.Bar(x=centers, y=counts, width=np.diff(edges), marker_color=HARVEST["chart"][1])
        )
        fig.update_xaxes(title=col)
        fig.update_yaxes(title="count")
        ids.append(
            store.add(
                "chart",
                "chart",
                "histogram",
                f"Chart: distribution of {col} ({stage})",
                {"figure": _fig(fig, f"Distribution of {col}"), "column": col},
            ).id
        )
    cats = [c for c, t in types.items() if t.kind in ("category", "boolean")][:MAX_BARS]
    for col in cats:
        top = columns[col].get("top_values") or []
        if not top:
            continue
        fig = go.Figure(
            go.Bar(x=[v for v, _ in top], y=[n for _, n in top], marker_color=HARVEST["chart"][2])
        )
        ids.append(
            store.add(
                "chart",
                "chart",
                "top_values",
                f"Chart: most common values of {col} ({stage})",
                {"figure": _fig(fig, f"Most common values: {col}"), "column": col},
            ).id
        )
    return ids


def profile_table(
    table: LoadedTable,
    store: EvidenceStore,
    *,
    goal: str = "",
    stage: str = "raw",
    types: dict[str, ColumnType] | None = None,
) -> ProfileResult:
    df = table.df
    types = types or {c: detect_type(df[c], c) for c in df.columns}
    result = ProfileResult(types=types)

    dup = int(df.duplicated().sum())
    overview = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": dup,
        "duplicate_pct": r4(100 * dup / max(len(df), 1)),
        "source": table.source,
        "format": table.format,
        "encoding": table.encoding,
        "delimiter": table.delimiter,
        "sheet": table.sheet,
        "sheets": table.sheets,
        "header_row": table.header_row,
        "total_rows": table.total_rows,
        "notes": table.notes,
        "kinds": {
            k: sum(t.kind == k for t in types.values()) for k in {t.kind for t in types.values()}
        },
    }
    a = store.add(
        "tbl_overview",
        "profile",
        "profile_table",
        f"Table ({stage}): {len(df):,} rows x {len(df.columns)} columns; "
        f"{dup} exact duplicate rows "
        f"({overview['duplicate_pct']}%)."
        + (f" Notes: {' '.join(table.notes)}" if table.notes else ""),
        overview,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    columns = {c: _column_stats(df[c], types[c]) for c in df.columns}
    lines = "\n  ".join(_summary_line(c, s) for c, s in columns.items())
    a = store.add(
        "tbl_columns",
        "profile",
        "profile_columns",
        f"Columns ({stage}):\n  {lines}",
        {"columns": columns},
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    typed = {c: typed_series(df[c], types[c]) for c in df.columns if types[c].kind in NUMERIC_KINDS}
    if len(typed) >= 2:
        frame = pd.DataFrame(typed)
        pearson = frame.corr(method="pearson")
        spearman = frame.corr(method="spearman")
        pairs = []
        cols = list(frame.columns)
        for i, x in enumerate(cols):
            for y in cols[i + 1 :]:
                p, s = pearson.loc[x, y], spearman.loc[x, y]
                if pd.notna(p):
                    pairs.append({"a": x, "b": y, "pearson": r4(p), "spearman": r4(s)})
        pairs.sort(key=lambda d: -abs(d["pearson"]))
        strong = [p for p in pairs if abs(p["pearson"]) >= 0.5]
        text = (
            "; ".join(f"{p['a']}~{p['b']} r={p['pearson']}" for p in strong[:8])
            or "none above |r|=0.5"
        )
        pair_map = {
            f"{p['a']}~{p['b']}": {"pearson": p["pearson"], "spearman": p["spearman"]}
            for p in pairs
        }
        a = store.add(
            "tbl_assoc",
            "profile",
            "correlations",
            f"Numeric correlations ({stage}): {text} (cite as '<a>~<b>.pearson')",
            pair_map,
            {"stage": stage},
        )
        result.artifact_ids.append(a.id)

    target = guess_target(goal, list(df.columns))
    if target and stage == "raw":
        result.target = target
        leak = _leakage(df, types, target)
        a = store.add(
            "tbl_target", "profile", "target_check", leak.pop("_summary"), leak, {"target": target}
        )
        result.artifact_ids.append(a.id)

    quality = _quality(overview, columns)
    result.quality = quality["score"]
    a = store.add(
        "tbl_quality",
        "profile",
        "quality_score",
        f"Data quality score ({stage}): {quality['score']}/100 (deductions: {quality['parts']}).",
        quality,
        {"stage": stage},
    )
    result.artifact_ids.append(a.id)

    result.chart_ids = _charts(df, types, columns, store, stage)
    return result


def single_feature_auc(x: pd.Series, y: pd.Series) -> float | None:
    """ROC AUC of one numeric feature for a binary target (Mann-Whitney U, handles ties)."""
    both = pd.DataFrame({"x": x, "y": y}).dropna()
    classes = sorted(both["y"].unique())
    if len(classes) != 2 or len(both) < 20:
        return None
    ranks = both["x"].rank()
    pos = both["y"] == classes[1]
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return None
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _leakage(df: pd.DataFrame, types: dict[str, ColumnType], target: str) -> dict:
    t_type = types[target]
    t_raw = df[target]
    present = ~missing_mask(t_raw)
    data: dict = {"target": target, "target_kind": t_type.kind, "suspects": []}
    if t_type.kind in ("category", "boolean"):
        counts = t_raw[present].str.strip().value_counts()
        data["class_balance"] = {str(k): r4(100 * v / counts.sum()) for k, v in counts.items()}
    y = typed_series(t_raw, t_type) if t_type.kind in NUMERIC_KINDS else None
    if y is None and t_type.kind in ("category", "boolean"):
        codes, _ = pd.factorize(t_raw.where(present, None))
        y = pd.Series(np.where(codes < 0, np.nan, codes), index=t_raw.index)
    binary = y is not None and y.dropna().nunique() == 2
    for col, ctype in types.items():
        if col == target or ctype.kind in ("empty", "constant", "id", "text"):
            continue
        if ctype.kind in NUMERIC_KINDS and y is not None:
            x = typed_series(df[col], ctype)
            corr = x.corr(y)
            if pd.notna(corr) and abs(corr) >= 0.95:
                data["suspects"].append(
                    {"column": col, "reason": "near-perfect correlation", "value": r4(corr)}
                )
            elif binary:
                auc = single_feature_auc(x, y)
                if auc is not None and max(auc, 1 - auc) >= 0.97:
                    data["suspects"].append(
                        {
                            "column": col,
                            "reason": "separates the target almost perfectly (AUC)",
                            "value": r4(max(auc, 1 - auc)),
                        }
                    )
        elif ctype.kind in ("category", "boolean") and t_type.kind in ("category", "boolean"):
            both = df.loc[present & ~missing_mask(df[col]), [col, target]]
            if both[col].nunique() >= 2 and len(both) >= 20:
                purity = both.groupby(col)[target].agg(
                    lambda s: s.value_counts(normalize=True).iloc[0]
                )
                if purity.min() >= 0.999 and both[col].nunique() < len(both) * 0.5:
                    data["suspects"].append(
                        {
                            "column": col,
                            "reason": "each value maps to one target class",
                            "value": 1.0,
                        }
                    )
    data["leak"] = {
        s["column"]: {"value": s["value"], "reason": s["reason"]} for s in data.pop("suspects")
    }
    names = (
        ", ".join(f"{c} ({v['reason']}, value {v['value']})" for c, v in data["leak"].items())
        or "none found"
    )
    balance = data.get("class_balance")
    data["_summary"] = f"Target '{target}' ({t_type.kind}). Leakage suspects: {names}." + (
        f" Class balance %: {balance}." if balance else ""
    )
    return data
