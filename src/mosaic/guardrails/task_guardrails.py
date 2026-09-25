"""CrewAI task guardrails. Each returns (True, output) or (False, a specific fix request).

The message on failure goes back to the agent, which then retries with that feedback.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel, ValidationError

from mosaic.events.reporter import RunReporter
from mosaic.evidence.store import EvidenceStore
from mosaic.models.agent_outputs import FindingsReport, ReviewVerdict, TriageBrief
from mosaic.tables.cleaning import PlanRun, distribution_shifts, execute_plan
from mosaic.tables.ops import CleaningPlan

Guardrail = Callable[[TaskOutput], tuple[bool, Any]]
REL_TOL = 0.01
ABS_TOL = 0.06  # allows rounding to one decimal place
MIN_FINDINGS = 3
NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
CAUSAL = re.compile(r"\b(causes?|caused by|because of|leads? to|results? in|drives?)\b", re.I)


@dataclass
class GuardContext:
    store: EvidenceStore
    reporter: RunReporter
    df: pd.DataFrame | None = None
    columns: list[str] = field(default_factory=list)
    plan_run: PlanRun | None = None
    verified: int = 0
    passing: list[dict] = field(default_factory=list)  # findings that passed on the last attempt
    passing_verified: int = 0


def parse_output(output: TaskOutput, model: type[BaseModel]) -> BaseModel:
    if isinstance(output.pydantic, model):
        return output.pydantic
    raw = output.raw or ""
    match = re.search(r"\{.*\}", raw, re.S)
    return model.model_validate_json(match.group(0) if match else raw)


def close(claimed: float, actual: Any) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return abs(claimed - actual) <= max(REL_TOL * abs(actual), ABS_TOL)


def statement_numbers(text: str) -> list[float]:
    """Numbers a reader would take as facts. Small counts (under 10) and years are exempt."""
    out = []
    for token in NUMBER.findall(text):
        plain = token.rstrip("%").replace(",", "")
        value = float(plain)
        is_int = "." not in plain and not token.endswith("%")
        if is_int and (abs(value) < 10 or 1900 <= value <= 2100):
            continue
        out.append(value)
    return out


def _reject(ctx: GuardContext, title: str, problems: list[str]) -> tuple[bool, str]:
    ctx.reporter.emit("guardrail", title, "\n".join(problems[:6]), "rejected")
    return False, (
        "Your answer was rejected by an automatic check:\n- "
        + "\n- ".join(problems)
        + "\nFix every point and return the complete corrected answer in the same JSON format."
    )


def triage_guardrail(ctx: GuardContext) -> Guardrail:
    def check(output: TaskOutput):  # CrewAI rejects string annotations here
        try:
            brief = parse_output(output, TriageBrief)
        except (ValidationError, ValueError) as exc:
            return _reject(ctx, "Triage output isn't valid", [f"Invalid JSON: {str(exc)[:300]}"])
        if brief.target_column and brief.target_column not in ctx.columns:
            return _reject(
                ctx,
                "Triage named a column that doesn't exist",
                [
                    f"target_column '{brief.target_column}' isn't a column. "
                    f"Use an exact name from {ctx.columns} or null."
                ],
            )
        return True, output

    return check


def plan_guardrail(ctx: GuardContext) -> Guardrail:
    def check(output: TaskOutput):  # CrewAI rejects string annotations here
        try:
            plan = parse_output(output, CleaningPlan)
        except (ValidationError, ValueError) as exc:
            return _reject(ctx, "Cleaning plan isn't valid", [f"Invalid JSON: {str(exc)[:300]}"])
        if not plan.ops:
            return _reject(ctx, "Cleaning plan is empty", ["Propose at least one operation."])
        run = execute_plan(plan, ctx.df, ctx.store)
        if not run.ok:
            return _reject(ctx, "Cleaning plan failed the dry run", run.errors)
        shifts = distribution_shifts(ctx.df, run)
        if shifts:
            return _reject(ctx, "Cleaning plan distorts the data", shifts)
        ctx.plan_run = run
        ctx.reporter.emit(
            "step",
            "Cleaning plan passed the dry run",
            f"{len(plan.ops)} operations; rows {len(ctx.df):,} -> {len(run.df):,}",
            "done",
        )
        return True, output

    return check


def findings_guardrail(ctx: GuardContext) -> Guardrail:
    def check(output: TaskOutput):  # CrewAI rejects string annotations here
        try:
            report = parse_output(output, FindingsReport)
        except (ValidationError, ValueError) as exc:
            return _reject(ctx, "Findings aren't valid", [f"Invalid JSON: {str(exc)[:300]}"])
        problems: list[str] = []
        verified = 0
        passing: list[dict] = []
        if len(report.findings) < MIN_FINDINGS:
            problems.append(
                f"Return at least {MIN_FINDINGS} findings (you returned {len(report.findings)})."
            )
        for i, f in enumerate(report.findings, 1):
            label = f"Finding {i} ('{f.title}')"
            before, checked_before = len(problems), verified
            if not f.evidence:
                problems.append(f"{label}: cite at least one evidence artifact ID.")
                continue
            unknown = [e for e in f.evidence if e not in ctx.store]
            if unknown:
                problems.append(f"{label}: evidence IDs {unknown} don't exist.")
                continue
            if CAUSAL.search(f.statement):
                problems.append(
                    f"{label}: states a cause ('{CAUSAL.search(f.statement).group(0)}') that the "
                    "evidence can't show. Describe the association instead."
                )
            for metric in f.claimed_metrics:
                key, claimed = metric.key, metric.value
                candidates = ctx.store.lookup_all(f.evidence, key)
                actual = next((c for c in candidates if close(claimed, c)), None)
                if not candidates:
                    problems.append(
                        f"{label}: '{key}' isn't in {f.evidence}. Use '<column>.<stat>' keys that "
                        "appear in the evidence, and cite the artifact that contains them."
                    )
                elif actual is None:
                    shown = " or ".join(str(c) for c in candidates)
                    where = [f"'{k}' in {a}" for a, k in ctx.store.find_value(claimed)]
                    elsewhere = f" ({claimed} is {' or '.join(where)})" if where else ""
                    problems.append(
                        f"{label}: claims {key} = {claimed}, but the cited evidence says "
                        f"{shown}{elsewhere}."
                    )
                else:
                    verified += 1
            claimed_values = [m.value for m in f.claimed_metrics]
            for value in statement_numbers(f.statement):
                if not any(close(value, c) or close(value, round(c, 2)) for c in claimed_values):
                    hint = "Add it under its evidence key, or remove the number."
                    where = ctx.store.find_value(value)
                    if where:
                        spots = " or ".join(f"'{k}' in {a}" for a, k in where)
                        hint = f"It matches {spots}: add that key and cite that artifact."
                    problems.append(
                        f"{label}: the statement uses {value:g}, but claimed_metrics has no "
                        f"matching entry. {hint}"
                    )
            if len(problems) == before:
                passing.append(f.model_dump())
            else:
                verified = checked_before  # only count numbers in findings that pass
        ctx.passing = passing
        ctx.passing_verified = verified
        if problems:
            return _reject(ctx, "Findings failed the fact check", problems)
        ctx.verified = verified
        ctx.reporter.counters["facts_verified"] = verified  # latest accepted set
        ctx.reporter.emit(
            "step",
            "Findings passed the fact check",
            f"{verified} numbers checked against the evidence",
            "done",
        )
        return True, output

    return check


def review_guardrail(ctx: GuardContext, n_findings: Callable[[], int]) -> Guardrail:
    def check(output: TaskOutput):  # CrewAI rejects string annotations here
        try:
            verdict = parse_output(output, ReviewVerdict)
        except (ValidationError, ValueError) as exc:
            return _reject(ctx, "Review isn't valid", [f"Invalid JSON: {str(exc)[:300]}"])
        count = n_findings()
        problems = [
            f"Issue about finding {i.finding}: findings are numbered 1 to {count}."
            for i in verdict.issues
            if not 1 <= i.finding <= count
        ]
        if not verdict.approved and not verdict.issues and not verdict.missed:
            problems.append("You didn't approve, so list the issues or missed problems.")
        if problems:
            return _reject(ctx, "Review output isn't consistent", problems)
        return True, output

    return check


def dump_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(), indent=1, default=str)
