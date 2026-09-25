"""Structured outputs the agents must return (checked by Pydantic and by guardrails)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TriageBrief(BaseModel):
    dataset_description: str = Field(
        description="What this dataset appears to be, in 1-2 sentences"
    )
    focus_areas: list[str] = Field(description="The 3-5 things the analysis should focus on")
    target_column: str | None = Field(
        default=None, description="Column the user wants to predict, if any (exact name)"
    )


class Metric(BaseModel):
    key: str = Field(description="Evidence key, for example 'revenue.skew' or 'duplicate_rows'")
    value: float


class Finding(BaseModel):
    title: str = Field(description="Short headline, under 10 words")
    statement: str = Field(description="One or two sentences stating the finding with its numbers")
    severity: Literal["info", "warning", "critical"]
    evidence: list[str] = Field(description="Artifact IDs that support this finding")
    claimed_metrics: list[Metric] = Field(
        default_factory=list,
        description="Every number used in the statement, with its evidence key "
        "(for example {'key': 'revenue.skew', 'value': 19.21})",
    )
    recommendation: str = Field(default="", description="What the user should do about it")


class FindingsReport(BaseModel):
    findings: list[Finding]


class ReportNarrative(BaseModel):
    headline: str = Field(description="One-line verdict on the dataset")
    executive_summary: str = Field(description="3-5 sentences for a busy data scientist")
    next_steps: list[str] = Field(description="3-5 concrete next steps")
