"""Structured outputs the agents must return (checked by Pydantic and by guardrails)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TriageBrief(BaseModel):
    dataset_description: str = Field(
        description="What this dataset appears to be, in 1-2 sentences"
    )
    focus_areas: list[str] = Field(description="The 3-5 things the analysis should focus on")
    target_column: str | None = Field(
        default=None, description="Column the user wants to predict, if any (exact name)"
    )

    @field_validator("target_column", mode="before")
    @classmethod
    def _null_words_are_null(cls, value: object) -> object:
        # models sometimes write the word "null" instead of JSON null
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a", "na"}:
            return None
        return value


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


class ReviewIssue(BaseModel):
    finding: int = Field(description="1-based number of the finding this is about")
    kind: Literal["severity", "overclaim", "causal", "unsupported", "unclear", "other"]
    problem: str = Field(description="What is wrong, in one sentence")
    fix_request: str = Field(description="Exactly what the analyst should change")
    blocking: bool = Field(
        default=True, description="True if the finding is misleading unless it's fixed"
    )


class ReviewVerdict(BaseModel):
    approved: bool = Field(description="True if the findings can be published as they are")
    issues: list[ReviewIssue] = Field(default_factory=list)
    missed: list[str] = Field(
        default_factory=list,
        description="Important problems in the evidence that no finding covers",
    )
