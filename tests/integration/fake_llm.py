"""A scripted stand-in for Gemini, so the whole Flow can run offline in tests."""

from __future__ import annotations

import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr


def _text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(m.get("content", "")) for m in messages)


class ScriptedLLM(BaseLLM):
    """Answers each task type with valid JSON built from the real evidence.

    The first findings answer claims a wrong number, so the fact-check guardrail
    must reject it before the corrected answer is accepted.
    """

    _state: dict = PrivateAttr(default_factory=dict)

    def bind(self, state: dict) -> ScriptedLLM:
        self._state = state
        return self

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        text = _text(messages)
        self._state.setdefault("calls", []).append(text[:80])
        if "Allowed operations" in text:
            answer = self._plan(text)
        elif "Write 5-8 findings" in text:
            answer = self._findings(text)
        elif "Write the report summary" in text:
            answer = {
                "headline": "Usable after cleaning, with one leak to remove",
                "executive_summary": "The table was cleaned and profiled.",
                "next_steps": ["Drop the leaky column before modeling."],
            }
        else:
            answer = {
                "dataset_description": "Sales orders with a churn flag.",
                "focus_areas": ["types", "missing values", "leakage"],
                "target_column": "churned",
            }
        return json.dumps(answer)

    def _plan(self, text: str) -> dict:
        return {
            "summary": "Fix tokens and types, flag outliers, remove duplicates.",
            "ops": [
                {"op": "standardize_null_tokens", "rationale": "Hidden missing tokens."},
                {
                    "op": "parse_dates",
                    "columns": ["order_date"],
                    "rationale": "Mixed date formats.",
                },
                {
                    "op": "strip_currency",
                    "columns": ["unit_price"],
                    "rationale": "Currency strings.",
                },
                {"op": "parse_percent", "columns": ["discount"], "rationale": "Percent strings."},
                {
                    "op": "cast_numeric",
                    "columns": ["quantity", "revenue", "refund_amount"],
                    "rationale": "Numbers stored as text.",
                },
                {"op": "parse_boolean", "columns": ["churned"], "rationale": "yes/no target."},
                {"op": "mark_as_id", "columns": ["order_id"], "rationale": "Identifier."},
                {
                    "op": "drop_exact_duplicates",
                    "rationale": "Exact duplicate rows.",
                    "evidence": ["tbl_overview_001"],
                },
                {
                    "op": "flag_outliers",
                    "columns": ["revenue"],
                    "rationale": "Extreme revenue values.",
                },
            ],
        }

    def _findings(self, text: str) -> dict:
        skew = float(re.search(r"revenue: numeric;.*?skew (-?[\d.]+)", text).group(1))
        dup = float(re.search(r"duplicate rows \(([\d.]+)%\)", text).group(1))
        first = not self._state.get("findings_attempted")
        self._state["findings_attempted"] = True
        stubborn = self._state.get("stubborn_analyst", False)
        claimed_skew = skew + 5 if (first or stubborn) else skew  # wrong on the first attempt
        return {
            "findings": [
                {
                    "title": "Revenue is heavily right-skewed",
                    "severity": "warning",
                    "statement": f"Revenue has a skew of {claimed_skew}.",
                    "evidence": ["tbl_columns_001"],
                    "claimed_metrics": [{"key": "revenue.skew", "value": claimed_skew}],
                    "recommendation": "Use a log scale.",
                },
                {
                    "title": "Duplicate rows",
                    "severity": "warning",
                    "statement": f"{dup}% of rows were exact duplicates.",
                    "evidence": ["tbl_overview_001"],
                    "claimed_metrics": [{"key": "duplicate_pct", "value": dup}],
                },
                {
                    "title": "Refund amount leaks the target",
                    "severity": "critical",
                    "statement": "refund_amount separates churned customers almost perfectly.",
                    "evidence": ["tbl_target_001"],
                    "claimed_metrics": [],
                },
            ]
        }
