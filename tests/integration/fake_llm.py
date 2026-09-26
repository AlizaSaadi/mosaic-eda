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
        if "The reviewer asked for these changes" in text:
            answer = self._revise(text)
        elif "Check every finding against this list" in text:
            answer = self._review()
        elif "Allowed operations" in text:
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
            answer = self._triage()
        return json.dumps(answer)

    def _triage(self) -> dict:
        return {
            "dataset_description": "Sales orders with a churn flag.",
            "focus_areas": ["types", "missing values", "leakage"],
            "target_column": "churned",
        }

    def _review(self) -> dict:
        n = self._state.get("reviews", 0) + 1
        self._state["reviews"] = n
        stubborn = self._state.get("stubborn_reviewer", False)
        if n == 1 or stubborn:
            return {
                "approved": False,
                "missed": [],
                "issues": [
                    {
                        "finding": 1,
                        "kind": "severity",
                        "blocking": True,
                        "problem": "Skew alone isn't a warning-level problem for this goal.",
                        "fix_request": "Lower the severity to info.",
                    }
                ],
            }
        return {"approved": True, "issues": [], "missed": []}

    def _revise(self, text: str) -> dict:
        self._state["revisions"] = self._state.get("revisions", 0) + 1
        self._state["findings_attempted"] = True  # revisions use the correct numbers
        answer = self._findings(text.replace("The reviewer asked", ""))
        answer["findings"][0]["severity"] = "info"
        return answer

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
        if "revenue: numeric" not in text:  # pragma: no cover - defensive for prompt changes
            raise AssertionError("evidence missing from prompt")
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


# ---- image datasets ----


def _num(pattern: str, text: str) -> float:
    return float(re.search(pattern, text).group(1))


class ImageScriptedLLM(ScriptedLLM):
    """Answers the image prompts; review and revision reuse the table behavior."""

    def _triage(self) -> dict:
        return {
            "dataset_description": "Three-class shapes dataset.",
            "focus_areas": ["class balance", "duplicates", "mislabels"],
            "target_column": None,
        }

    def _plan(self, text: str) -> dict:
        blur = _num(r"Blur threshold ([\d.]+)", text)
        return {
            "summary": "Remove unusable and duplicate images; flag suspected mislabels.",
            "ops": [
                {
                    "op": "remove_corrupt",
                    "rationale": "Unreadable files.",
                    "evidence": ["img_overview_001"],
                },
                {
                    "op": "drop_exact_duplicates",
                    "rationale": "Byte-identical copies.",
                    "evidence": ["img_dupes_001"],
                },
                {
                    "op": "drop_cross_class_duplicates",
                    "rationale": "Ambiguous labels.",
                    "evidence": ["img_dupes_001"],
                },
                {
                    "op": "drop_near_duplicates",
                    "rationale": "Resized copies.",
                    "evidence": ["img_dupes_001"],
                },
                {
                    "op": "drop_blurry",
                    "params": {"min_blur": blur},
                    "rationale": "Blurry.",
                    "evidence": ["img_quality_001"],
                },
                {
                    "op": "flag_suspected_mislabels",
                    "params": {"files": ["circles/circle_extra_0.jpg"]},
                    "rationale": "Vision review suspicion.",
                    "evidence": ["img_vision_001"],
                },
                {"op": "fix_exif_orientation", "rationale": "Upright images."},
                {"op": "convert_to_rgb", "rationale": "Consistent mode."},
            ],
        }

    def _findings(self, text: str) -> dict:
        ratio = _num(r"largest/smallest ratio ([\d.]+)", text)
        share = _num(r"triangles \d+ \(([\d.]+)%\)", text)
        copies = _num(r"with (\d+) extra copies \(exact_extra_copies\)", text)
        first = not self._state.get("findings_attempted")
        self._state["findings_attempted"] = True
        claimed = round(ratio + 3, 2) if first else round(ratio, 2)  # wrong on the first try
        return {
            "findings": [
                {
                    "title": "Revenue is heavily right-skewed",
                    "severity": "warning",
                    "statement": f"The largest class is {claimed} times the smallest.",
                    "evidence": ["img_balance_001"],
                    "claimed_metrics": [{"key": "imbalance_ratio", "value": claimed}],
                },
                {
                    "title": "Triangles are under-represented",
                    "severity": "warning",
                    "statement": f"Triangles are {round(share, 2)}% of images.",
                    "evidence": ["img_balance_001"],
                    "claimed_metrics": [
                        {"key": "classes.triangles.share", "value": round(share, 2)}
                    ],
                },
                {
                    "title": "Duplicate copies",
                    "severity": "warning",
                    "statement": f"There are {int(copies)} exact extra copies.",
                    "evidence": ["img_dupes_001"],
                    "claimed_metrics": [{"key": "exact_extra_copies", "value": copies}],
                },
            ]
        }


class FakeVisionModels:
    def generate_content(self, model, contents, config):
        from types import SimpleNamespace

        from mosaic.images.vision import SheetReview, VisionReport

        sheets = re.findall(r"Sheet (\d+): class '([^']+)'", contents[0])
        reviews = []
        for number, label in sheets:
            outliers = [23, 24, 25] if (label == "circles" and number == "2") else []
            reviews.append(
                SheetReview(
                    sheet=int(number),
                    description=f"{label} shapes",
                    outliers=outliers,
                    reasons=["a square"] * len(outliers),
                )
            )
        report = VisionReport(sheets=reviews, overall="Three shape classes.")
        return SimpleNamespace(parsed=report, text=report.model_dump_json())


class FakeVisionClient:
    def __init__(self, api_key=None):
        self.models = FakeVisionModels()


# ---- audio datasets ----


class AudioScriptedLLM(ScriptedLLM):
    """Answers the audio prompts; review and revision reuse the table behavior."""

    def _triage(self) -> dict:
        return {
            "dataset_description": "Spoken commands: yes, no, stop.",
            "focus_areas": ["label mismatches", "silence", "duplicates"],
            "target_column": None,
        }

    def _plan(self, text: str) -> dict:
        return {
            "summary": "Remove unusable and duplicate clips; flag mislabels; make formats uniform.",
            "ops": [
                {
                    "op": "remove_corrupt",
                    "rationale": "Undecodable.",
                    "evidence": ["aud_overview_001"],
                },
                {
                    "op": "drop_exact_duplicates",
                    "rationale": "Copies.",
                    "evidence": ["aud_dupes_001"],
                },
                {
                    "op": "drop_near_duplicates",
                    "rationale": "Re-encoded copies.",
                    "evidence": ["aud_dupes_001"],
                },
                {
                    "op": "drop_silent",
                    "rationale": "Silent clips.",
                    "evidence": ["aud_quality_001"],
                },
                {
                    "op": "flag_suspected_mislabels",
                    "params": {"files": ["stop/stop_005.wav", "yes/yes_012.wav"]},
                    "rationale": "Transcript doesn't match label.",
                    "evidence": ["aud_transcripts_001"],
                },
                {"op": "to_mono", "rationale": "One stereo file."},
                {"op": "resample", "params": {"sample_rate": 16000}, "rationale": "Mixed rates."},
                {"op": "convert_to_wav", "rationale": "One format."},
            ],
        }

    def _findings(self, text: str) -> dict:
        mismatches = _num(r"in (\d+) clips \(label_mismatches", text)
        ratio = _num(r"largest/smallest ratio ([\d.]+)", text)
        first = not self._state.get("findings_attempted")
        self._state["findings_attempted"] = True
        claimed = round(ratio + 2, 2) if first else round(ratio, 2)
        return {
            "findings": [
                {
                    "title": "Revenue is heavily right-skewed",
                    "severity": "warning",
                    "statement": f"The largest class is {claimed} times the smallest.",
                    "evidence": ["aud_balance_001"],
                    "claimed_metrics": [{"key": "imbalance_ratio", "value": claimed}],
                },
                {
                    "title": "Clips that don't say their label",
                    "severity": "critical",
                    "statement": f"Transcripts of {int(mismatches)} clips don't match their label.",
                    "evidence": ["aud_transcripts_001"],
                    "claimed_metrics": [{"key": "label_mismatches", "value": mismatches}],
                },
                {
                    "title": "One copy of a clip",
                    "severity": "info",
                    "statement": "An exact duplicate clip was found.",
                    "evidence": ["aud_dupes_001"],
                    "claimed_metrics": [],
                },
            ]
        }


class FakeListenModels:
    def generate_content(self, model, contents, config):
        from types import SimpleNamespace

        from mosaic.audio.listen import ClipReview, ListenReport

        count = len(re.findall(r"Clip (\d+):", contents[0]))
        report = ListenReport(
            clips=[
                ClipReview(
                    clip=n,
                    description="a short spoken word",
                    speech=True,
                    speakers=1,
                    background="quiet",
                )
                for n in range(1, count + 1)
            ],
            overall="Short spoken commands.",
        )
        return SimpleNamespace(parsed=report, text=report.model_dump_json())


class FakeListenClient:
    def __init__(self, api_key=None):
        self.models = FakeListenModels()


# ---- text datasets ----

TEXT_MISLABELS = [
    "billing/ticket_bill_052.txt",
    "shipping/ticket_ship_047.txt",
    "technical/ticket_tech_033.txt",
]


class TextScriptedLLM(ScriptedLLM):
    """Answers the text prompts; review and revision reuse the table behavior."""

    def _triage(self) -> dict:
        return {
            "dataset_description": "Support tickets labeled billing, shipping, technical.",
            "focus_areas": ["mislabels", "personal data", "duplicates"],
            "target_column": None,
        }

    def _plan(self, text: str) -> dict:
        return {
            "summary": "Repair text, mask personal data, remove duplicates and empty tickets.",
            "ops": [
                {"op": "fix_encoding", "rationale": "Garbled characters."},
                {"op": "strip_html", "rationale": "Web form markup."},
                {"op": "normalize_whitespace", "rationale": "Tidy spacing."},
                {
                    "op": "strip_boilerplate",
                    "params": {"min_share": 0.3},
                    "rationale": "Disclaimer under many tickets.",
                    "evidence": ["txt_boilerplate_001"],
                },
                {"op": "mask_pii", "rationale": "Emails and cards.", "evidence": ["txt_pii_001"]},
                {
                    "op": "drop_exact_duplicates",
                    "rationale": "Copies.",
                    "evidence": ["txt_dupes_001"],
                },
                {
                    "op": "drop_near_duplicates",
                    "params": {"threshold": 0.8},
                    "rationale": "Edited copies.",
                    "evidence": ["txt_dupes_001"],
                },
                {
                    "op": "drop_short_documents",
                    "params": {"min_words": 5},
                    "rationale": "Empty and one-word tickets.",
                    "evidence": ["txt_quality_001"],
                },
                {
                    "op": "flag_suspected_mislabels",
                    "params": {"documents": TEXT_MISLABELS},
                    "rationale": "They read like another label.",
                    "evidence": ["txt_labels_001"],
                },
            ],
        }

    def _findings(self, text: str) -> dict:
        mismatches = _num(r"(\d+) documents read more like another label", text)
        ratio = _num(r"largest/smallest ratio ([\d.]+)", text)
        pii = _num(r"(\d+) documents \([\d.]+%\) contain personal data", text)
        first = not self._state.get("findings_attempted")
        self._state["findings_attempted"] = True
        claimed = round(ratio + 2, 2) if first else round(ratio, 2)
        return {
            "findings": [
                {
                    "title": "Labels are unevenly sized",
                    "severity": "warning",
                    "statement": f"The largest label is {claimed} times the smallest.",
                    "evidence": ["txt_balance_001"],
                    "claimed_metrics": [{"key": "imbalance_ratio", "value": claimed}],
                },
                {
                    "title": "Some tickets read like another label",
                    "severity": "critical",
                    "statement": f"{int(mismatches)} documents read like another label.",
                    "evidence": ["txt_labels_001"],
                    "claimed_metrics": [{"key": "label_mismatches", "value": mismatches}],
                },
                {
                    "title": "Personal data in tickets",
                    "severity": "warning",
                    "statement": f"{int(pii)} documents contain personal data.",
                    "evidence": ["txt_pii_001"],
                    "claimed_metrics": [{"key": "documents_with_pii", "value": pii}],
                },
            ]
        }


class FakeReadModels:
    def generate_content(self, model, contents, config):
        from types import SimpleNamespace

        from mosaic.text.review import DocReview, ReadReport

        count = len(re.findall(r"Document (\d+) \(label", contents[0]))
        report = ReadReport(
            docs=[
                DocReview(doc=n, summary="a support request", sentiment="negative")
                for n in range(1, count + 1)
            ],
            overall="Short customer support tickets.",
        )
        return SimpleNamespace(parsed=report, text=report.model_dump_json())


class FakeReadClient:
    def __init__(self, api_key=None):
        self.models = FakeReadModels()


def fake_whisper(expected: dict):
    """Returns the known transcript for each clip, in the order the adapter sends them."""
    order = sorted(expected)

    def engine(clips):
        return [(expected[p], "en") for p in order[: len(clips)]]

    return engine, "fake-whisper"
