"""The evaluation's ground truth and scoring (eval/golden.py, eval/evaluate.py)."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from evaluate import score  # noqa: E402
from golden import DATASETS  # noqa: E402


def test_every_golden_dataset_exists_and_every_pattern_compiles():
    assert len(DATASETS) == 10
    for name, spec in DATASETS.items():
        assert spec["path"].exists(), name
        ids = [p["id"] for p in spec["problems"]]
        assert len(ids) == len(set(ids)), name
        for p in spec["problems"]:
            assert p.get("patterns") or p.get("ops") or p.get("check"), (name, p["id"])
            for pattern in p.get("patterns", []):
                re.compile(pattern, re.I)


def test_scoring_credits_findings_cleaning_and_code_checks():
    result = {
        "findings": [
            {
                "title": "Target leak",
                "statement": "refund_amount reveals the target.",
                "severity": "critical",
                "recommendation": "",
            },
            {
                "title": "Odd weekday pattern",
                "statement": "More orders on Mondays.",
                "severity": "warning",
                "recommendation": "",
            },
        ],
        "narrative": {},
        "notes": [],
        "cleaning": [
            {"op": "drop_exact_duplicates", "columns": [], "changes": "Removed 12 rows."},
            {"op": "normalize_case", "columns": ["product"], "changes": "Changed 3 values."},
            {
                "op": "parse_dates",
                "columns": ["order_date"],
                "changes": "Nothing needed changing: the data already met this rule.",
            },
        ],
        "checks": {"title_row": True},
    }
    sc = score(result, DATASETS["sales_table"])
    found = {p["id"]: p["how"] for p in sc["problems"] if p["found"]}
    assert found["leak"] == ["finding"] and found["duplicates"] == ["cleaning"]
    assert found["title_row"] == ["code"]
    assert "region_variants" not in found  # normalize_case ran, but on another column
    assert "mixed_dates" not in found  # the step changed nothing
    assert [f["title"] for f in sc["unplanned"]] == ["Odd weekday pattern"]
