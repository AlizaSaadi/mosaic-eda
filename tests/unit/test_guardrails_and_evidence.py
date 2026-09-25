import json

import pandas as pd
import pytest
from crewai.tasks.task_output import TaskOutput

from mosaic.events.reporter import RunReporter
from mosaic.evidence.store import EvidenceStore
from mosaic.guardrails.task_guardrails import (
    GuardContext,
    findings_guardrail,
    plan_guardrail,
    statement_numbers,
    triage_guardrail,
)


@pytest.fixture
def ctx(tmp_path):
    store = EvidenceStore(tmp_path)
    store.add(
        "tbl_columns",
        "profile",
        "t",
        "cols",
        {
            "columns": {
                "revenue": {"skew": 19.2144, "mean": 4938.7058, "missing_pct": 0.0},
                "region": {"missing_pct": 10.6796},
            }
        },
    )
    store.add("tbl_overview", "profile", "t", "overview", {"duplicate_pct": 2.9126, "rows": 412})
    df = pd.DataFrame({"price": ["$1.00", "$2.50", "N/A"], "name": ["a", "b", "c"]}, dtype="str")
    return GuardContext(store=store, reporter=RunReporter(), df=df, columns=list(df.columns))


def out(payload: dict) -> TaskOutput:
    return TaskOutput(description="t", raw=json.dumps(payload), agent="a")


def finding(statement, metrics, evidence=("tbl_columns_001",), title="T"):
    return {
        "title": title,
        "statement": statement,
        "severity": "info",
        "evidence": list(evidence),
        "claimed_metrics": [{"key": k, "value": v} for k, v in metrics.items()],
    }


GOOD = [
    finding("Revenue skew is 19.21.", {"revenue.skew": 19.21}),
    finding("Region is 10.68% missing.", {"region.missing_pct": 10.68}),
    finding("2.91% of rows are duplicates.", {"duplicate_pct": 2.91}, ("tbl_overview_001",)),
]


def test_evidence_lookup_by_suffix_and_order(ctx):
    assert ctx.store.lookup(["tbl_columns_001"], "revenue.skew") == 19.2144
    assert ctx.store.lookup(["tbl_overview_001", "tbl_columns_001"], "duplicate_pct") == 2.9126
    assert ctx.store.lookup(["tbl_columns_001"], "missing_pct") is None  # ambiguous
    assert ctx.store.lookup(["nope_001"], "rows") is None


def test_good_findings_pass_and_count_verified(ctx):
    ok, _ = findings_guardrail(ctx)(out({"findings": GOOD}))
    assert ok and ctx.verified == 3


def test_wrong_number_is_rejected_with_the_real_value(ctx):
    bad = [finding("Revenue skew is 24.2.", {"revenue.skew": 24.2}), *GOOD[1:]]
    ok, message = findings_guardrail(ctx)(out({"findings": bad}))
    assert not ok and "evidence says 19.2144" in message


def test_unclaimed_number_must_be_in_the_cited_evidence(ctx):
    # 4938.71 is revenue.mean in the cited artifact: verified even though not declared
    fine = [
        finding("Mean revenue is 4938.71 and skew is 19.21.", {"revenue.skew": 19.21}),
        *GOOD[1:],
    ]
    ok, _ = findings_guardrail(ctx)(out({"findings": fine}))
    assert ok and ctx.verified == 4
    # 5100.5 appears nowhere in the evidence: rejected
    bad = [finding("Mean revenue is 5100.5 and skew is 19.21.", {"revenue.skew": 19.21}), *GOOD[1:]]
    ok, message = findings_guardrail(ctx)(out({"findings": bad}))
    assert not ok and "uses 5100.5, which isn't in the cited evidence" in message


def test_params_travel_as_a_json_string():
    from mosaic.tables.ops import CleaningOp

    op = CleaningOp.model_validate(
        {"op": "drop_blurry", "params_json": '{"min_blur": 12.5}', "rationale": "r"}
    )
    assert op.params == {"min_blur": 12.5}
    assert CleaningOp(op="x", params={"a": 1}, rationale="r").params_json == '{"a": 1}'
    assert "params" not in CleaningOp.model_json_schema()["properties"]
    with pytest.raises(ValueError):
        CleaningOp.model_validate({"op": "x", "params_json": "{oops", "rationale": "r"})


def test_too_few_findings_and_unknown_evidence(ctx):
    ok, message = findings_guardrail(ctx)(out({"findings": [finding("x", {}, ("tbl_ghost_009",))]}))
    assert not ok and "at least 3" in message and "don't exist" in message


def test_causal_language_is_rejected(ctx):
    bad = [finding("High discounts cause churn.", {}), *GOOD[1:]]
    ok, message = findings_guardrail(ctx)(out({"findings": bad}))
    assert not ok and "cause" in message


def test_plan_guardrail_runs_the_dry_run(ctx):
    good = {
        "summary": "s",
        "ops": [
            {"op": "standardize_null_tokens", "rationale": "r"},
            {"op": "strip_currency", "columns": ["price"], "rationale": "r"},
        ],
    }
    ok, _ = plan_guardrail(ctx)(out(good))
    assert ok and ctx.plan_run.df["price"].tolist()[:2] == [1.0, 2.5]

    bad = {"summary": "s", "ops": [{"op": "cast_numeric", "columns": ["name"], "rationale": "r"}]}
    ok, message = plan_guardrail(ctx)(out(bad))
    assert not ok and "into missing" in message


def test_triage_rejects_unknown_target(ctx):
    ok, message = triage_guardrail(ctx)(
        out({"dataset_description": "d", "focus_areas": [], "target_column": "churn"})
    )
    assert not ok and "isn't a column" in message


def test_statement_numbers_exempts_small_counts_and_years():
    assert statement_numbers("3 regions in 2025, skew 19.21, 10.7% missing, 1,204.5 max") == [
        19.21,
        10.7,
        1204.5,
    ]


def test_passing_findings_are_kept_when_others_fail(ctx):
    bad = [finding("Revenue skew is 24.2.", {"revenue.skew": 24.2}), *GOOD[1:]]
    ok, _ = findings_guardrail(ctx)(out({"findings": bad}))
    assert not ok
    assert [f["claimed_metrics"][0]["key"] for f in ctx.passing] == [
        "region.missing_pct",
        "duplicate_pct",
    ]


def test_plan_that_distorts_a_distribution_is_rejected(tmp_path):
    import numpy as np

    rng = np.random.default_rng(1)
    values = [f"{v:.2f}" for v in rng.lognormal(3, 1, 300)]
    df = pd.DataFrame({"amount": values}, dtype="str")
    ctx = GuardContext(
        store=EvidenceStore(tmp_path), reporter=RunReporter(), df=df, columns=["amount"]
    )
    plan = {
        "summary": "s",
        "ops": [
            {"op": "cast_numeric", "columns": ["amount"], "rationale": "r"},
            {
                "op": "winsorize",
                "columns": ["amount"],
                "params": {"lower": 0.25, "upper": 0.75},
                "rationale": "r",
            },
        ],
    }
    ok, message = plan_guardrail(ctx)(out(plan))
    assert not ok and "shifts its distribution" in message


def test_review_guardrail_checks_finding_numbers(ctx):
    from mosaic.guardrails.task_guardrails import review_guardrail

    check = review_guardrail(ctx, lambda: 3)
    bad = {
        "approved": False,
        "missed": [],
        "issues": [{"finding": 7, "kind": "severity", "problem": "p", "fix_request": "f"}],
    }
    ok, message = check(out(bad))
    assert not ok and "numbered 1 to 3" in message
    ok, message = check(out({"approved": False, "issues": [], "missed": []}))
    assert not ok and "list the issues" in message
    ok, _ = check(out({"approved": True, "issues": [], "missed": []}))
    assert ok


def test_rejection_points_to_where_a_number_really_is(ctx):
    ctx.store.add("tbl_quality", "profile", "t", "q", {"score": 100.0})
    bad = [finding("Quality reached 100 after cleaning.", {}), *GOOD[1:]]
    ok, message = findings_guardrail(ctx)(out({"findings": bad}))
    assert not ok and "'score' in tbl_quality_001" in message


def test_rounded_numbers_pass_the_fact_check(ctx):
    rounded = [
        finding("Region is 10.7% missing.", {"region.missing_pct": 10.7}),
        *GOOD[:1],
        GOOD[2],
    ]
    ok, _ = findings_guardrail(ctx)(out({"findings": rounded}))
    assert ok


def test_triage_turns_null_words_into_null(ctx):
    for word in ("null", "None", " n/a ", ""):
        ok, _ = triage_guardrail(ctx)(
            out({"dataset_description": "d", "focus_areas": [], "target_column": word})
        )
        assert ok


def test_tolerant_keys_still_need_the_right_value(ctx):
    ctx.store.add(
        "img_balance",
        "profile",
        "t",
        "b",
        {"classes": {"triangles": {"share": 11.38}, "circles": {"share": 56.1}}},
    )
    look = ctx.store.lookup_all
    assert look(["img_balance_001"], "img_balance_001.classes.triangles.share") == [11.38]
    assert look(["img_balance_001"], "share.triangles") == [11.38]  # parts in another order
    assert look(["img_balance_001"], "Classes.Triangles.Share") == [11.38]  # case
    assert look(["img_balance_001"], "share") == []  # ambiguous: two classes have a share


def test_scale_denominators_are_not_claims():
    assert statement_numbers("Quality rose from 79.7/100 to 91.6 out of 100.") == [79.7, 91.6]


def test_identifiers_with_digits_are_not_claims():
    text = "The vision review by gemini-3.5-flash-lite (img_vision_001) flagged 5 files."
    assert statement_numbers(text) == []


def test_underscore_keys_and_bare_artifact_ids(ctx):
    ctx.store.add(
        "img_balance",
        "profile",
        "t",
        "b",
        {"classes": {"triangles": {"share": 11.38}, "circles": {"share": 56.1}}},
    )
    assert ctx.store.lookup_all(["img_balance_001"], "circles_share") == [56.1]
    findings = [
        finding("Circles are 56.1% of images.", {"img_balance_001": 56.1}, ("img_balance_001",)),
        *GOOD[1:],
    ]
    ok, _ = findings_guardrail(ctx)(out({"findings": findings}))
    assert ok


def test_file_ops_accept_files_removed_by_earlier_steps():
    from mosaic.audio.ops import AUDIO_OPS, audio_namespace
    from mosaic.tables.cleaning import execute_plan
    from mosaic.tables.ops import CleaningOp, CleaningPlan

    paths = ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"]  # small removals stay under 30%
    df = pd.DataFrame(
        {
            "path": paths,
            "class": ["x"] * 5,
            "corrupt": [True] + [False] * 4,
            "suspected_mislabel": [False] * 5,
        }
    )
    plan = CleaningPlan(
        summary="s",
        ops=[
            CleaningOp(op="remove_corrupt", rationale="r", evidence=["e"]),
            CleaningOp(op="drop_files", params={"files": ["a.wav"]}, rationale="r", evidence=["e"]),
        ],
    )
    run = execute_plan(plan, df, catalog=AUDIO_OPS, namespace=audio_namespace(), unit="clips")
    assert run.ok and run.df["path"].tolist() == paths[1:]
    bad = CleaningPlan(
        summary="s",
        ops=[
            CleaningOp(
                op="drop_files", params={"files": ["ghost.wav"]}, rationale="r", evidence=["e"]
            )
        ],
    )
    run = execute_plan(bad, df, catalog=AUDIO_OPS, namespace=audio_namespace(), unit="clips")
    assert not run.ok and "ghost.wav" in run.errors[0]


def test_list_items_and_numeric_keys_are_citable(ctx):
    ctx.store.add(
        "aud_overview",
        "profile",
        "t",
        "o",
        {"sample_rates": {"16000": 20, "22050": 3}, "sample_rates_khz": [16.0, 22.05]},
    )
    assert ctx.store.lookup_all(["aud_overview_001"], "sample_rates_khz") == [16.0, 22.05]
    assert ctx.store.find_value(16000, within=["aud_overview_001"]) == [
        ("aud_overview_001", "sample_rates")
    ]
    assert ctx.store.find_value(1, within=["aud_overview_001"]) == []  # list indexes aren't values
    findings = [
        finding(
            "Clips use 16000 Hz and 22.05 kHz.",
            {"sample_rates_khz": 22.05},
            ("aud_overview_001",),
        ),
        *GOOD[1:],
    ]
    ok, _ = findings_guardrail(ctx)(out({"findings": findings}))
    assert ok
