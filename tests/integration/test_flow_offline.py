"""The whole Table Flow, end to end, with a scripted model instead of Gemini."""

from pathlib import Path

import pandas as pd
import pytest

from mosaic.config import Settings
from mosaic.events.reporter import ACTIVE, ensure_listener
from mosaic.flow.eda_flow import EDAFlow
from mosaic.flow.runtime import JobRuntime
from mosaic.llm.routing import build_tracker
from mosaic.workspace import create_workspace

from .fake_llm import ScriptedLLM

MESSY = Path(__file__).parents[2] / "examples" / "datasets" / "messy_sales.csv"


@pytest.fixture
def run_flow(tmp_path):
    def run(source, goal="predict churned", stubborn=False):
        settings = Settings(_env_file=None, workspace_root=tmp_path)
        rt = JobRuntime(
            settings=settings,
            ws=create_workspace(tmp_path),
            tracker=build_tracker(settings),
            api_key="unused",
        )
        shared: dict = {"stubborn_analyst": stubborn}
        rt.llm_for = lambda role, temperature: ScriptedLLM(model=f"scripted/{role}").bind(shared)
        ensure_listener()
        ACTIVE.reporter = rt.reporter
        flow = EDAFlow.for_job(rt)
        flow.kickoff(inputs={"source": str(source), "goal": goal})
        return flow.state, rt, shared

    return run


def test_messy_csv_end_to_end(run_flow):
    state, _rt, _shared = run_flow(MESSY)
    assert state.status == "done", state.error
    assert state.rows_before == 412 and state.rows_after == 400
    assert state.quality_clean > state.quality_raw
    assert state.target == "churned"
    for name in ("report", "cleaned", "pipeline", "trace"):
        assert Path(state.outputs[name]).exists()
    html = Path(state.outputs["report"]).read_text(encoding="utf-8")
    assert "Refund amount leaks the target" in html and "Plotly.newPlot" in html
    cleaned = pd.read_csv(state.outputs["cleaned"])
    assert len(cleaned) == 400 and "revenue_outlier" in cleaned.columns


def test_fact_check_rejects_then_accepts(run_flow):
    _state, rt, _ = run_flow(MESSY)
    kinds = [(e.kind, e.status) for e in rt.reporter.events()]
    assert ("guardrail", "rejected") in kinds
    assert ("fix", "fixed") in kinds
    assert rt.reporter.counters["self_corrections"] == 1
    assert rt.reporter.counters["facts_verified"] == 2


def test_non_table_input_is_reported_as_unsupported(run_flow, tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64)
    state, _, shared = run_flow(wav)
    assert state.status == "unsupported"
    assert "coming next" in state.error
    assert not shared.get("calls")  # no model calls for unsupported input


def test_bad_link_fails_cleanly_without_model_calls(run_flow):
    state, _, shared = run_flow("http://127.0.0.1:8080/secret.csv")
    assert state.status == "failed"
    assert "private or local network" in state.error
    assert not shared.get("calls")


def test_stubborn_analyst_degrades_to_verified_findings_only(run_flow):
    state, rt, _ = run_flow(MESSY, stubborn=True)
    assert state.status == "done", state.error
    titles = [f["title"] for f in state.findings]
    assert "Revenue is heavily right-skewed" not in titles  # its number never verified
    assert "Duplicate rows" in titles
    assert any("partially verified" in n for n in state.notes)
    assert rt.reporter.counters["facts_verified"] == 1
    assert not [e for e in rt.reporter.events() if e.kind == "error"]
