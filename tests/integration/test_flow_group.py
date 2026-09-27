"""Mixed datasets: the Flow pauses for a choice; group mode runs a Flow per data type,
links them in code, and reviews the Cross-Type Synthesizer's findings (offline)."""

import zipfile
from pathlib import Path

from mosaic.config import Settings
from mosaic.events.reporter import ACTIVE, ensure_listener
from mosaic.flow.eda_flow import EDAFlow
from mosaic.flow.runtime import JobRuntime
from mosaic.llm.routing import build_tracker
from mosaic.workspace import create_workspace

from .fake_llm import FakeVisionClient, GroupScriptedLLM

SURVEY = Path(__file__).parents[2] / "examples" / "datasets" / "shapes_survey.zip"


def run_mixed(tmp_path, choice):
    settings = Settings(_env_file=None, workspace_root=tmp_path / "jobs")
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
        client_factory=FakeVisionClient,
    )
    shared: dict = {}
    rt.llm_for = lambda role, t: GroupScriptedLLM(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(inputs={"source": str(SURVEY), "goal": "", "mixed_choice": choice})
    return flow.state, rt, shared


def test_a_mixed_dataset_pauses_for_a_choice(tmp_path):
    state, _, shared = run_mixed(tmp_path, "ask")
    assert state.status == "needs_choice"
    assert state.mixed_counts == {"table": 1, "image": 38}
    assert not shared.get("calls")  # nothing is spent before the user chooses


def test_dominant_mode_analyzes_only_the_main_type(tmp_path):
    state, _, _ = run_mixed(tmp_path, "dominant")
    assert state.status == "done", state.error
    assert state.modality == "image"
    assert any("only the main data type" in n for n in state.notes)


def test_group_mode_analyzes_each_type_and_links_them(tmp_path):
    state, rt, _ = run_mixed(tmp_path, "group")
    assert state.status == "done", state.error
    assert [p["modality"] for p in state.parts] == ["table", "image"]
    assert all(p["status"] == "done" for p in state.parts)
    titles = [c.data["figure"]["layout"]["title"]["text"] for c in rt.store.all("chart")]
    assert titles[0] == "Data quality by type" and len(titles) == 5  # plus 2 per type
    assert sum(t.startswith("Image: ") for t in titles) == 2
    links = rt.store.get("mix_links_001").data
    assert links["rows_missing_file"] == 3 and links["files_unreferenced"] == 4
    assert links["label_disagreements"] == 3
    # the first cross-type answer claimed a wrong number; the fact check caught it
    assert any(
        e.kind == "guardrail" and "rows_missing_file" in e.detail for e in rt.reporter.events()
    )
    assert {f["title"] for f in state.findings} >= {"Labels disagree with folders"}
    html = Path(state.outputs["report"]).read_text(encoding="utf-8")
    assert "Each data type" in html and "Table-to-file links" in html
    assert all(p["cleaning"] for p in state.parts) and "Cleaning steps" in html
    with zipfile.ZipFile(state.outputs["cleaned"]) as z:
        names = z.namelist()
    assert "table/report.html" in names and "image/report.html" in names
    assert "image/cleaned_images.zip" in names and "table/cleaned.csv" in names
