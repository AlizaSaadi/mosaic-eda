"""The text path through the Flow, offline: scripted agents and a fake reading review."""

import json
import runpy
import sys
import zipfile
from pathlib import Path

import pandas as pd

from mosaic.config import Settings
from mosaic.events.reporter import ACTIVE, ensure_listener
from mosaic.flow.eda_flow import EDAFlow
from mosaic.flow.runtime import JobRuntime
from mosaic.llm.routing import build_tracker
from mosaic.workspace import create_workspace

from .fake_llm import TEXT_MISLABELS, FakeReadClient, TextScriptedLLM

TICKETS = Path(__file__).parents[2] / "examples" / "datasets" / "support_tickets.zip"


def read_jsonl(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


def run_text(tmp_path, source=TICKETS, goal="train a ticket classifier"):
    settings = Settings(_env_file=None, workspace_root=tmp_path / "jobs")
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
        client_factory=FakeReadClient,
    )
    shared: dict = {}
    rt.llm_for = lambda role, t: TextScriptedLLM(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(inputs={"source": str(source), "goal": goal})
    return flow.state, rt


def test_text_dataset_end_to_end(tmp_path):
    state, rt = run_text(tmp_path)
    assert state.status == "done", state.error
    assert state.modality == "text" and state.unit == "documents"
    assert state.rows_before == 148 and state.rows_after < state.rows_before
    labels = rt.store.get("txt_labels_001").data
    assert set(TEXT_MISLABELS) <= {m["document"] for m in labels["mismatches"]}
    quality = rt.store.get("txt_quality_001").data["counts"]
    assert quality["encoding_damage"] == 3 and quality["html_markup"] == 3
    assert quality["empty"] == 2 and quality["other_language"] == 5
    dupes = rt.store.get("txt_dupes_001").data
    assert dupes["cross_label_groups"] == 1
    assert rt.store.get("txt_read_001").data["read"] >= 5
    # the first findings claimed a wrong ratio, so the fact check had to catch it
    assert any(e.kind == "guardrail" for e in rt.reporter.events())

    records = read_jsonl(state.outputs["cleaned"])
    assert len(records) == state.rows_after
    cleaned = " ".join(r["text"] for r in records)
    assert "4111 1111 1111 1111" not in cleaned and "[CARD]" in cleaned
    assert "CONFIDENTIALITY NOTICE" not in cleaned and "<p>" not in cleaned
    assert "café" in cleaned  # the garbled ticket was repaired
    manifest = pd.read_csv(state.outputs["manifest"])
    assert manifest.loc[manifest.path.isin(TEXT_MISLABELS), "suspected_mislabel"].all()

    html = Path(state.outputs["report"]).read_text(encoding="utf-8")
    assert "Topics" in html and "Personal data" in html and "Reading review" in html
    assert "4111 1111 1111 1111" not in html  # quotes in the report are masked
    for artifact in rt.store.all():
        assert "4111 1111 1111 1111" not in json.dumps(artifact.data)


def test_exported_text_pipeline_reproduces_the_result(tmp_path):
    state, _ = run_text(tmp_path)
    raw = tmp_path / "raw"
    zipfile.ZipFile(TICKETS).extractall(raw)
    out = tmp_path / "rerun"
    argv = sys.argv
    sys.argv = ["cleaning_pipeline.py", str(raw / "support_tickets"), str(out)]
    try:
        runpy.run_path(state.outputs["pipeline"], run_name="__main__")
    finally:
        sys.argv = argv
    assert read_jsonl(out / "cleaned_documents.jsonl") == read_jsonl(state.outputs["cleaned"])


def test_a_transcript_is_split_into_speaker_turns(tmp_path):
    lines = []
    for i in range(30):
        lines.append(f"Agent: Thanks for calling, how can I help you with order number {i} today?")
        lines.append(f"Customer: My parcel {i} never arrived and the tracking page is stuck.")
    path = tmp_path / "call_transcript.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    state, rt = run_text(tmp_path, source=path, goal="")
    assert state.status == "done", state.error
    overview = rt.store.get("txt_overview_001").data
    assert overview["structure"] == "transcript" and overview["documents"] == 60
    assert set(rt.store.get("txt_balance_001").data["classes"]) == {"Agent", "Customer"}


def test_a_run_that_runs_out_of_time_stops_with_a_clear_message(tmp_path):
    class SlowGemini(TextScriptedLLM):
        """Every call first asks the job's tracker for a model, like the real PooledLLM."""

        def call(self, messages, *args, **kwargs):
            self._state["rt"].tracker.acquire(self._state["rt"].routes["default"])
            return super().call(messages, *args, **kwargs)

    settings = Settings(_env_file=None, workspace_root=tmp_path / "jobs", max_job_seconds=0)
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
        client_factory=FakeReadClient,
    )
    shared: dict = {"rt": rt}
    rt.llm_for = lambda role, t: SlowGemini(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(inputs={"source": str(TICKETS), "goal": ""})
    assert flow.state.status == "failed"
    assert "took longer than 0 minutes" in flow.state.error


def test_a_hanging_model_call_cannot_hold_the_run(tmp_path, monkeypatch):
    import time as clock

    import mosaic.flow.eda_flow as eda_flow

    class HangingGemini(TextScriptedLLM):
        def call(self, messages, *args, **kwargs):
            clock.sleep(6)  # a request that ignores its own timeout
            return super().call(messages, *args, **kwargs)

    monkeypatch.setattr(eda_flow, "STAGE_GRACE_S", 0)
    settings = Settings(_env_file=None, workspace_root=tmp_path / "jobs", max_job_seconds=2)
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
        client_factory=FakeReadClient,
    )
    rt.llm_for = lambda role, t: HangingGemini(model=f"scripted/{role}").bind({})
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    started = clock.time()
    flow.kickoff(inputs={"source": str(TICKETS), "goal": ""})
    assert flow.state.status == "failed" and "took longer" in flow.state.error
    assert clock.time() - started < 6
