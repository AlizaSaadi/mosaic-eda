"""Logs are parsed into a table and handed to the Table crew (offline)."""

import runpy
import sys

import pandas as pd

from mosaic.config import Settings
from mosaic.events.reporter import ACTIVE, ensure_listener
from mosaic.flow.eda_flow import EDAFlow
from mosaic.flow.runtime import JobRuntime
from mosaic.llm.routing import build_tracker
from mosaic.workspace import create_workspace

from .fake_llm import ScriptedLLM


class LogScriptedLLM(ScriptedLLM):
    def _triage(self) -> dict:
        return {
            "dataset_description": "Application log records.",
            "focus_areas": ["error rate", "repeated messages"],
            "target_column": None,
        }

    def _plan(self, text: str) -> dict:
        return {
            "summary": "Mark missing tokens; nothing else is needed.",
            "ops": [{"op": "standardize_null_tokens", "rationale": "Empty loggers."}],
        }

    def _findings(self, text: str) -> dict:
        self._state["findings_attempted"] = True
        finding = {
            "severity": "info",
            "evidence": ["tbl_overview_001"],
            "claimed_metrics": [],
        }
        return {
            "findings": [
                {**finding, "title": "Log parsed", "statement": "The log parsed into records."},
                {**finding, "title": "Levels", "statement": "Records carry a level."},
                {**finding, "title": "Traces", "statement": "Some records have stack traces."},
            ]
        }


def write_log(path, n=60):
    lines = []
    for i in range(n):
        level = "ERROR" if i % 10 == 0 else "INFO"
        lines.append(
            f"2026-09-25 10:{i // 60:02d}:{i % 60:02d},120 {level} [worker-{i % 3}] job {i}"
        )
        if level == "ERROR":
            lines += ["Traceback (most recent call last):", '  File "jobs.py", line 7']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_log_is_analyzed_as_a_table(tmp_path):
    log = write_log(tmp_path / "server.log")
    settings = Settings(_env_file=None, workspace_root=tmp_path / "jobs")
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
    )
    shared: dict = {}
    rt.llm_for = lambda role, t: LogScriptedLLM(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(inputs={"source": str(log), "goal": ""})
    state = flow.state
    assert state.status == "done", state.error
    assert state.modality == "table" and state.rows_before == 60
    cleaned = pd.read_csv(state.outputs["cleaned"])
    assert {"timestamp", "level", "logger", "message", "extra_lines"} <= set(cleaned.columns)
    assert (cleaned["extra_lines"] == 2).sum() == 6

    out = tmp_path / "rerun.csv"
    argv = sys.argv
    sys.argv = ["cleaning_pipeline.py", str(log), str(out)]
    try:
        runpy.run_path(state.outputs["pipeline"], run_name="__main__")
    finally:
        sys.argv = argv
    assert len(pd.read_csv(out)) == 60
