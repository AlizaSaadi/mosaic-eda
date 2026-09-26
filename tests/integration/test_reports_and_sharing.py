"""The PDF report and opt-in sharing to a Hugging Face Dataset repo (offline)."""

import json
from pathlib import Path

import plotly.graph_objects as go
import pytest

from mosaic.config import Settings
from mosaic.events.reporter import ACTIVE, ensure_listener
from mosaic.flow.eda_flow import EDAFlow
from mosaic.flow.runtime import JobRuntime
from mosaic.llm.routing import build_tracker
from mosaic.reporting.share import ShareUnavailable, share_report
from mosaic.reporting.static_charts import render_png
from mosaic.workspace import create_workspace

from .fake_llm import FakeReadClient, TextScriptedLLM
from .test_flow_text import TICKETS


class FakeHfApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def create_commit(self, repo_id, operations, **kwargs):
        self.calls.append(("create_commit", repo_id, [op.path_in_repo for op in operations]))


def run(tmp_path, *, share, token="hf_test", api=None):
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path / "jobs",
        hf_token=token,
        reports_repo="someone/mosaic-eda-reports",
    )
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
        client_factory=FakeReadClient,
        share_api=api,
    )
    shared: dict = {}
    rt.llm_for = lambda role, t: TextScriptedLLM(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(inputs={"source": str(TICKETS), "goal": "", "share": share})
    return flow.state, rt


def fig_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


def test_charts_are_redrawn_as_pngs(tmp_path):
    import numpy as np

    bars = fig_json(go.Figure(go.Bar(x=["a", "b"], y=[3, 5])))
    hist = fig_json(go.Figure(go.Bar(x=np.array([1.5, 2.5]), y=np.array([4, 2]), width=[1, 1])))
    assert isinstance(hist["data"][0]["x"], dict)  # Plotly's compact typed array
    grouped = fig_json(go.Figure([go.Bar(name="before", x=["t"], y=[1]), go.Bar(x=["t"], y=[2])]))
    scatter = fig_json(go.Figure(go.Scatter(x=[1, 2], y=[1, 2])))
    for i, fig in enumerate((bars, hist, grouped)):
        assert render_png(fig, tmp_path / f"{i}.png")
        assert (tmp_path / f"{i}.png").read_bytes()[:4] == b"\x89PNG"
    assert not render_png(scatter, tmp_path / "s.png")  # not drawn rather than drawn wrong


def test_every_report_also_comes_as_a_pdf(tmp_path):
    state, _ = run(tmp_path, share=False)
    assert state.status == "done", state.error
    pdf = Path(state.outputs["pdf"])
    data = pdf.read_bytes()
    assert data[:5] == b"%PDF-" and len(data) > 20_000
    assert data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= 2  # several pages


def test_sharing_is_opt_in_and_uploads_only_the_reports(tmp_path):
    api = FakeHfApi()
    state, _ = run(tmp_path, share=False, api=api)
    assert not api.calls and not state.share_urls

    state, rt = run(tmp_path, share=True, api=api)
    assert state.status == "done"
    job = rt.ws.job_id
    assert api.calls[0] == (
        "create_repo",
        "someone/mosaic-eda-reports",
        {"repo_type": "dataset", "exist_ok": True, "private": False},
    )
    assert api.calls[1][2] == [f"reports/{job}/report.html", f"reports/{job}/report.pdf"]
    assert state.share_urls["PDF report"] == (
        f"https://huggingface.co/datasets/someone/mosaic-eda-reports/resolve/main/reports/"
        f"{job}/report.pdf"
    )
    assert "Interactive report" not in state.share_urls  # not running on a Space


def test_a_shared_report_opens_inside_the_space(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import mosaic.ui.app as app
    from mosaic.reporting.share import fetch_shared_report, viewer_url

    monkeypatch.setenv("SPACE_HOST", "someone-mosaic-eda.hf.space")
    assert viewer_url("20260926-223337-cc6613db") == (
        "https://someone-mosaic-eda.hf.space/?report=20260926-223337-cc6613db"
    )
    settings = Settings(_env_file=None, reports_repo="someone/mosaic-eda-reports")
    page = tmp_path / "report.html"
    page.write_text('<h1>Report</h1><script>draw("a & b")</script>', encoding="utf-8")
    asked = []

    def download(**kwargs):
        asked.append(kwargs)
        return str(page)

    html = fetch_shared_report(settings, "20260926-223337-cc6613db", download=download)
    assert asked[0]["filename"] == "reports/20260926-223337-cc6613db/report.html"
    with pytest.raises(ValueError):  # no paths or other files through the link
        fetch_shared_report(settings, "../../secrets", download=download)

    monkeypatch.setattr(app, "fetch_shared_report", lambda s, job: html)
    shown, welcome = app.show_shared_report(SimpleNamespace(query_params={"report": "x"}))
    assert shown["visible"] and 'sandbox="allow-scripts"' in shown["value"]
    assert "&lt;h1&gt;Report" in shown["value"]  # escaped into the frame, not into the app
    assert welcome["visible"] is False  # a shared link skips the intro
    hidden, welcome = app.show_shared_report(SimpleNamespace(query_params={}))
    assert hidden["visible"] is False and welcome["visible"] is True


def test_sharing_without_a_token_warns_but_the_run_succeeds(tmp_path):
    state, rt = run(tmp_path, share=True, token=None)
    assert state.status == "done" and not state.share_urls
    assert any(e.title == "The report couldn't be shared" for e in rt.reporter.events())
    with pytest.raises(ShareUnavailable):
        share_report(rt.settings, "job", rt.ws.out)
