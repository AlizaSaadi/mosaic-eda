"""The image path through the Flow, offline: scripted agents and a fake vision client."""

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

from .fake_llm import FakeVisionClient, ImageScriptedLLM

SHAPES = Path(__file__).parents[2] / "examples" / "datasets" / "shapes_dataset.zip"


def run_images(tmp_path):
    settings = Settings(_env_file=None, workspace_root=tmp_path / "jobs")
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
        client_factory=FakeVisionClient,
    )
    shared: dict = {}
    rt.llm_for = lambda role, t: ImageScriptedLLM(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(inputs={"source": str(SHAPES), "goal": "train an image classifier"})
    return flow.state, rt


def test_image_dataset_end_to_end(tmp_path):
    state, rt = run_images(tmp_path)
    assert state.status == "done", state.error
    assert state.modality == "image" and state.unit == "images"
    assert state.rows_before == 123 and state.rows_after < state.rows_before
    assert state.quality_clean > state.quality_raw
    vision = rt.store.get("img_vision_001").data
    assert vision["classes"]["circles"]["suspected_mislabels"] == [
        "circles/circle_extra_0.jpg",
        "circles/circle_extra_1.jpg",
        "circles/circle_extra_2.jpg",
    ]
    html = Path(state.outputs["report"]).read_text(encoding="utf-8")
    assert "Contact sheets reviewed by the vision model" in html and "Class balance" in html
    manifest = pd.read_csv(state.outputs["manifest"])
    assert len(manifest) == state.rows_after
    assert manifest.loc[manifest.path == "circles/circle_extra_0.jpg", "suspected_mislabel"].all()
    with zipfile.ZipFile(state.outputs["cleaned"]) as z:
        images = [n for n in z.namelist() if n.endswith((".jpg", ".png"))]
    assert len(images) == state.rows_after


def test_exported_image_pipeline_reproduces_the_result(tmp_path):
    state, _ = run_images(tmp_path)
    raw = tmp_path / "raw"
    zipfile.ZipFile(SHAPES).extractall(raw)
    out = tmp_path / "rerun"
    argv = sys.argv
    sys.argv = ["cleaning_pipeline.py", str(raw / "shapes"), str(out)]
    try:
        runpy.run_path(state.outputs["pipeline"], run_name="__main__")
    finally:
        sys.argv = argv
    rerun = pd.read_csv(out / "image_manifest.csv")
    app = pd.read_csv(state.outputs["manifest"])
    assert sorted(rerun["path"]) == sorted(app["path"])
