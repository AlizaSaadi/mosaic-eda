"""The video path through the Flow, offline: scripted agents, fake vision, fake Whisper."""

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

from .fake_llm import FakeVisionClient, VideoScriptedLLM, fake_scene_whisper

CLIPS = Path(__file__).parents[2] / "examples" / "datasets" / "pattern_clips.zip"


def run_video(tmp_path):
    settings = Settings(_env_file=None, workspace_root=tmp_path / "jobs")
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
        client_factory=FakeVisionClient,
        transcriber=fake_scene_whisper(),
    )
    shared: dict = {}
    rt.llm_for = lambda role, t: VideoScriptedLLM(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(inputs={"source": str(CLIPS), "goal": "train a video classifier"})
    return flow.state, rt


def test_video_dataset_end_to_end(tmp_path):
    state, rt = run_video(tmp_path)
    assert state.status == "done", state.error
    assert state.modality == "video" and state.unit == "videos"
    assert state.rows_before == 30 and state.rows_after < state.rows_before
    overview = rt.store.get("vid_overview_001").data
    assert overview["corrupt"] == 1 and overview["rotated"] == 1 and overview["without_audio"] == 1
    quality = rt.store.get("vid_quality_001").data
    assert quality["files"]["black"] == ["patterns/pattern_10.mp4"]
    assert quality["files"]["frozen"] == ["fractals/fractal_09.mp4"]
    assert quality["files"]["too_short"] == ["cells/cell_07.mp4"]
    assert quality["silent_audio_files"] == ["fractals/fractal_04.mp4"]
    dupes = rt.store.get("vid_dupes_001").data
    assert dupes["exact_extra_copies"] == 1 and dupes["near_extra_copies"] == 3
    timeline = rt.store.get("vid_timeline_001").data
    assert timeline["scenes"] > 20 and 0 < timeline["scenes_with_speech"] < timeline["scenes"]
    assert rt.store.get("vid_vision_001").data["source"].startswith("vision model")
    html = Path(state.outputs["report"]).read_text(encoding="utf-8")
    assert "Timeline: scenes and speech" in html and "Keyframe sheets" in html
    manifest = pd.read_csv(state.outputs["manifest"])
    assert len(manifest) == state.rows_after
    assert manifest.loc[manifest.path == "patterns/pattern_09.mp4", "suspected_mislabel"].all()
    with zipfile.ZipFile(state.outputs["cleaned"]) as z:
        names = z.namelist()
    videos = [n for n in names if n.startswith("videos/") and n.endswith(".mp4")]
    assert len(videos) == state.rows_after
    assert any(n.startswith("audio/") and n.endswith(".wav") for n in names)


def test_exported_video_pipeline_reproduces_the_result(tmp_path):
    state, _ = run_video(tmp_path)
    raw = tmp_path / "raw"
    zipfile.ZipFile(CLIPS).extractall(raw)
    out = tmp_path / "rerun"
    argv = sys.argv
    sys.argv = ["cleaning_pipeline.py", str(raw / "pattern_clips"), str(out)]
    try:
        runpy.run_path(state.outputs["pipeline"], run_name="__main__")
    finally:
        sys.argv = argv
    rerun = pd.read_csv(out / "video_manifest.csv")
    app = pd.read_csv(state.outputs["manifest"])
    assert sorted(rerun["path"]) == sorted(app["path"])
