"""The audio path through the Flow, offline: scripted agents, fake Whisper, fake listening."""

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

from .fake_llm import AudioScriptedLLM, FakeListenClient, fake_whisper

SPEECH = Path(__file__).parents[2] / "examples" / "datasets" / "speech_commands.zip"


def expected_transcripts() -> dict:
    with zipfile.ZipFile(SPEECH) as z:
        paths = [
            n.split("/", 1)[1]
            for n in z.namelist()
            if n.endswith((".wav", ".mp3")) and "yes_007" not in n
        ]  # yes_007 is corrupt
    heard = {p: p.split("/")[0] for p in paths}
    heard.update(
        {
            "stop/stop_005.wav": "No.",
            "yes/yes_012.wav": "Stop.",
            "no/no_009.wav": "",
            "stop/stop_004.wav": "",
        }
    )
    return heard


def run_audio(tmp_path):
    settings = Settings(_env_file=None, workspace_root=tmp_path / "jobs")
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(tmp_path / "jobs"),
        tracker=build_tracker(settings),
        api_key="unused",
        client_factory=FakeListenClient,
        transcriber=fake_whisper(expected_transcripts()),
    )
    shared: dict = {}
    rt.llm_for = lambda role, t: AudioScriptedLLM(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(inputs={"source": str(SPEECH), "goal": "train a keyword-spotting model"})
    return flow.state, rt


def test_audio_dataset_end_to_end(tmp_path):
    state, rt = run_audio(tmp_path)
    assert state.status == "done", state.error
    assert state.modality == "audio" and state.unit == "clips"
    assert state.rows_before == 31 and state.rows_after < state.rows_before
    transcripts = rt.store.get("aud_transcripts_001").data
    assert [m["path"] for m in transcripts["mismatches"]] == [
        "stop/stop_005.wav",
        "yes/yes_012.wav",
    ]
    assert transcripts["non_speech"] == ["stop/stop_004.wav"]  # the music clip; silence is separate
    quality = rt.store.get("aud_quality_001").data
    assert quality["files"]["silent"] == ["no/no_009.wav"]
    assert "yes/yes_003.wav" in quality["files"]["clipped"]
    assert "no/no_008.wav" in quality["files"]["noisy"]
    assert "yes/yes_011.wav" in quality["files"]["too_short"]
    html = Path(state.outputs["report"]).read_text(encoding="utf-8")
    assert "Transcription" in html and "Spectrograms" in html
    manifest = pd.read_csv(state.outputs["manifest"])
    assert len(manifest) == state.rows_after
    assert manifest.loc[manifest.path == "yes/yes_012.wav", "suspected_mislabel"].all()
    with zipfile.ZipFile(state.outputs["cleaned"]) as z:
        clips = [n for n in z.namelist() if n.endswith(".wav")]
    assert len(clips) == state.rows_after


def test_exported_audio_pipeline_reproduces_the_result(tmp_path):
    state, _ = run_audio(tmp_path)
    raw = tmp_path / "raw"
    zipfile.ZipFile(SPEECH).extractall(raw)
    out = tmp_path / "rerun"
    argv = sys.argv
    sys.argv = ["cleaning_pipeline.py", str(raw / "speech_commands"), str(out)]
    try:
        runpy.run_path(state.outputs["pipeline"], run_name="__main__")
    finally:
        sys.argv = argv
    rerun = pd.read_csv(out / "audio_manifest.csv")
    app = pd.read_csv(state.outputs["manifest"])
    assert sorted(rerun["path"]) == sorted(app["path"])
