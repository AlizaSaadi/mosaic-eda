"""Record an example run as a replay (examples/replays/<name>/).

Usage:
    python eval/record_replay.py <example> [label]        # a live run (uses Gemini quota)
    python eval/record_replay.py <example> [label] --fake # scripted agents, for UI tests only

Examples: table, text, image, audio, video, mixed (mixed runs in group mode).
Set MOSAIC_REPLAY_DIR to write somewhere other than examples/replays.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mosaic.config import get_settings  # noqa: E402
from mosaic.events.reporter import ACTIVE, ensure_listener  # noqa: E402
from mosaic.flow.eda_flow import EDAFlow  # noqa: E402
from mosaic.flow.runtime import JobRuntime  # noqa: E402
from mosaic.llm.routing import build_tracker  # noqa: E402
from mosaic.ui import replay  # noqa: E402
from mosaic.ui.app import results_markdown, tiles_html  # noqa: E402
from mosaic.workspace import create_workspace  # noqa: E402

EXAMPLES = {
    "table": ("messy_sales.csv", "Predict which customers churned", "Messy sales table"),
    "text": ("support_tickets.zip", "Train a support ticket classifier", "Support tickets"),
    "image": ("shapes_dataset.zip", "Train an image classifier", "Shapes (images)"),
    "audio": ("speech_commands.zip", "Train a keyword-spotting model", "Spoken commands"),
    "video": ("pattern_clips.zip", "Train a video classifier", "Pattern clips (video)"),
    "mixed": ("shapes_survey.zip", "Train an image classifier", "Images + a table"),
}


def main(example: str, label: str = "", fake: bool = False) -> Path:
    file, goal, default_label = EXAMPLES[example]
    settings = get_settings()
    key = settings.gemini_api_key.get_secret_value().strip() if settings.has_gemini_key else ""
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(Path(tempfile.mkdtemp())),
        tracker=build_tracker(settings),
        api_key=key or "unused",
    )
    if fake:
        from tests.integration.fake_llm import (
            FakeReadClient,
            FakeVisionClient,
            GroupScriptedLLM,
            TextScriptedLLM,
        )

        llm = {"text": TextScriptedLLM, "mixed": GroupScriptedLLM}.get(example, GroupScriptedLLM)
        rt.client_factory = FakeReadClient if example == "text" else FakeVisionClient
        shared: dict = {}
        rt.llm_for = lambda role, t: llm(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    flow.kickoff(
        inputs={
            "source": str(ROOT / "examples" / "datasets" / file),
            "goal": goal,
            "mixed_choice": "group",
        }
    )
    state = flow.state
    if state.status != "done":
        raise SystemExit(f"The run didn't finish ({state.status}): {state.error}")
    adapter = flow._adapter
    chart_ids = adapter.chart_ids if adapter else [a.id for a in rt.store.all("chart")]
    charts = [rt.store.get(c).data["figure"] for c in chart_ids[:4]]
    files = [state.outputs[k] for k in ("report", "pdf") if k in state.outputs]
    folder = replay.save_replay(
        replay.REPLAY_DIR / example,
        label=label or default_label,
        events=rt.reporter.events(),
        counters=dict(rt.reporter.counters),
        results_md=results_markdown(state),
        tiles=tiles_html(state),
        charts=json.loads(json.dumps(charts)),
        files=files,
    )
    print(f"Saved the replay to {folder}")
    return folder


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1] if len(args) > 1 else "", fake="--fake" in sys.argv)
