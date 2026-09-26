"""Run the examples several times against live Gemini and summarize run-to-run variation.

Usage:  python eval/stability.py [runs_per_example]
Each run costs about 6-14 model calls. Prints, per example: time, model calls, retries per
stage, and every guardrail rejection reason, so prompt problems show up as patterns.
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mosaic.config import get_settings  # noqa: E402
from mosaic.events.reporter import ACTIVE, ensure_listener  # noqa: E402
from mosaic.flow.eda_flow import EDAFlow  # noqa: E402
from mosaic.flow.runtime import JobRuntime  # noqa: E402
from mosaic.llm.routing import build_tracker  # noqa: E402
from mosaic.workspace import create_workspace  # noqa: E402

EXAMPLES = {
    "table": (ROOT / "examples/datasets/messy_sales.csv", "Predict which customers churned"),
    "image": (ROOT / "examples/datasets/shapes_dataset.zip", "Train an image classifier"),
    "audio": (ROOT / "examples/datasets/speech_commands.zip", "Train a keyword-spotting model"),
    "text": (ROOT / "examples/datasets/support_tickets.zip", "Train a support ticket classifier"),
}


def run_once(source: Path, goal: str, tracker) -> dict:
    settings = get_settings()
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(Path(tempfile.mkdtemp())),
        tracker=tracker,
        api_key=settings.gemini_api_key.get_secret_value().strip(),
    )
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    start = time.time()
    flow.kickoff(inputs={"source": str(source), "goal": goal})
    events = rt.reporter.events()
    rejections = [(e.title, e.detail) for e in events if e.kind == "guardrail"]
    return {
        "status": flow.state.status,
        "seconds": round(time.time() - start),
        "calls": rt.reporter.counters["model_calls"],
        "findings": len(flow.state.findings),
        "withheld": sum("Withheld" in n for n in flow.state.notes),
        "fallback_plan": any("safe-only plan" in n for n in flow.state.notes),
        "partial": any("partially verified" in n for n in flow.state.notes),
        "rejections": rejections,
        "review": [e.title for e in events if e.kind == "review"],
        "error": flow.state.error,
        "switches": [e.detail for e in events if e.kind == "fallback"],
    }


def main(runs: int, only: str = "") -> None:
    tracker = build_tracker(get_settings())
    for name, (source, goal) in EXAMPLES.items():
        if only and name != only:
            continue
        print(f"\n===== {name} =====")
        reasons: Counter[str] = Counter()
        for i in range(runs):
            r = run_once(source, goal, tracker)
            print(
                f"run {i + 1}: {r['status']} {r['seconds']}s calls={r['calls']} "
                f"findings={r['findings']} withheld={r['withheld']} "
                f"fallback_plan={r['fallback_plan']} partial={r['partial']} "
                f"review={r['review']}"
            )
            if r["error"]:
                print(f"   error: {r['error'][:300]}")
            if r["switches"]:
                print(f"   model switches ({len(r['switches'])}): {r['switches'][:4]}")
            for title, detail in r["rejections"]:
                first = detail.splitlines()[0] if detail else ""
                print(f"   - {title}: {first[:230]}")
                reasons[title] += 1
        print("rejections by stage:", dict(reasons))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2, sys.argv[2] if len(sys.argv) > 2 else "")
