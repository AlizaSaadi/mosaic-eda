"""Replays: recorded runs that play back in the office, with their real results.

A replay needs no Gemini calls, so the demo still works when the free quota runs out.
Each one is a folder under examples/replays/<name>/ with replay.json (the timed events,
counters, results text, and charts) and the downloadable files from the recorded run.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mosaic.config import PROJECT_ROOT
from mosaic.events.reporter import RunEvent

# MOSAIC_REPLAY_DIR lets tests and local previews use another folder
REPLAY_DIR = Path(os.environ.get("MOSAIC_REPLAY_DIR") or PROJECT_ROOT / "examples" / "replays")
SPEED = 5.0  # a replay plays this many times faster than the recorded run
MAX_GAP_S = 3.0  # but never waits longer than this between two events
MIN_ACT_S = 3.2  # and gives the office this long for each step it acts out


def save_replay(
    folder: Path,
    *,
    label: str,
    events: list[RunEvent],
    counters: dict[str, int],
    results_md: str,
    tiles: str,
    charts: list[dict],
    files: list[str],
) -> Path:
    """Write a replay from a finished run (used by eval/record_replay.py)."""
    folder.mkdir(parents=True, exist_ok=True)
    start = events[0].ts if events else 0.0
    kept = []
    for f in files:
        target = folder / Path(f).name
        shutil.copyfile(f, target)
        kept.append(target.name)
    data = {
        "label": label,
        "events": [{**asdict(e), "ts": round(e.ts - start, 2)} for e in events],
        "counters": dict(counters),
        "results_md": results_md,
        "tiles": tiles,
        "charts": charts,
        "files": kept,
    }
    (folder / "replay.json").write_text(json.dumps(data), encoding="utf-8")
    return folder


def available(root: Path = REPLAY_DIR) -> list[tuple[str, str]]:
    """(name, label) of every replay, sorted by name."""
    if not root.exists():
        return []
    out = []
    for folder in sorted(p for p in root.iterdir() if (p / "replay.json").exists()):
        label = json.loads((folder / "replay.json").read_text(encoding="utf-8"))["label"]
        out.append((folder.name, label))
    return out


def load(name: str, root: Path = REPLAY_DIR) -> dict[str, Any]:
    folder = root / name
    if not name or "/" in name or "\\" in name or not (folder / "replay.json").exists():
        raise ValueError("That replay doesn't exist.")
    data = json.loads((folder / "replay.json").read_text(encoding="utf-8"))
    data["events"] = [RunEvent(**e) for e in data["events"]]
    data["files"] = [str(folder / f) for f in data["files"]]
    return data


def _acts(e: RunEvent) -> bool:
    """Events the office acts out (a walk or a scene takes a few seconds)."""
    return e.kind in ("guardrail", "review") or (e.kind == "agent" and "is working" in e.title)


def schedule(events: list[RunEvent], speed: float = SPEED) -> list[float]:
    """When (seconds after the replay starts) each event should appear: faster than the
    recording, but with time for the office to act out each step."""
    times, clock, last = [], 0.0, 0.0
    for e in events:
        gap = min(max(e.ts - last, 0.0) / speed, MAX_GAP_S)
        clock += max(gap, MIN_ACT_S) if _acts(e) else gap
        last = e.ts
        times.append(clock)
    return times
