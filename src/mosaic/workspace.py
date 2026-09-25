"""Per-job workspaces: input/, work/, artifacts/, out/ under one temporary folder."""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

SUBDIRS = ("input", "work", "artifacts", "out")


@dataclass(frozen=True)
class JobWorkspace:
    job_id: str
    root: Path

    @property
    def input(self) -> Path:
        return self.root / "input"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def out(self) -> Path:
        return self.root / "out"

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def create_workspace(base: Path, job_id: str | None = None) -> JobWorkspace:
    job_id = job_id or new_job_id()
    root = base / job_id
    for name in SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return JobWorkspace(job_id=job_id, root=root)


def sweep_stale(base: Path, ttl_minutes: int) -> list[str]:
    """Delete job folders older than the TTL. Returns the removed job IDs."""
    if not base.exists():
        return []
    cutoff = time.time() - ttl_minutes * 60
    removed = []
    for child in base.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
    return removed
