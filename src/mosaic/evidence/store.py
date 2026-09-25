"""Evidence store: every tool result is saved as an artifact with an ID agents must cite.

Agents never see raw data. A tool saves its full result here and hands the agent
a short summary plus the artifact ID. Guardrails later look up the numbers an
agent claims and compare them with the artifact.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    id: str
    kind: str  # "profile", "chart", "cleaning", ...
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


def _clean(value: Any) -> Any:
    """Make values JSON-safe (NaN/inf become None, numpy scalars become Python)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        with contextlib.suppress(ValueError, AttributeError):
            value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}. Lists are indexed: "a.0.b"."""
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(flatten(value, f"{prefix}{key}."))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            out.update(flatten(value, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = data
    return out


class EvidenceStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, Artifact] = {}
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def add(
        self,
        prefix: str,
        kind: str,
        tool: str,
        summary: str,
        data: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> Artifact:
        with self._lock:
            self._counters[prefix] += 1
            artifact_id = f"{prefix}_{self._counters[prefix]:03d}"
            artifact = Artifact(
                id=artifact_id,
                kind=kind,
                tool=tool,
                params=_clean(params or {}),
                summary=summary,
                data=_clean(data),
            )
            self._items[artifact_id] = artifact
        (self.folder / f"{artifact_id}.json").write_text(
            artifact.model_dump_json(indent=1), encoding="utf-8"
        )
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        return self._items.get(artifact_id)

    def __contains__(self, artifact_id: str) -> bool:
        return artifact_id in self._items

    def all(self, kind: str | None = None) -> list[Artifact]:
        return [a for a in self._items.values() if kind is None or a.kind == kind]

    def lookup(self, artifact_ids: list[str], key: str) -> Any:
        """Find a value for `key` in the cited artifacts.

        `key` may be a full dotted path ("columns.price.mean") or a trailing part of one
        ("price.mean"). Artifacts are searched in the order cited. Returns None when the
        key isn't found or is ambiguous within an artifact.
        """
        for artifact_id in artifact_ids:  # the first cited artifact that has the key wins
            artifact = self._items.get(artifact_id)
            if artifact is None:
                continue
            flat = flatten(artifact.data)
            if key in flat:
                return flat[key]
            matches = [v for k, v in flat.items() if k.endswith("." + key)]
            unique = {json.dumps(m, sort_keys=True, default=str) for m in matches}
            if len(unique) == 1:
                return matches[0]
        return None

    def lookup_all(self, artifact_ids: list[str], key: str) -> list[Any]:
        """Every value for `key` across the cited artifacts (one per artifact at most)."""
        return [v for a in artifact_ids if (v := self.lookup([a], key)) is not None]

    def find_value(
        self,
        value: float,
        rel_tol: float = 0.01,
        abs_tol: float = 0.06,
        limit: int = 3,
        within: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """(artifact ID, key) pairs whose value matches, optionally only in `within` artifacts."""
        hits = []
        for artifact in self._items.values():
            if artifact.kind == "chart" or (within is not None and artifact.id not in within):
                continue
            for key, v in flatten(artifact.data).items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                if abs(v - value) <= max(rel_tol * abs(v), abs_tol):
                    hits.append((artifact.id, key.removeprefix("columns.")))
                    if len(hits) >= limit:
                        return hits
        return hits

    def brief(self, kinds: tuple[str, ...] = ()) -> str:
        """The summaries agents see: one line per artifact, with its ID."""
        items = [a for a in self._items.values() if not kinds or a.kind in kinds]
        return "\n".join(f"[{a.id}] {a.summary}" for a in items)
