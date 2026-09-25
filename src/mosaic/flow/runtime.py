"""Everything one job needs at run time: settings, workspace, evidence, events, and models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mosaic.config import Settings
from mosaic.events.reporter import RunReporter
from mosaic.evidence.store import EvidenceStore
from mosaic.llm.pooled_llm import PooledLLM
from mosaic.llm.quota import PoolRule, QuotaTracker
from mosaic.llm.routing import build_routes
from mosaic.workspace import JobWorkspace


@dataclass
class JobRuntime:
    settings: Settings
    ws: JobWorkspace
    tracker: QuotaTracker
    api_key: str
    reporter: RunReporter = field(default_factory=RunReporter)
    store: EvidenceStore | None = None
    routes: dict[str, list[PoolRule]] = field(default_factory=dict)
    models_used: set[str] = field(default_factory=set)
    client_factory: Any = None  # tests inject a fake Gemini client for vision/audio calls
    transcriber: Any = None  # (engine, name); tests inject a fake Whisper

    def __post_init__(self) -> None:
        self.store = self.store or EvidenceStore(self.ws.artifacts)
        self.routes = self.routes or build_routes(self.settings)

    def _on_call(self, info: dict) -> None:
        if info["ok"]:
            self.models_used.add(info["model"])
            self.reporter.count("model_calls")
            return
        self.reporter.count("fallbacks")
        self.reporter.emit(
            "fallback",
            f"Switched models for the {info['role']}",
            f"{info['model']} hit its {info['reason']}; trying the next model in the pool.",
            "warning",
        )

    def llm_for(self, role: str, temperature: float) -> PooledLLM:
        return PooledLLM.create(
            role=role,
            route=self.routes.get(role, self.routes["default"]),
            tracker=self.tracker,
            api_key=self.api_key,
            temperature=temperature,
            on_call=self._on_call,
        )
