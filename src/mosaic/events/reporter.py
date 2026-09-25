"""Collects everything that happens during a run for the live feed and the trace file.

Flow steps, guardrails, and model fallbacks emit events here directly. A CrewAI
event listener adds agent, task, and LLM-call events. The UI reads new events by
index while the run is in progress.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crewai.events import BaseEventListener
from crewai.events.types.agent_events import AgentExecutionStartedEvent
from crewai.events.types.llm_events import LLMCallCompletedEvent
from crewai.events.types.llm_guardrail_events import LLMGuardrailCompletedEvent
from crewai.events.types.task_events import TaskCompletedEvent, TaskFailedEvent


@dataclass
class RunEvent:
    ts: float
    kind: str  # step | agent | llm | guardrail | fallback | fix | info | error
    title: str
    detail: str = ""
    status: str = "info"  # running | done | rejected | fixed | warning | error | info
    data: dict[str, Any] = field(default_factory=dict)


class RunReporter:
    def __init__(self) -> None:
        self._events: list[RunEvent] = []
        self._lock = threading.Lock()
        self.counters: Counter[str] = Counter()

    def emit(
        self, kind: str, title: str, detail: str = "", status: str = "info", **data: Any
    ) -> None:
        with self._lock:
            self._events.append(RunEvent(time.time(), kind, title, detail, status, data))

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def since(self, index: int) -> tuple[list[RunEvent], int]:
        with self._lock:
            return self._events[index:], len(self._events)

    def events(self) -> list[RunEvent]:
        with self._lock:
            return list(self._events)

    def write_trace(self, path: Path) -> Path:
        with self._lock:
            payload = {
                "counters": dict(self.counters),
                "events": [asdict(e) for e in self._events],
            }
        path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        return path


class _Active:
    reporter: RunReporter | None = None


ACTIVE = _Active()  # one live job at a time (MAX_CONCURRENT_JOBS=1), so a module slot is enough


def _role(event: Any) -> str:
    agent = getattr(event, "agent", None)
    return getattr(agent, "role", None) or getattr(event, "agent_role", None) or "Agent"


class FeedListener(BaseEventListener):
    """Forwards CrewAI events to the active RunReporter."""

    def setup_listeners(self, crewai_event_bus) -> None:
        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def _agent_started(source, event):
            if ACTIVE.reporter:
                ACTIVE.reporter.emit("agent", f"{_role(event)} is working", status="running")

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _llm_done(source, event):
            if ACTIVE.reporter:
                ACTIVE.reporter.count("llm_calls")

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def _guardrail(source, event):
            reporter = ACTIVE.reporter
            if not reporter:
                return
            if getattr(event, "success", False):
                if getattr(event, "retry_count", 0):
                    reporter.count("self_corrections")
                    reporter.emit("fix", "Revised output accepted", status="fixed")
            else:
                reporter.count("guardrail_catches")  # the guardrail itself posts the details

        @crewai_event_bus.on(TaskCompletedEvent)
        def _task_done(source, event):
            if ACTIVE.reporter:
                name = getattr(getattr(event, "task", None), "name", None) or "Task"
                ACTIVE.reporter.emit("agent", f"{name} finished", status="done")

        @crewai_event_bus.on(TaskFailedEvent)
        def _task_failed(source, event):
            if ACTIVE.reporter:  # the Flow decides whether this stops the run
                ACTIVE.reporter.count("task_failures")


_LISTENER: FeedListener | None = None


def ensure_listener() -> None:
    global _LISTENER
    if _LISTENER is None:
        _LISTENER = FeedListener()
