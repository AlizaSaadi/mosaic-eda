"""A CrewAI LLM that spreads calls over Gemini model pools with automatic fallback.

Each agent gets a PooledLLM for its role ("reviewer", "strategist", or "default").
On every call the quota tracker picks the first model on that role's route with
room, and the call goes to CrewAI's native Gemini client for that model, so tool
calling, structured output, and CrewAI events all work as usual. If Google answers
with a rate-limit or overload error, the model is marked and the call moves on to
the next model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from crewai.llms.base_llm import BaseLLM
from crewai.llms.providers.gemini.completion import GeminiCompletion
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import PrivateAttr

from mosaic.llm.quota import PoolRule, QuotaTracker

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
CALL_TIMEOUT_S = 45  # a slow model is treated like a busy one: fall back to the next

DelegateFactory = Callable[[str, "PooledLLM"], BaseLLM]
CallEvent = Callable[[dict[str, Any]], None]


def is_daily_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "per day" in text or "perday" in text or "per_day" in text or "daily" in text


def default_delegate(model: str, owner: PooledLLM) -> BaseLLM:
    return GeminiCompletion(
        model=model,
        api_key=owner.api_key,
        temperature=owner.temperature,
        stop=list(owner.stop),
        client_params={"http_options": genai_types.HttpOptions(timeout=CALL_TIMEOUT_S * 1000)},
    )


class PooledLLM(BaseLLM):
    role: str = "default"
    provider: str = "gemini"
    llm_type: str = "pooled-gemini"

    _route: list[PoolRule] = PrivateAttr(default_factory=list)
    _tracker: QuotaTracker | None = PrivateAttr(default=None)
    _factory: DelegateFactory | None = PrivateAttr(default=None)
    _delegates: dict[str, BaseLLM] = PrivateAttr(default_factory=dict)
    _on_call: CallEvent | None = PrivateAttr(default=None)
    last_model: str | None = None

    @classmethod
    def create(
        cls,
        *,
        role: str,
        route: list[PoolRule],
        tracker: QuotaTracker,
        api_key: str,
        temperature: float | None = None,
        delegate_factory: DelegateFactory | None = None,
        on_call: CallEvent | None = None,
    ) -> PooledLLM:
        llm = cls(model=f"pooled/{role}", role=role, api_key=api_key, temperature=temperature)
        llm._route = route
        llm._tracker = tracker
        llm._factory = delegate_factory
        llm._on_call = on_call
        return llm

    # ---- delegate management ----

    def _delegate(self, model: str) -> BaseLLM:
        delegate = self._delegates.get(model)
        if delegate is None:
            delegate = (self._factory or default_delegate)(model, self)
            self._delegates[model] = delegate
        delegate.stop = list(self.stop)  # CrewAI may set stop words after creation
        return delegate

    def _emit(self, model: str, *, ok: bool, fallback: bool, reason: str = "") -> None:
        if self._on_call:
            self._on_call(
                {
                    "role": self.role,
                    "model": model,
                    "ok": ok,
                    "fallback": fallback,
                    "reason": reason,
                }
            )

    def _handle_error(self, model: str, exc: genai_errors.APIError) -> bool:
        """Record the error. Returns True if the call should move to the next model."""
        if exc.code == 429:
            daily = is_daily_limit(exc)
            self._tracker.mark_rate_limited(model, daily=daily)
            reason = "daily limit" if daily else "per-minute limit"
        elif exc.code in (500, 502, 503, 504):
            seconds = self._tracker.note_failure(model)
            reason = f"server error {exc.code} (set aside for {seconds:.0f}s)"
        else:
            return False
        log.warning("Gemini %s on %s, trying the next model", reason, model)
        self._emit(model, ok=False, fallback=True, reason=reason)
        return True

    def _handle_timeout(self, model: str) -> None:
        self._tracker.note_failure(model)
        log.warning("Gemini %s timed out after %ss, trying the next model", model, CALL_TIMEOUT_S)
        self._emit(model, ok=False, fallback=True, reason=f"{CALL_TIMEOUT_S}s timeout")

    # ---- CrewAI interface ----

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            model = self._tracker.acquire(self._route)  # raises QuotaExhausted when all are out
            try:
                result = self._delegate(model).call(messages, *args, **kwargs)
            except genai_errors.APIError as exc:
                if not self._handle_error(model, exc):
                    raise
                last_error = exc
                continue
            except httpx.TimeoutException as exc:
                self._handle_timeout(model)
                last_error = exc
                continue
            self.last_model = model
            self._tracker.note_success(model)
            self._emit(model, ok=True, fallback=False)
            return result
        raise RuntimeError(f"No model on the '{self.role}' route could answer") from last_error

    async def acall(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            model = self._tracker.acquire(self._route)
            try:
                result = await self._delegate(model).acall(messages, *args, **kwargs)
            except genai_errors.APIError as exc:
                if not self._handle_error(model, exc):
                    raise
                last_error = exc
                continue
            except httpx.TimeoutException as exc:
                self._handle_timeout(model)
                last_error = exc
                continue
            self.last_model = model
            self._tracker.note_success(model)
            self._emit(model, ok=True, fallback=False)
            return result
        raise RuntimeError(f"No model on the '{self.role}' route could answer") from last_error

    def supports_function_calling(self) -> bool:
        return True

    def supports_multimodal(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 1_000_000
