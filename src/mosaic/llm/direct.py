"""Direct Gemini calls (for vision and audio) with the same pools, quotas, and fallback
as PooledLLM, for requests that aren't agent turns."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from mosaic.llm.pooled_llm import CALL_TIMEOUT_S, MAX_ATTEMPTS, is_daily_limit
from mosaic.llm.quota import PoolRule, QuotaTracker

log = logging.getLogger(__name__)

OnEvent = Callable[[dict[str, Any]], None]


def generate_structured(
    *,
    tracker: QuotaTracker,
    route: list[PoolRule],
    api_key: str,
    contents: list[Any],
    schema: type[BaseModel],
    role: str,
    on_event: OnEvent | None = None,
    client_factory: Callable[[str], Any] | None = None,
) -> tuple[BaseModel, str]:
    """Ask Gemini for JSON matching `schema`. Returns (parsed result, model used)."""
    make_client = client_factory or (
        lambda key: genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=CALL_TIMEOUT_S * 1000)
        )
    )
    client = make_client(api_key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=schema, temperature=0.2
    )
    last: Exception | None = None
    for _ in range(MAX_ATTEMPTS):
        model = tracker.acquire(route)  # raises QuotaExhausted when every model is out
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.APIError as exc:
            last = exc
            if exc.code == 429:
                tracker.mark_rate_limited(model, daily=is_daily_limit(exc))
                reason = "rate limit"
            elif exc.code in (500, 502, 503, 504):
                seconds = tracker.note_failure(model)
                reason = f"server error {exc.code} (set aside for {seconds:.0f}s)"
            else:
                raise
        except httpx.TimeoutException as exc:
            last = exc
            tracker.note_failure(model)
            reason = f"{CALL_TIMEOUT_S}s timeout"
        else:
            tracker.note_success(model)
            parsed = response.parsed
            if not isinstance(parsed, schema):
                parsed = schema.model_validate_json(response.text or "{}")
            if on_event:
                on_event(
                    {"role": role, "model": model, "ok": True, "fallback": False, "reason": ""}
                )
            return parsed, model
        log.warning("Gemini %s on %s, trying the next model", reason, model)
        if on_event:
            on_event(
                {"role": role, "model": model, "ok": False, "fallback": True, "reason": reason}
            )
    raise RuntimeError(f"No model on the '{role}' route could answer") from last
