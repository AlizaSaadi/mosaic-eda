from typing import ClassVar

import pytest
from crewai.llms.base_llm import BaseLLM
from google.genai import errors as genai_errors

from mosaic.llm.pooled_llm import PooledLLM
from mosaic.llm.quota import PoolRule, QuotaExhausted, QuotaTracker


def api_error(code: int, message: str) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": message}})


class FakeDelegate(BaseLLM):
    """Stands in for a native Gemini client: returns text or raises a scripted error."""

    script: ClassVar[dict] = {}

    def call(self, messages, *args, **kwargs):
        self.script.setdefault("calls", []).append(self.model)
        outcome = self.script.get(self.model, f"answer from {self.model}")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_llm(script: dict, events: list | None = None) -> tuple[PooledLLM, QuotaTracker]:
    tracker = QuotaTracker(
        {"flash": (["flash-a", "flash-b"], 5, 20), "lite": (["lite-a"], 15, 500)}
    )

    def factory(model, owner):
        FakeDelegate.script = script
        return FakeDelegate(model=model)

    llm = PooledLLM.create(
        role="reviewer",
        route=[PoolRule("flash"), PoolRule("lite")],
        tracker=tracker,
        api_key="test-key",
        delegate_factory=factory,
        on_call=events.append if events is not None else None,
    )
    return llm, tracker


def test_first_model_answers():
    script = {}
    llm, _ = make_llm(script)
    assert llm.call("hi") == "answer from flash-a"
    assert script["calls"] == ["flash-a"]
    assert llm.last_model == "flash-a"


def test_daily_limit_falls_back_and_blocks_model_for_the_day():
    script = {"flash-a": api_error(429, "Quota exceeded: requests per day per model")}
    events = []
    llm, tracker = make_llm(script, events)
    assert llm.call("hi") == "answer from flash-b"
    assert llm.call("again") == "answer from flash-b"
    assert script["calls"] == ["flash-a", "flash-b", "flash-b"]
    assert events[0] == {
        "role": "reviewer",
        "model": "flash-a",
        "ok": False,
        "fallback": True,
        "reason": "daily limit",
    }
    assert tracker.pool_remaining("flash") == 20 - 2  # flash-a is out for the day


def test_overloaded_model_is_skipped():
    script = {"flash-a": api_error(503, "The model is overloaded")}
    llm, _ = make_llm(script)
    assert llm.call("hi") == "answer from flash-b"


def test_falls_through_to_lite_pool():
    daily = api_error(429, "limit per day")
    llm, _ = make_llm({"flash-a": daily, "flash-b": daily})
    assert llm.call("hi") == "answer from lite-a"
    assert llm.last_model == "lite-a"


def test_raises_when_everything_is_exhausted():
    daily = api_error(429, "limit per day")
    llm, _ = make_llm({"flash-a": daily, "flash-b": daily, "lite-a": daily})
    with pytest.raises(QuotaExhausted):
        llm.call("hi")


def test_other_client_errors_are_not_retried():
    llm, _ = make_llm({"flash-a": api_error(400, "bad request")})
    with pytest.raises(genai_errors.APIError):
        llm.call("hi")


def test_stop_words_reach_the_delegate():
    llm, _ = make_llm({})
    llm.stop = ["\nObservation:"]
    llm.call("hi")
    assert llm._delegates["flash-a"].stop == ["\nObservation:"]
