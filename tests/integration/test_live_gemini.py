"""Live checks against the real Gemini API. Run with: pytest -m live"""

import pytest

from mosaic.config import get_settings
from mosaic.llm.pooled_llm import PooledLLM
from mosaic.llm.routing import build_routes, build_tracker

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings():
    s = get_settings()
    if not s.has_gemini_key:
        pytest.skip("GEMINI_API_KEY not set")
    return s


def test_lite_route_answers(settings):
    llm = PooledLLM.create(
        role="default",
        route=build_routes(settings)["default"],
        tracker=build_tracker(settings),
        api_key=settings.gemini_api_key.get_secret_value().strip(),
    )
    answer = llm.call("Reply with the single word OK.")
    assert "ok" in str(answer).lower()
    assert llm.last_model in settings.gemini_lite_pool


def test_crewai_agent_runs_on_pooled_llm(settings):
    from crewai import Agent, Crew, Task

    llm = PooledLLM.create(
        role="default",
        route=build_routes(settings)["default"],
        tracker=build_tracker(settings),
        api_key=settings.gemini_api_key.get_secret_value().strip(),
    )
    agent = Agent(
        role="Greeter",
        goal="Answer briefly",
        backstory="You reply in one word.",
        llm=llm,
        max_iter=2,
        verbose=False,
    )
    task = Task(description="Say hello in one word.", expected_output="One word.", agent=agent)
    result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    assert str(result).strip()
