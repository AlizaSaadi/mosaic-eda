"""The Table crew: triage, cleaning strategy, insight analysis, and the report summary.

Each stage is a one-task crew, because plain code runs between stages (applying
the cleaning plan, re-profiling). Agents get precomputed evidence in the prompt,
so a stage usually costs one model call plus any guardrail retries.
"""

from __future__ import annotations

from collections.abc import Callable

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.project import CrewBase, agent, task

from mosaic.guardrails.task_guardrails import (
    GuardContext,
    findings_guardrail,
    plan_guardrail,
    triage_guardrail,
)
from mosaic.models.agent_outputs import FindingsReport, ReportNarrative, TriageBrief
from mosaic.tables.ops import CleaningPlan

LLMFactory = Callable[[str, float], BaseLLM]  # (role, temperature) -> LLM

AGENT_DEFAULTS = {
    "allow_delegation": False,
    "max_iter": 3,
    "max_execution_time": 240,
    "verbose": False,
}


@CrewBase
class TableCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, llm_for: LLMFactory, guard: GuardContext, retries: int = 2) -> None:
        self.llm_for = llm_for
        self.guard = guard
        self.retries = retries

    # ---- agents ----

    @agent
    def dataset_triage(self) -> Agent:
        return Agent(
            config=self.agents_config["dataset_triage"],
            llm=self.llm_for("default", 0.2),
            **AGENT_DEFAULTS,
        )

    @agent
    def cleaning_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["cleaning_strategist"],
            llm=self.llm_for("strategist", 0.2),
            **AGENT_DEFAULTS,
        )

    @agent
    def insight_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["insight_analyst"],
            llm=self.llm_for("analyst", 0.3),
            **AGENT_DEFAULTS,
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"],
            llm=self.llm_for("default", 0.4),
            **AGENT_DEFAULTS,
        )

    # ---- tasks ----

    @task
    def triage(self) -> Task:
        return Task(
            config=self.tasks_config["triage"],
            agent=self.dataset_triage(),
            output_pydantic=TriageBrief,
            guardrail=triage_guardrail(self.guard),
            guardrail_max_retries=self.retries,
            name="Triage",
        )

    @task
    def plan_cleaning(self) -> Task:
        return Task(
            config=self.tasks_config["plan_cleaning"],
            agent=self.cleaning_strategist(),
            output_pydantic=CleaningPlan,
            guardrail=plan_guardrail(self.guard),
            guardrail_max_retries=self.retries,
            name="Cleaning plan",
        )

    @task
    def analyze(self) -> Task:
        return Task(
            config=self.tasks_config["analyze"],
            agent=self.insight_analyst(),
            output_pydantic=FindingsReport,
            guardrail=findings_guardrail(self.guard),
            guardrail_max_retries=self.retries + 1,  # the strictest check gets one more try
            name="Findings",
        )

    @task
    def write_summary(self) -> Task:
        return Task(
            config=self.tasks_config["write_summary"],
            agent=self.report_writer(),
            output_pydantic=ReportNarrative,
            name="Report summary",
        )

    # ---- one-task crews, run by the Flow between code steps ----

    def stage(self, name: str) -> Crew:
        pairs = {
            "triage": (self.dataset_triage, self.triage),
            "plan_cleaning": (self.cleaning_strategist, self.plan_cleaning),
            "analyze": (self.insight_analyst, self.analyze),
            "write_summary": (self.report_writer, self.write_summary),
        }
        make_agent, make_task = pairs[name]
        return Crew(
            agents=[make_agent()], tasks=[make_task()], process=Process.sequential, verbose=False
        )
