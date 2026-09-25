"""The Audio crew: the same agents and guardrails as the Table crew, with audio prompts."""

from __future__ import annotations

from crewai.project import CrewBase

from mosaic.crews.table.crew import TableCrew


@CrewBase
class AudioCrew(TableCrew):
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
