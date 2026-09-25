"""The Image crew: the same agents and guardrails as the Table crew, with image prompts."""

from __future__ import annotations

from crewai.project import CrewBase

from mosaic.crews.table.crew import TableCrew


@CrewBase
class ImageCrew(TableCrew):
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
