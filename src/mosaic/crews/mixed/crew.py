"""The Cross-Type crew (group mode): the Table crew's agents and guardrails, where the analyst
is the Cross-Type Synthesizer. Each data type was already analyzed by its own crew; this
crew writes the findings that link them, and the Reviewer checks them as usual."""

from __future__ import annotations

from crewai.project import CrewBase

from mosaic.crews.table.crew import TableCrew


@CrewBase
class MixedCrew(TableCrew):
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
