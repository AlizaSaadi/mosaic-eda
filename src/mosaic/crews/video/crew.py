"""The Video crew: the Table crew's agents and guardrails with video prompts. Its analyst is
the Video Synthesizer, which merges visual, audio, and timeline evidence."""

from __future__ import annotations

from crewai.project import CrewBase

from mosaic.crews.table.crew import TableCrew


@CrewBase
class VideoCrew(TableCrew):
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
