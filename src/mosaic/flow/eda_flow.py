"""The MOSAIC Flow: code routes the job, agents reason inside their own steps.

Phase 2 handles tables end to end:
ingest -> route -> profile (code) -> triage (agent) -> plan cleaning (agent, dry-run
guardrail) -> apply cleaning + re-profile (code) -> findings (agent, fact-check
guardrail) -> summary (agent) -> report (code).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from crewai.flow.flow import Flow, listen, router, start
from crewai.flow.runtime import FlowState
from pydantic import Field, PrivateAttr

from mosaic.crews.table.crew import TableCrew
from mosaic.flow.runtime import JobRuntime
from mosaic.guardrails.task_guardrails import GuardContext, parse_output
from mosaic.ingest.models import IngestError, Modality
from mosaic.ingest.service import ingest
from mosaic.llm.quota import QuotaExhausted
from mosaic.models.agent_outputs import FindingsReport, ReportNarrative, TriageBrief
from mosaic.reporting.html import render_report
from mosaic.tables.cleaning import export_clean, pipeline_script
from mosaic.tables.load import LoadedTable, load_table
from mosaic.tables.ops import CleaningPlan, catalog_text
from mosaic.tables.profile import ProfileResult, profile_table


class EDAState(FlowState):
    source: str = ""
    goal: str = ""
    status: str = "pending"  # pending | running | done | failed | unsupported
    error: str = ""
    source_name: str = ""
    dominant: str = ""
    notes: list[str] = Field(default_factory=list)
    target: str | None = None
    triage: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = Field(default_factory=list)
    narrative: dict[str, Any] | None = None
    quality_raw: float | None = None
    quality_clean: float | None = None
    rows_before: int = 0
    rows_after: int = 0
    outputs: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)


class EDAFlow(Flow[EDAState]):
    _rt: JobRuntime | None = PrivateAttr(default=None)
    _table: LoadedTable | None = PrivateAttr(default=None)
    _raw: ProfileResult | None = PrivateAttr(default=None)
    _clean: ProfileResult | None = PrivateAttr(default=None)
    _guard: GuardContext | None = PrivateAttr(default=None)
    _crew: TableCrew | None = PrivateAttr(default=None)

    @classmethod
    def for_job(cls, runtime: JobRuntime) -> EDAFlow:
        flow = cls()
        flow._rt = runtime
        flow._guard = GuardContext(store=runtime.store, reporter=runtime.reporter)
        flow._crew = TableCrew(runtime.llm_for, flow._guard, retries=2)
        return flow

    # ---- helpers ----

    def _step(self, title: str, detail: str = "", status: str = "running") -> None:
        self._rt.reporter.emit("step", title, detail, status)

    def _ok(self) -> bool:
        return self.state.status == "running"

    def _fail(self, message: str, status: str = "failed") -> None:
        self.state.status = status
        self.state.error = message
        self._rt.reporter.emit("error", "Run stopped", message, "error")

    def _timed(self, name: str, start: float) -> None:
        self.state.timings[name] = round(time.time() - start, 2)

    def _run_stage(self, name: str, inputs: dict[str, Any], *, fatal: bool = True):
        """Run a one-task crew. Non-fatal stages post a warning and let the Flow degrade."""
        try:
            return self._crew.stage(name).kickoff(inputs=inputs)
        except QuotaExhausted:
            self._fail(
                "Today's free Gemini quota is used up. Try again tomorrow, or add your own "
                "API key in the settings."
            )
        except Exception as exc:  # a crew that still fails after its retries
            message = f"The {name.replace('_', ' ')} step failed: {str(exc)[:300]}"
            if fatal:
                self._fail(message)
            else:
                self._rt.reporter.emit("info", "Retries used up", message, "warning")
        return None

    # ---- steps ----

    @start()
    def ingest_input(self) -> None:
        t0 = time.time()
        self.state.status = "running"
        self._step("Reading your data")
        try:
            manifest = ingest(self.state.source, self._rt.ws, self._rt.settings)
        except IngestError as exc:
            self._fail(exc.message)
            return
        self.state.source_name = manifest.source_name
        self.state.dominant = manifest.dominant.value if manifest.dominant else ""
        tables = sorted(manifest.files_of(Modality.TABLE), key=lambda f: -f.size)
        if manifest.is_mixed:
            self.state.notes.append(
                "The input has several data types. This version analyzes the table only."
            )
        if not tables:
            found = ", ".join(f"{n} {m}" for m, n in manifest.counts.items())
            self._fail(
                f"Found {found}. This version analyzes tables (CSV, Excel, JSON Lines) "
                "first; images, audio, text, and video are coming next.",
                "unsupported",
            )
            return
        if len(tables) > 1:
            self.state.notes.append(
                f"Found {len(tables)} tables; analyzing the largest, '{tables[0].path}'."
            )
        path = Path(manifest.root) / tables[0].path
        self._table = load_table(path, tables[0].format)
        self._step(
            "Data loaded",
            f"{tables[0].path}: {len(self._table.df):,} rows x {self._table.df.shape[1]} columns",
            "done",
        )
        self._timed("ingest", t0)

    @router(ingest_input)
    def route(self) -> str:
        return "table" if self._ok() else "stop"

    @listen("table")
    def profile_raw(self) -> None:
        t0 = time.time()
        self._step("Profiling the raw data (code only, no AI)")
        self._raw = profile_table(self._table, self._rt.store, goal=self.state.goal, stage="raw")
        self.state.quality_raw = self._raw.quality
        self.state.target = self._raw.target
        self.state.rows_before = len(self._table.df)
        self._guard.df = self._table.df
        self._guard.columns = list(self._table.df.columns)
        self._rt.reporter.count("tool_runs", len(self._raw.artifact_ids) + len(self._raw.chart_ids))
        self._step(
            "Profile ready",
            f"{len(self._raw.artifact_ids)} evidence artifacts, "
            f"quality score {self._raw.quality}/100",
            "done",
        )
        self._timed("profile_raw", t0)

    @listen(profile_raw)
    def triage(self) -> None:
        t0 = time.time()
        result = self._run_stage(
            "triage",
            {
                "source_name": self.state.source_name,
                "goal": self.state.goal or "(none given)",
                "profile_brief": self._rt.store.brief(("profile",)),
            },
        )
        if result is None:
            return
        brief = parse_output(result.tasks_output[0], TriageBrief)
        self.state.triage = brief.model_dump()
        if brief.target_column:
            self.state.target = brief.target_column
        self._step("Triage done", brief.dataset_description, "done")
        self._timed("triage", t0)

    @listen(triage)
    def plan_cleaning(self) -> None:
        if not self._ok():
            return
        t0 = time.time()
        result = self._run_stage(
            "plan_cleaning",
            {
                "source_name": self.state.source_name,
                "focus_areas": "; ".join(self.state.triage.get("focus_areas", [])),
                "target": self.state.target or "none",
                "profile_brief": self._rt.store.brief(("profile",)),
                "catalog": catalog_text(),
            },
        )
        if result is None:
            return
        plan = parse_output(result.tasks_output[0], CleaningPlan)
        self.state.plan = plan.model_dump()
        self._timed("plan_cleaning", t0)

    @listen(plan_cleaning)
    def apply_cleaning(self) -> None:
        if not self._ok():
            return
        t0 = time.time()
        run = self._guard.plan_run
        plan = CleaningPlan.model_validate(self.state.plan)
        self._step("Applying the cleaning plan", f"{len(run.steps)} operations")
        clean_table = LoadedTable(
            df=run.df.astype("str"), source=self._table.source, format=self._table.format
        )
        self._clean = profile_table(clean_table, self._rt.store, stage="clean")
        self.state.quality_clean = self._clean.quality
        self.state.rows_after = len(run.df)
        out = self._rt.ws.out
        export_clean(run.df, out / "cleaned.csv")
        (out / "cleaning_pipeline.py").write_text(
            pipeline_script(plan, run, self._table), encoding="utf-8"
        )
        self.state.outputs.update(
            cleaned=str(out / "cleaned.csv"), pipeline=str(out / "cleaning_pipeline.py")
        )
        self._rt.reporter.count("tool_runs", len(self._clean.artifact_ids))
        self._step(
            "Cleaned and re-profiled",
            f"quality {self.state.quality_raw} -> {self.state.quality_clean}; rows "
            f"{self.state.rows_before:,} -> {self.state.rows_after:,}",
            "done",
        )
        self._timed("apply_cleaning", t0)

    def _cleaning_log(self) -> str:
        run = self._guard.plan_run
        return "\n".join(
            f"{s.index}. {s.op} on {s.columns or 'all'} [{s.risk}]: rows {s.rows_before}->"
            f"{s.rows_after}. {s.rationale}"
            for s in run.steps
        )

    @listen(apply_cleaning)
    def analyze(self) -> None:
        if not self._ok():
            return
        t0 = time.time()
        result = self._run_stage(
            "analyze",
            {
                "source_name": self.state.source_name,
                "goal": self.state.goal or "(none given)",
                "focus_areas": "; ".join(self.state.triage.get("focus_areas", [])),
                "target": self.state.target or "none",
                "evidence_brief": self._rt.store.brief(("profile",)),
                "cleaning_log": self._cleaning_log(),
            },
            fatal=False,
        )
        if result is None:
            if not self._ok():
                return  # quota ran out: that stays fatal
            passing = self._guard.passing
            if not passing:
                self._fail("No findings passed the fact check, so none are shown.")
                return
            # Keep only findings that passed the fact check on their own (never unverified ones)
            self.state.findings = passing
            self._guard.verified = self._guard.passing_verified
            self._rt.reporter.count("facts_verified", self._guard.passing_verified)
            self.state.notes.append(
                f"Analysis partially verified: {len(passing)} finding(s) passed the fact check; "
                "the rest were withheld because their numbers couldn't be verified."
            )
            self._step(
                "Kept the findings that passed the fact check", f"{len(passing)} kept", "warning"
            )
            return
        report = parse_output(result.tasks_output[0], FindingsReport)
        self.state.findings = [f.model_dump() for f in report.findings]
        self._timed("analyze", t0)

    @listen(analyze)
    def write_report(self) -> None:
        if not self._ok():
            return
        t0 = time.time()
        findings = json.dumps(
            [{k: f[k] for k in ("title", "statement", "severity")} for f in self.state.findings],
            indent=1,
        )
        result = self._run_stage(
            "write_summary",
            {
                "source_name": self.state.source_name,
                "goal": self.state.goal or "(none given)",
                "quality_raw": self.state.quality_raw,
                "quality_clean": self.state.quality_clean,
                "findings": findings,
                "cleaning_log": self._cleaning_log(),
            },
            fatal=False,  # the report is useful without a written summary
        )
        if result is not None:
            self.state.narrative = parse_output(
                result.tasks_output[0], ReportNarrative
            ).model_dump()
        elif not self._ok():
            return  # quota ran out
        else:
            self.state.notes.append("The written summary couldn't be generated for this run.")
        self._render()
        self._timed("write_report", t0)

    def _render(self) -> None:
        run = self._guard.plan_run
        plan = CleaningPlan.model_validate(self.state.plan)
        out = self._rt.ws.out
        charts = [
            self._rt.store.get(cid).data["figure"]
            for cid in (self._raw.chart_ids + self._clean.chart_ids)[:8]
        ]
        render_report(
            out / "report.html",
            source_name=self.state.source_name,
            narrative=self.state.narrative,
            triage=self.state.triage,
            target=self.state.target,
            findings=self.state.findings,
            quality_raw=self.state.quality_raw,
            quality_clean=self.state.quality_clean,
            rows_before=self.state.rows_before,
            rows_after=self.state.rows_after,
            facts_verified=self._guard.verified,
            plan_summary=plan.summary,
            steps=run.steps,
            charts=charts,
            columns=self._rt.store.get(self._raw.artifact_ids[1]).data["columns"],
            notes=self.state.notes + self._table.notes,
            models=sorted(self._rt.models_used),
        )
        self._rt.reporter.write_trace(out / "trace.json")
        self.state.outputs.update(report=str(out / "report.html"), trace=str(out / "trace.json"))
        self.state.status = "done"
        self._step("Report ready", "", "done")
