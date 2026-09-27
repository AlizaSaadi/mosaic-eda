"""The MOSAIC Flow: code routes the job, agents reason inside their own steps.

Tables, text, images, audio, and video, end to end (a data-type adapter supplies what
differs): ingest -> route -> profile (code) -> triage (agent) -> plan cleaning (agent;
dry-run and distribution guardrails; safe-only fallback) -> apply cleaning + re-profile
(code) -> findings (agent; fact-check guardrail) -> review (agent) -> [revise -> review] at
most twice -> summary (agent) -> report (code).

Mixed datasets pause for the user's choice. In group mode each data type runs this whole
Flow on its own (two at a time), code links the types, and the Cross-Type Synthesizer's
findings go through the same fact check, review, and revision loop.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.flow.runtime import FlowState
from pydantic import Field, PrivateAttr

from mosaic.crews.mixed.crew import MixedCrew
from mosaic.flow.adapters import Adapter, make_adapter
from mosaic.flow.group import bundle, group_chart, part_chart_ids, record_group, run_parts
from mosaic.flow.runtime import JobRuntime
from mosaic.guardrails.task_guardrails import GuardContext, parse_output
from mosaic.ingest.models import ANALYZABLE, FileManifest, IngestError, count_label
from mosaic.ingest.service import ingest
from mosaic.llm.quota import JobTimeout, QuotaExhausted
from mosaic.models.agent_outputs import (
    FindingsReport,
    ReportNarrative,
    ReviewVerdict,
    TriageBrief,
)
from mosaic.reporting.html import render_report
from mosaic.reporting.pdf import render_pdf
from mosaic.reporting.share import share_report
from mosaic.tables.ops import CleaningPlan


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
    modality: str = ""
    unit: str = "rows"
    rows_before: int = 0  # items before cleaning: rows for tables, images for images
    rows_after: int = 0
    outputs: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    review: dict[str, Any] | None = None
    review_history: list[dict[str, Any]] = Field(default_factory=list)
    revision_round: int = 0
    revise_failed: bool = False
    mixed_choice: str = "ask"  # ask | group | dominant (for datasets with several types)
    mixed_counts: dict[str, int] = Field(default_factory=dict)
    group: bool = False
    parts: list[dict[str, Any]] = Field(default_factory=list)
    share: bool = False  # opt-in: save a public copy of the report
    share_urls: dict[str, str] = Field(default_factory=dict)
    # what each cleaning step did: {"op", "description", "changes", "rationale"}
    cleaning: list[dict[str, Any]] = Field(default_factory=list)


MAX_REVISIONS = 2
STAGE_GRACE_S = 30  # a stage may finish its last request after the deadline
CHECK_KINDS = ("guardrail", "fix", "fallback", "review")


class EDAFlow(Flow[EDAState]):
    _rt: JobRuntime | None = PrivateAttr(default=None)
    _adapter: Adapter | None = PrivateAttr(default=None)
    _guard: GuardContext | None = PrivateAttr(default=None)
    _crew: Any = PrivateAttr(default=None)
    _preset_manifest: FileManifest | None = PrivateAttr(default=None)  # group-mode parts
    _manifest: FileManifest | None = PrivateAttr(default=None)
    _label: str = PrivateAttr(default="")  # a part's data type, shown in the feed
    _part_flows: list[Any] = PrivateAttr(default_factory=list)
    _group_charts: list[str] = PrivateAttr(default_factory=list)

    @classmethod
    def for_job(cls, runtime: JobRuntime) -> EDAFlow:
        flow = cls()
        flow._rt = runtime
        flow._guard = GuardContext(store=runtime.store, reporter=runtime.reporter)
        return flow

    # ---- helpers ----

    def _step(self, title: str, detail: str = "", status: str = "running") -> None:
        prefix = f"[{self._label}] " if self._label else ""
        self._rt.reporter.emit("step", prefix + title, detail, status)

    def _ok(self) -> bool:
        return self.state.status == "running"

    def _fail(self, message: str, status: str = "failed") -> None:
        self.state.status = status
        self.state.error = message
        self._rt.reporter.emit("error", "Run stopped", message, "error")

    def _timed(self, name: str, start: float) -> None:
        self.state.timings[name] = round(time.time() - start, 2)

    def _timed_out(self) -> None:
        minutes = round(self._rt.settings.max_job_seconds / 60)
        self._fail(
            f"This run took longer than {minutes} minutes, usually because Gemini's free tier "
            "is overloaded right now. Try again in a few minutes."
        )

    def _kickoff_within_deadline(self, name: str, inputs: dict[str, Any]):
        """Run the stage in a worker thread and stop waiting when the job's time is up, even
        if a model request hangs past its own timeout. (An abandoned stage finishes in the
        background; its result is ignored.)"""
        left = self._rt.tracker.left()
        if left <= 0:
            raise JobTimeout("The run used up its time budget.")
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["result"] = self._crew.stage(name).kickoff(inputs=inputs)
            except BaseException as exc:  # handed back to the Flow's thread below
                box["error"] = exc

        worker = threading.Thread(target=work, name=f"stage-{name}", daemon=True)
        worker.start()
        worker.join(left + STAGE_GRACE_S)
        if worker.is_alive():
            raise JobTimeout(f"The {name} step was still running when time ran out.")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _run_stage(self, name: str, inputs: dict[str, Any], *, fatal: bool = True):
        """Run a one-task crew. Non-fatal stages post a warning and let the Flow degrade."""
        try:
            return self._kickoff_within_deadline(name, inputs)
        except Exception as exc:
            if isinstance(exc, JobTimeout) or self._rt.expired():  # can look like quota
                self._timed_out()
                return None
            if isinstance(exc, QuotaExhausted):
                self._fail(
                    "Today's free Gemini quota is used up. Try again tomorrow, or add your own "
                    "API key in the settings."
                )
                return None
            # a crew that still fails after its retries
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
            manifest = self._preset_manifest or ingest(
                self.state.source, self._rt.ws, self._rt.settings
            )
        except IngestError as exc:
            self._fail(exc.message)
            return
        self.state.source_name = manifest.source_name
        self.state.dominant = manifest.dominant.value if manifest.dominant else ""
        if manifest.is_mixed and self.state.mixed_choice != "dominant":
            self._mixed(manifest)
            return
        adapter = make_adapter(manifest, self._rt)
        if adapter is None:
            found = ", ".join(f"{n} {m.value}" for m, n in manifest.counts.items())
            self._fail(
                f"Found {found}, which this version can't analyze. It handles tables, text, "
                "logs, images, audio, and video.",
                "unsupported",
            )
            return
        if manifest.is_mixed:
            others = ", ".join(
                f"{n} {m.value}" for m, n in manifest.counts.items() if m != manifest.dominant
            )
            self.state.notes.append(
                f"The input also contains {others}. As chosen, only the main data type "
                f"({manifest.dominant.value}) was analyzed."
            )
        self._adapter = adapter
        self.state.modality, self.state.unit = adapter.modality, adapter.unit
        self._crew = adapter.crew_cls(
            self._rt.llm_for, self._guard, retries=2, n_findings=lambda: len(self.state.findings)
        )
        try:
            described = adapter.load(manifest)
        except ValueError as exc:
            self._fail(str(exc))
            return
        self._step("Data loaded", described, "done")
        self._timed("ingest", t0)

    def _mixed(self, manifest: FileManifest) -> None:
        """Several data types: pause for the user's choice, or set up group mode."""
        counts = {m.value: n for m, n in manifest.counts.items() if m in ANALYZABLE}
        self.state.mixed_counts = counts
        listing = ", ".join(count_label(m, n) for m, n in counts.items())
        if self.state.mixed_choice != "group":
            self.state.status = "needs_choice"
            self._step("Several data types found: choose how to analyze them", listing, "done")
            return
        self.state.group = True
        self._manifest = manifest
        self.state.modality, self.state.unit = "mixed", "items"
        self._crew = MixedCrew(
            self._rt.llm_for, self._guard, retries=2, n_findings=lambda: len(self.state.findings)
        )
        self._step("Several data types found: analyzing each, then linking them", listing, "done")

    @router(ingest_input)
    def route(self) -> str:
        if not self._ok():
            return "stop"
        return "group" if self.state.group else "analyze_data"

    @listen("group")
    def run_group(self) -> None:
        """Each type runs this Flow on its own; then code links them and the Cross-Type
        Synthesizer writes findings, which the review loop checks like any others."""
        t0 = time.time()
        flows = run_parts(type(self), self._rt, self._manifest, self.state.source, self.state.goal)
        self._part_flows = flows
        parts = record_group(self._rt.store, self._manifest, flows)
        self.state.parts = parts
        done = [p for p in parts if p["status"] == "done"]
        if not done:
            self._fail(
                "None of the data types could be analyzed: "
                + "; ".join(f"{p['modality']}: {p['error']}" for p in parts)
            )
            return
        self.state.quality_raw = round(sum(p["quality_before"] for p in done) / len(done), 1)
        self.state.quality_clean = round(sum(p["quality_after"] for p in done) / len(done), 1)
        self.state.rows_before = sum(p["items_before"] for p in done)
        self.state.rows_after = sum(p["items_after"] for p in done)
        self._group_charts = [
            group_chart(self._rt.store, parts),
            *part_chart_ids(self._rt.store, parts),
        ]
        links = [a for a in self._rt.store.all("profile") if a.id.startswith("mix_links")]
        self._step(
            "Linked the data types",
            links[0].summary if links else "No table describes the other files.",
            "done",
        )
        self._timed("parts", t0)
        part_findings = "\n".join(
            f"- {p['modality']}: " + "; ".join(f["title"] for f in p["finding_list"]) for p in done
        )
        self._findings_stage(
            {
                "source_name": self.state.source_name,
                "goal": self.state.goal or "(none given)",
                "part_findings": part_findings,
                "evidence_brief": self._rt.store.brief(("profile",)),
            }
        )

    @listen("analyze_data")
    def profile_raw(self) -> None:
        t0 = time.time()
        adapter = self._adapter
        self._step("Profiling the raw data (code, plus one model review for images, audio, text)")
        quality, target, items = adapter.profile_raw(self.state.goal)
        self.state.quality_raw, self.state.target, self.state.rows_before = quality, target, items
        self._guard.df = adapter.guard_df
        self._guard.columns = adapter.columns
        self._guard.execute = adapter.execute
        self._guard.post_checks = adapter.post_checks
        artifacts = self._rt.store.all("profile")
        self._rt.reporter.count("tool_runs", len(artifacts) + len(adapter.chart_ids))
        self._step(
            "Profile ready",
            f"{len(artifacts)} evidence artifacts, quality score {quality}/100",
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
            fatal=False,
        )
        if result is None:
            if not self._ok():
                return
            # Continue without the agent's brief: the code profile still guides the others
            fallback = TriageBrief(
                dataset_description=f"A {self.state.modality} dataset (automatic triage "
                "was unavailable).",
                focus_areas=["data quality problems in the profile"],
                target_column=self.state.target,
            )
            self.state.triage = fallback.model_dump()
            self.state.notes.append("Triage was unavailable, so a basic brief was used.")
            self._step("Using a basic triage brief", "", "warning")
            return
        brief = parse_output(result.tasks_output[0], TriageBrief)
        self.state.triage = brief.model_dump()
        if brief.target_column and self._adapter.columns:
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
                "catalog": self._adapter.catalog(),
            },
            fatal=False,
        )
        if result is not None:
            plan = parse_output(result.tasks_output[0], CleaningPlan)
            self.state.plan = plan.model_dump()
        elif self._ok():
            self._use_conservative_plan()
        self._timed("plan_cleaning", t0)

    def _use_conservative_plan(self) -> None:
        """Self-correction level 3 fallback: only format and type fixes, nothing lossy."""
        plan = self._adapter.conservative_plan()
        run = self._adapter.execute(plan)
        if not run.ok:
            self._fail("Neither the proposed plans nor the safe fallback passed the checks.")
            return
        self._guard.plan_run = run
        self.state.plan = plan.model_dump()
        self.state.notes.append(
            "The strategist's plans kept failing the checks, so a safe-only plan was used "
            "(only format fixes; nothing beyond unreadable data was removed)."
        )
        self._step("Using the safe-only fallback plan", f"{len(plan.ops)} operations", "warning")

    @listen(plan_cleaning)
    def apply_cleaning(self) -> None:
        if not self._ok():
            return
        t0 = time.time()
        run = self._guard.plan_run
        plan = CleaningPlan.model_validate(self.state.plan)
        self._step("Applying the cleaning plan", f"{len(run.steps)} operations")
        outputs, quality, items = self._adapter.apply(plan, run)
        self.state.quality_clean, self.state.rows_after = quality, items
        self.state.outputs.update(outputs)
        self._record_cleaning_summary(run)
        self._rt.reporter.count("tool_runs", 3)
        self._step(
            "Cleaned and re-profiled",
            f"quality {self.state.quality_raw} -> {self.state.quality_clean}; "
            f"{self.state.unit} {self.state.rows_before:,} -> {self.state.rows_after:,}",
            "done",
        )
        self._timed("apply_cleaning", t0)

    def _record_cleaning_summary(self, run) -> None:
        """Evidence for "cleaning removed N items", which agents naturally want to say."""
        unit = self.state.unit
        before, after = self.state.rows_before, self.state.rows_after
        per_step = {f"{s.index}_{s.op}": s.rows_before - s.rows_after for s in run.steps}
        removed_by = "; ".join(
            f"step {k.replace('_', ' ', 1)} removed {v}" for k, v in per_step.items() if v
        )
        self._rt.store.add(
            "cleaning_summary",
            "profile",
            "cleaning_summary",
            f"Cleaning result: {unit} {before} -> {after} (keys before, after, removed = "
            f"{before - after}; quality score {self.state.quality_raw} -> "
            f"{self.state.quality_clean}: keys quality_before, quality_after). "
            f"{removed_by or 'No step removed anything.'} (keys removed_by_step.<n>_<op>)",
            {
                "before": before,
                "after": after,
                "removed": before - after,
                "quality_before": self.state.quality_raw,
                "quality_after": self.state.quality_clean,
                "removed_by_step": per_step,
            },
        )

    def _cleaning_log(self) -> str:
        if self.state.group:
            return "\n".join(
                f"{p['modality']}: cleaned by its own crew, {p['unit']} {p['items_before']} -> "
                f"{p['items_after']}, quality {p['quality_before']} -> {p['quality_after']}"
                if p["status"] == "done"
                else f"{p['modality']}: not analyzed ({p['error'][:100]})"
                for p in self.state.parts
            )
        run = self._guard.plan_run
        return "\n".join(
            f"{s.index}. {s.op} on {s.columns or 'all'} [{s.risk}]: {self.state.unit} "
            f"{s.rows_before}->"
            f"{s.rows_after}. {s.rationale}"
            for s in run.steps
        )

    @listen(apply_cleaning)
    def analyze(self) -> None:
        if not self._ok():
            return
        t0 = time.time()
        self._findings_stage(
            {
                "source_name": self.state.source_name,
                "goal": self.state.goal or "(none given)",
                "focus_areas": "; ".join(self.state.triage.get("focus_areas", [])),
                "target": self.state.target or "none",
                "evidence_brief": self._rt.store.brief(("profile",)),
                "cleaning_log": self._cleaning_log(),
            }
        )
        self._timed("analyze", t0)

    def _findings_stage(self, inputs: dict[str, Any]) -> None:
        """The analyst writes findings; if its retries run out, keep only verified ones."""
        result = self._run_stage("analyze", inputs, fatal=False)
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
            self._rt.reporter.counters["facts_verified"] = self._guard.passing_verified
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

    # ---- review loop (self-correction level 4) ----

    def _numbered_findings(self) -> str:
        lines = []
        for i, f in enumerate(self.state.findings, 1):
            evidence = ", ".join(f["evidence"])
            lines.append(
                f"{i}. [{f['severity']}] {f['title']}: {f['statement']} "
                f"(evidence {evidence}; recommendation: {f.get('recommendation', '')})"
            )
        return "\n".join(lines)

    def _common_inputs(self) -> dict[str, Any]:
        return {
            "source_name": self.state.source_name,
            "goal": self.state.goal or "(none given)",
            "target": self.state.target or "none",
            "evidence_brief": self._rt.store.brief(("profile",)),
            "cleaning_log": self._cleaning_log(),
        }

    # "revised" is a router label, so the loop can repeat; run_group feeds cross-type findings
    @listen(or_("analyze", "run_group", "revised"))
    def review(self) -> None:
        if not self._ok() or self.state.revise_failed or not self.state.findings:
            return
        t0 = time.time()
        self._rt.reporter.count("review_rounds")
        result = self._run_stage(
            "review",
            {
                **self._common_inputs(),
                "findings": self._numbered_findings(),
                "sample_note": "; ".join(n for n in self.state.notes if "sample" in n)
                or "The full dataset was analyzed (no sampling).",
                "previous_review": self._previous_review_text(),
            },
            fatal=False,
        )
        if result is None:
            self.state.review = None
            if self._ok():
                self.state.notes.append("The reviewer couldn't run, so findings weren't reviewed.")
            return
        verdict = parse_output(result.tasks_output[0], ReviewVerdict)
        self.state.review = verdict.model_dump()
        self.state.review_history.append(self.state.review)
        blocking = [i for i in verdict.issues if i.blocking]
        lines = [
            f"Finding {i.finding} ({i.kind}): {i.problem} Fix: {i.fix_request}"
            for i in verdict.issues
        ]
        lines += [f"Missed: {m}" for m in verdict.missed]
        if verdict.approved and not blocking and not verdict.missed:
            self._rt.reporter.emit("review", "Reviewer approved the findings", "", "done")
        else:
            title = f"Reviewer requested {len(verdict.issues)} change(s)"
            if verdict.missed:
                title += f" and {len(verdict.missed)} addition(s)"
            self._rt.reporter.emit("review", title, "\n".join(lines), "rejected")
        self._timed(f"review_{len(self.state.review_history)}", t0)

    def _previous_review_text(self) -> str:
        if not self.state.review_history:
            return "This is the first review."
        last = self.state.review_history[-1]
        lines = [f"- Finding {i['finding']}: {i['fix_request']}" for i in last["issues"]]
        lines += [f"- Add: {m}" for m in last["missed"]]
        return "Your previous review asked for:\n" + ("\n".join(lines) or "- nothing")

    @router(review)
    def review_gate(self) -> str:
        if not self._ok():
            return "stop"
        if self.state.revise_failed:
            return "partial"
        verdict = self.state.review
        if verdict is None:
            return "approved"
        blocking = [i for i in verdict["issues"] if i["blocking"]]
        wants_additions = bool(verdict["missed"]) and self.state.revision_round == 0
        if not blocking and (verdict["approved"] or not wants_additions):
            return "approved"  # later "missed" items become report notes, not more rounds
        if self.state.revision_round < MAX_REVISIONS:
            return "revise"
        return "partial" if blocking else "approved"

    @router("revise")
    def revise_findings(self) -> str:
        self.state.revision_round += 1
        t0 = time.time()
        verdict = self.state.review
        requests = [
            f"- Finding {i['finding']} ({i['kind']}): {i['problem']} Change: {i['fix_request']}"
            for i in verdict["issues"]
        ] + [f"- Missing: add a finding about {m}" for m in verdict["missed"]]
        self._step(f"Revising the findings (round {self.state.revision_round} of {MAX_REVISIONS})")
        result = self._run_stage(
            "revise",
            {
                **self._common_inputs(),
                "findings": self._numbered_findings(),
                "review": "\n".join(requests),
            },
            fatal=False,
        )
        if result is None:
            if self._ok():
                self.state.revise_failed = True
            return "revised"  # the review step skips itself and the gate publishes partially
        report = parse_output(result.tasks_output[0], FindingsReport)
        self.state.findings = [f.model_dump() for f in report.findings]
        self._rt.reporter.count("self_corrections")
        self._step(
            "Revised findings passed the fact check", f"{len(report.findings)} findings", "done"
        )
        self._timed(f"revise_{self.state.revision_round}", t0)
        return "revised"

    def _apply_partial_publication(self) -> None:
        """Withhold findings the reviewer still considers misleading."""
        verdict = self.state.review or {"issues": []}
        flagged = {i["finding"] for i in verdict["issues"] if i["blocking"]}
        kept = [f for n, f in enumerate(self.state.findings, 1) if n not in flagged]
        withheld = [f["title"] for n, f in enumerate(self.state.findings, 1) if n in flagged]
        if withheld:
            self.state.notes.append(
                "Withheld after review (still misleading after revisions): " + "; ".join(withheld)
            )
            self._step(
                "Withheld findings the reviewer didn't accept",
                f"{len(withheld)} withheld",
                "warning",
            )
        self.state.findings = kept

    def _route_was_partial(self) -> bool:
        verdict = self.state.review
        if verdict is None:
            return False
        blocking = [i for i in verdict["issues"] if i["blocking"]]
        return bool(blocking) and self.state.revision_round >= MAX_REVISIONS

    @listen(or_("approved", "partial"))
    def write_report(self) -> None:
        if not self._ok():
            return
        t0 = time.time()
        if self.state.revise_failed or self._route_was_partial():
            self._apply_partial_publication()
        if not self.state.findings:
            self._fail("No findings passed both the fact check and the review.")
            return
        # Every published finding passed the fact check, so count exactly what's shown
        shown = sum(len(f["claimed_metrics"]) for f in self.state.findings)
        self._guard.verified = shown
        self._rt.reporter.counters["facts_verified"] = shown
        missed = (self.state.review or {}).get("missed") or []
        if missed:
            self.state.notes.append(
                "The reviewer noted these weren't covered: " + "; ".join(missed)
            )
        shown_findings = [
            {k: f[k] for k in ("title", "statement", "severity")} for f in self.state.findings
        ]
        for p in self.state.parts:  # group mode: each type's own findings too
            shown_findings += [{**f, "type": p["modality"]} for f in p["finding_list"]]
        findings = json.dumps(shown_findings, indent=1)
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
        self._share()
        self._timed("write_report", t0)

    def _check_timeline(self) -> list[dict[str, Any]]:
        events = self._rt.reporter.events()
        start = events[0].ts if events else 0
        return [
            {"t": round(e.ts - start), "kind": e.kind, "title": e.title, "detail": e.detail}
            for e in events
            if e.kind in CHECK_KINDS or (e.status == "warning" and e.kind in ("step", "info"))
        ]

    def _share(self) -> None:
        """Upload the HTML and PDF reports and keep their links (only when asked)."""
        if not self.state.share or self.state.status != "done":
            return
        try:
            self.state.share_urls = share_report(
                self._rt.settings, self._rt.ws.job_id, self._rt.ws.out, api=self._rt.share_api
            )
        except Exception as exc:  # the report is still downloadable
            self._rt.reporter.emit(
                "info", "The report couldn't be shared", str(exc)[:300], "warning"
            )
            return
        links = "; ".join(self.state.share_urls.values())
        self._step("Saved a shareable copy", links, "done")

    def _write_reports(self, **ctx: Any) -> None:
        """The HTML report, and the same content as a PDF (a PDF problem isn't fatal)."""
        out = self._rt.ws.out
        render_report(out / "report.html", **ctx)
        try:
            render_pdf(out / "report.pdf", **ctx)
            self.state.outputs["pdf"] = str(out / "report.pdf")
        except Exception as exc:
            self._rt.reporter.emit("info", "The PDF couldn't be made", str(exc)[:300], "warning")

    def _render(self) -> None:
        if self.state.group:
            self._render_group()
            return
        run = self._guard.plan_run
        plan = CleaningPlan.model_validate(self.state.plan)
        out = self._rt.ws.out
        charts = [self._rt.store.get(cid).data["figure"] for cid in self._adapter.chart_ids[:8]]
        self.state.cleaning = [
            {"op": s.op, "description": s.description, "changes": s.changes,
             "rationale": s.rationale}
            for s in run.steps
        ]  # fmt: skip
        self._write_reports(
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
            unit=self.state.unit,
            modality=self.state.modality,
            extras=self._adapter.report_extras(),
            notes=self.state.notes + self._adapter.notes,
            models=sorted(self._rt.models_used),
            checks=self._check_timeline(),
            counters=dict(self._rt.reporter.counters),
        )
        self._rt.reporter.write_trace(out / "trace.json")
        self.state.outputs.update(report=str(out / "report.html"), trace=str(out / "trace.json"))
        self.state.status = "done"
        self._step("Report ready", "", "done")

    def _render_group(self) -> None:
        out = self._rt.ws.out
        charts = [self._rt.store.get(cid).data["figure"] for cid in self._group_charts]
        links = next(
            (a.data for a in self._rt.store.all("profile") if a.id == "mix_links_001"), None
        )
        self._write_reports(
            source_name=self.state.source_name,
            narrative=self.state.narrative,
            triage=None,
            target=None,
            findings=self.state.findings,
            quality_raw=self.state.quality_raw,
            quality_clean=self.state.quality_clean,
            rows_before=self.state.rows_before,
            rows_after=self.state.rows_after,
            facts_verified=self._guard.verified,
            plan_summary="Each data type was cleaned by its own crew.",
            steps=[],
            charts=charts,
            unit=self.state.unit,
            modality="mixed",
            extras={"group": self.state.parts, "links": links},
            notes=self.state.notes,
            models=sorted(self._rt.models_used),
            checks=self._check_timeline(),
            counters=dict(self._rt.reporter.counters),
        )
        self._rt.reporter.write_trace(out / "trace.json")
        self.state.outputs.update(
            report=str(out / "report.html"),
            cleaned=bundle(out, self.state.parts),
            trace=str(out / "trace.json"),
        )
        self.state.status = "done"
        self._step("Report ready", "", "done")
