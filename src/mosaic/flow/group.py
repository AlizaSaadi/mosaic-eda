"""Group mode for mixed datasets: every data type gets its own full Flow run (profile,
triage, cleaning plan, findings, review, report), two at a time in parallel, sharing the
quota tracker, the time budget, and the live feed. Code then links the types, and the
parent Flow's Cross-Type Synthesizer writes findings about the links."""

from __future__ import annotations

import copy
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mosaic.flow.runtime import JobRuntime
from mosaic.ingest.manifest import subset_manifest
from mosaic.ingest.models import FileManifest, Modality
from mosaic.mixed.links import link_table, record_links, record_overview
from mosaic.reporting import charts
from mosaic.workspace import create_workspace

GROUP_ORDER = (Modality.TABLE, Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO)
PARALLEL = 2  # parts analyzed at once; more would mostly queue on the shared rate limits
CHARTS_PER_PART = 2  # each type's first charts, shown with the group's own
SYNTHESIS_BUDGET_S = 240  # linking, cross-type findings, review, and the summary


def part_modalities(manifest: FileManifest) -> list[Modality]:
    return [m for m in GROUP_ORDER if manifest.counts.get(m)]


def run_parts(flow_cls: Any, rt: JobRuntime, manifest: FileManifest, source: str, goal: str):
    """Run one sub-Flow per data type. Returns the finished sub-Flows, in GROUP_ORDER."""
    modalities = part_modalities(manifest)
    rounds = math.ceil(len(modalities) / PARALLEL)
    budget = rt.settings.max_job_seconds * rounds + SYNTHESIS_BUDGET_S
    rt.tracker.deadline = max(rt.tracker.deadline, time.time() + budget)

    def one(modality: Modality):
        sub = JobRuntime(
            settings=rt.settings,
            ws=create_workspace(rt.ws.root / "parts", modality.value),
            tracker=rt.tracker,  # the same quota counts and deadline
            api_key=rt.api_key,
            reporter=rt.reporter,  # one live feed and one set of counters
            routes=rt.routes,
            models_used=rt.models_used,
            client_factory=rt.client_factory,
            transcriber=rt.transcriber,
        )
        sub.llm_for = rt.llm_for  # tests swap the parent's model factory
        flow = flow_cls.for_job(sub)
        flow._preset_manifest = subset_manifest(manifest, modality)
        flow._label = modality.value
        flow.kickoff(inputs={"source": source, "goal": goal, "mixed_choice": "dominant"})
        return flow

    with ThreadPoolExecutor(max_workers=PARALLEL, thread_name_prefix="part") as pool:
        return list(pool.map(one, modalities))


def summarize_part(flow: Any) -> dict[str, Any]:
    state = flow.state
    narrative = state.narrative or {}
    return {
        "modality": flow._label,
        "files": sum(flow._preset_manifest.counts.values()),
        "status": state.status,
        "error": state.error,
        "unit": state.unit,
        "items_before": state.rows_before,
        "items_after": state.rows_after,
        "quality_before": state.quality_raw,
        "quality_after": state.quality_clean,
        "findings": len(state.findings),
        "finding_list": [
            {k: f[k] for k in ("title", "statement", "severity")} for f in state.findings
        ],
        "headline": narrative.get("headline", ""),
        "out": str(flow._rt.ws.out),
        "charts": part_charts(flow),
    }


def part_charts(flow: Any, n: int = CHARTS_PER_PART) -> list[dict]:
    """A part's first few charts, with the data type added to each title."""
    adapter = flow._adapter
    if adapter is None or flow.state.status != "done":
        return []
    out = []
    for chart_id in adapter.chart_ids[:n]:
        figure = copy.deepcopy(flow._rt.store.get(chart_id).data["figure"])
        title = figure.get("layout", {}).get("title")
        if isinstance(title, dict) and title.get("text"):
            title["text"] = f"{flow._label.capitalize()}: {title['text']}"
        out.append(figure)
    return out


def link_parts(store, manifest: FileManifest, flows: list[Any]) -> list[str]:
    """Link every table that was loaded to the dataset's other files. Returns artifact IDs."""
    ids = []
    others = [
        f
        for f in manifest.files
        if f.modality in (Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO)
    ]
    for flow in flows:
        table = getattr(flow._adapter, "table", None)
        if table is None or not others:
            continue
        link = link_table(table.source, table.df, others)
        if link:
            ids.append(record_links(store, link))
    return ids


def part_chart_ids(store, parts: list[dict[str, Any]]) -> list[str]:
    """Save each part's charts in the group's store, so the app and report can show them."""
    ids = []
    for p in parts:
        for i, figure in enumerate(p.pop("charts", [])):  # stored here, not kept in state
            ids.append(
                store.add(
                    "chart",
                    "chart",
                    f"{p['modality']}_chart_{i + 1}",
                    f"Chart from the {p['modality']} analysis",
                    {"figure": figure},
                ).id
            )
    return ids


def group_chart(store, parts: list[dict[str, Any]]) -> str:
    done = [p for p in parts if p["status"] == "done"]
    names = [p["modality"] for p in done]
    before = [p["quality_before"] for p in done]
    after = [p["quality_after"] for p in done]
    gain = [round(b - a, 1) for a, b in zip(before, after, strict=True)]
    best = names[gain.index(max(gain))] if gain else ""
    figure = charts.grouped_bars(
        [n.capitalize() for n in names],
        {"Before cleaning": before, "After cleaning": after},
        title="Data quality by type",
        x="Data type",
        y="Quality score (out of 100)",
        y_range=(0, 108),
        subtitle=f"Cleaning helped {best} the most (+{max(gain):g} points)" if best else "",
    )
    return store.add(
        "chart",
        "chart",
        "group_quality",
        "Chart: data quality by type",
        {"figure": figure},
    ).id


def bundle(out: Path, parts: list[dict[str, Any]]) -> str:
    """One zip with each type's report, cleaned data, pipeline script, and trace."""
    staging = out.parent / "work" / "bundle"
    for p in parts:
        if Path(p["out"]).exists():
            shutil.copytree(p["out"], staging / p["modality"], dirs_exist_ok=True)
    return shutil.make_archive(str(out / "cleaned_by_type"), "zip", staging)


def record_group(store, manifest: FileManifest, flows: list[Any]) -> list[dict[str, Any]]:
    parts = [summarize_part(f) for f in flows]
    record_overview(store, parts)
    link_parts(store, manifest, flows)
    return parts
