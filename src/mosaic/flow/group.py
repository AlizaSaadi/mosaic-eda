"""Group mode for mixed datasets: every data type gets its own full Flow run (profile,
triage, cleaning plan, findings, review, report), two at a time in parallel, sharing the
quota tracker, the time budget, and the live feed. Code then links the types, and the
parent Flow's Cross-Type Synthesizer writes findings about the links."""

from __future__ import annotations

import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from mosaic.flow.runtime import JobRuntime
from mosaic.ingest.manifest import subset_manifest
from mosaic.ingest.models import FileManifest, Modality
from mosaic.mixed.links import link_table, record_links, record_overview
from mosaic.ui.palette import HARVEST
from mosaic.workspace import create_workspace

GROUP_ORDER = (Modality.TABLE, Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO)
PARALLEL = 2  # parts analyzed at once; more would mostly queue on the shared rate limits
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
    }


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


def group_chart(store, parts: list[dict[str, Any]]) -> str:
    done = [p for p in parts if p["status"] == "done"]
    names = [p["modality"] for p in done]
    fig = go.Figure(
        [
            go.Bar(name="Before cleaning", x=names, y=[p["quality_before"] for p in done]),
            go.Bar(name="After cleaning", x=names, y=[p["quality_after"] for p in done]),
        ]
    )
    fig.update_layout(
        title="Data quality by type (out of 100)",
        barmode="group",
        template="plotly_white",
        colorway=HARVEST["chart"],
        height=340,
        font={"family": "Inter, system-ui, sans-serif", "color": HARVEST["text"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    import json

    return store.add(
        "chart",
        "chart",
        "group_quality",
        "Chart: data quality by type",
        {"figure": json.loads(fig.to_json())},
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
