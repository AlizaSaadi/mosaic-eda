"""Gradio UI (basic version): input, live agent feed, results, and downloads."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import gradio as gr
import plotly.io as pio

from mosaic import __version__
from mosaic.config import PROJECT_ROOT, Settings, get_settings
from mosaic.events.reporter import ACTIVE, RunEvent, RunReporter, ensure_listener
from mosaic.flow.eda_flow import EDAFlow
from mosaic.flow.runtime import JobRuntime
from mosaic.llm.quota import QuotaTracker
from mosaic.llm.routing import build_tracker
from mosaic.workspace import create_workspace, sweep_stale

EXAMPLE_CSV = PROJECT_ROOT / "examples" / "datasets" / "messy_sales.csv"
EXAMPLE_IMAGES = PROJECT_ROOT / "examples" / "datasets" / "shapes_dataset.zip"
EXAMPLE_AUDIO = PROJECT_ROOT / "examples" / "datasets" / "speech_commands.zip"
EXAMPLE_TEXT = PROJECT_ROOT / "examples" / "datasets" / "support_tickets.zip"
MAX_PLOTS = 4
POLL_SECONDS = 0.6

LABELS = {
    "step": "Step",
    "agent": "Agent",
    "guardrail": "Guardrail",
    "fix": "Self-correction",
    "fallback": "Model switch",
    "review": "Review",
    "error": "Error",
    "info": "Note",
}

_TRACKER: QuotaTracker | None = None
_LOCK = threading.Lock()


def shared_tracker(settings: Settings) -> QuotaTracker:
    global _TRACKER
    with _LOCK:
        if _TRACKER is None:
            _TRACKER = build_tracker(settings)
        return _TRACKER


def runs_left_text(settings: Settings) -> str:
    left = shared_tracker(settings).runs_left_estimate()
    return f"Live runs left today on the shared free quota: about **{left}**"


def to_message(event: RunEvent) -> dict:
    label = LABELS.get(event.kind, "Note")
    if event.kind in ("guardrail", "fallback", "error", "review") and event.detail:
        return {
            "role": "assistant",
            "content": event.detail,
            "metadata": {"title": f"{label}: {event.title}", "status": "done"},
        }
    text = f"**{label}** · {event.title}"
    if event.detail:
        text += f"\n\n{event.detail}"
    return {"role": "assistant", "content": text}


def counters_text(reporter: RunReporter, started: float) -> str:
    c = reporter.counters
    return (
        f"Model calls **{c['model_calls']}** · Code tool runs **{c['tool_runs']}** · "
        f"Guardrail catches **{c['guardrail_catches']}** · Self-corrections "
        f"**{c['self_corrections']}** · Review rounds **{c['review_rounds']}** · "
        f"Model switches **{c['fallbacks']}** · "
        f"Numbers fact-checked **{c['facts_verified']}** · {time.time() - started:.0f}s"
    )


def results_markdown(state) -> str:
    if state.status in ("failed", "unsupported"):
        return f"### The run stopped\n\n{state.error}"
    n = state.narrative or {}
    lines = [
        f"## {n.get('headline', 'Analysis complete')}",
        "",
        n.get("executive_summary", ""),
        "",
        f"**Data quality:** {state.quality_raw}/100 before cleaning, "
        f"{state.quality_clean}/100 after · **{state.unit.capitalize()}:** "
        f"{state.rows_before:,} → "
        f"{state.rows_after:,}",
    ]
    if state.target:
        lines.append(f"· **Target:** `{state.target}`")
    lines += ["", "### Findings"]
    for f in state.findings:
        lines.append(
            f"- **[{f['severity'].upper()}] {f['title']}.** {f['statement']}"
            + (f" *{f['recommendation']}*" if f.get("recommendation") else "")
            + f" `{', '.join(f['evidence'])}`"
        )
    if n.get("next_steps"):
        lines += ["", "### Next steps", *[f"1. {s}" for s in n["next_steps"]]]
    if state.notes:
        lines += ["", "### Notes", *[f"- {x}" for x in state.notes]]
    return "\n".join(lines)


def run_analysis(upload, url: str, goal: str, user_key: str):
    settings = get_settings()
    source = upload if upload else (url or "").strip()
    empty_plots = [gr.update(value=None, visible=False)] * MAX_PLOTS
    hidden_gallery = gr.update(value=None, visible=False)
    if not source:
        yield (
            [],
            "Upload a file or paste a link first.",
            gr.update(),
            None,
            *empty_plots,
            hidden_gallery,
            runs_left_text(settings),
        )
        return

    sweep_stale(settings.workspace_root, settings.job_ttl_minutes)
    user_key = (user_key or "").strip()
    tracker = build_tracker(settings) if user_key else shared_tracker(settings)
    api_key = user_key or (
        settings.gemini_api_key.get_secret_value().strip() if settings.has_gemini_key else ""
    )
    if not api_key:
        yield (
            [],
            "No Gemini key is configured. Add your own key under Settings.",
            gr.update(),
            None,
            *empty_plots,
            hidden_gallery,
            runs_left_text(settings),
        )
        return

    runtime = JobRuntime(
        settings=settings,
        ws=create_workspace(settings.workspace_root),
        tracker=tracker,
        api_key=api_key,
    )
    ensure_listener()
    ACTIVE.reporter = runtime.reporter
    flow = EDAFlow.for_job(runtime)
    started = time.time()
    thread = threading.Thread(
        target=lambda: flow.kickoff(inputs={"source": str(source), "goal": goal or ""}),
        daemon=True,
    )
    thread.start()

    feed: list[dict] = []
    index = 0
    while thread.is_alive():
        events, index = runtime.reporter.since(index)
        feed.extend(to_message(e) for e in events)
        yield (
            feed,
            counters_text(runtime.reporter, started),
            gr.update(),
            None,
            *empty_plots,
            hidden_gallery,
            runs_left_text(settings),
        )
        time.sleep(POLL_SECONDS)
    thread.join()
    events, index = runtime.reporter.since(index)
    feed.extend(to_message(e) for e in events)
    ACTIVE.reporter = None

    state = flow.state
    if "trace" not in state.outputs:  # failed runs still get their trace
        state.outputs["trace"] = str(runtime.reporter.write_trace(runtime.ws.out / "trace.json"))
    files = [
        state.outputs[k]
        for k in ("report", "cleaned", "manifest", "pipeline", "trace")
        if k in state.outputs
    ]
    adapter = flow._adapter
    chart_ids = adapter.chart_ids if adapter else [a.id for a in runtime.store.all("chart")]
    plots = []
    for chart_id in chart_ids[:MAX_PLOTS]:
        fig = pio.from_json(json.dumps(runtime.store.get(chart_id).data["figure"]))
        plots.append(gr.update(value=fig, visible=True))
    pictures = adapter.gallery() if adapter and state.status == "done" else []
    gallery = gr.update(value=pictures or None, visible=bool(pictures))
    plots += [gr.update(value=None, visible=False)] * (MAX_PLOTS - len(plots))
    yield (
        feed,
        counters_text(runtime.reporter, started),
        gr.update(value=results_markdown(state), visible=True),
        files or None,
        *plots,
        gallery,
        runs_left_text(settings),
    )


def build_app() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="MOSAIC EDA") as demo:
        gr.Markdown(
            "# MOSAIC EDA\n"
            "**Multimodal Orchestrated System for Analysis, Inspection & Cleaning.** "
            "Drop in a messy table, text, logs, or a zip of documents, images, or audio. A crew of "
            "AI agents plans the "
            "cleaning, the code checks every plan and every number, and you get a report, the "
            "cleaned data, and a pipeline script you can rerun.\n\n"
            "This version analyzes tables (CSV, Excel, JSON Lines), text (documents, transcripts, "
            "and logs), and image and audio datasets (a zip with one folder per class). Video "
            "is coming next. "
            "*Uses the Gemini free tier: don't upload sensitive data.*"
        )
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Upload"):
                    upload = gr.File(
                        label="CSV, Excel, JSON Lines, text, a log, an image, audio, or a zip",
                        file_types=[
                            ".csv",
                            ".tsv",
                            ".txt",
                            ".log",
                            ".xlsx",
                            ".xls",
                            ".jsonl",
                            ".zip",
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".webp",
                            ".wav",
                            ".mp3",
                            ".m4a",
                            ".flac",
                            ".ogg",
                        ],
                        type="filepath",
                    )
                with gr.Tab("Link"):
                    url = gr.Textbox(
                        label="Link to a file",
                        placeholder="https://drive.google.com/file/d/.../view",
                    )
                goal = gr.Textbox(
                    label="What's your goal? (optional)",
                    placeholder="Predict which customers churn",
                )
                with gr.Accordion("Settings", open=False):
                    user_key = gr.Textbox(
                        label="Your own Gemini API key (optional)",
                        type="password",
                        info="Used for this run only, never stored.",
                    )
                run_btn = gr.Button("Analyze", variant="primary")
                example_btn = gr.Button("Try the messy sales CSV example")
                image_example_btn = gr.Button("Try the messy image dataset example")
                audio_example_btn = gr.Button("Try the messy audio dataset example")
                text_example_btn = gr.Button("Try the messy text dataset example")
                runs_left = gr.Markdown(runs_left_text(settings))
            with gr.Column(scale=3):
                feed = gr.Chatbot(label="Agent room", height=460)
                counters = gr.Markdown("")
        results = gr.Markdown(visible=False)
        with gr.Row():
            plots = [gr.Plot(visible=False) for _ in range(MAX_PLOTS)]
        gallery = gr.Gallery(
            label="Contact sheets or spectrograms",
            visible=False,
            columns=2,
            height="auto",
        )
        downloads = gr.File(
            label="Downloads: report, cleaned data, pipeline script, trace", file_count="multiple"
        )
        gr.Markdown(
            f"<small>Version {__version__} · Built with CrewAI and Gradio · "
            "Illustrations planned from Highlights (CC0)</small>"
        )

        outputs = [feed, counters, results, downloads, *plots, gallery, runs_left]
        run_btn.click(run_analysis, [upload, url, goal, user_key], outputs)
        example_btn.click(
            lambda: (str(EXAMPLE_CSV), "Predict which customers churned"), None, [upload, goal]
        ).then(run_analysis, [upload, url, goal, user_key], outputs)
        image_example_btn.click(
            lambda: (str(EXAMPLE_IMAGES), "Train an image classifier"), None, [upload, goal]
        ).then(run_analysis, [upload, url, goal, user_key], outputs)
        audio_example_btn.click(
            lambda: (str(EXAMPLE_AUDIO), "Train a keyword-spotting model"), None, [upload, goal]
        ).then(run_analysis, [upload, url, goal, user_key], outputs)
        text_example_btn.click(
            lambda: (str(EXAMPLE_TEXT), "Train a support ticket classifier"), None, [upload, goal]
        ).then(run_analysis, [upload, url, goal, user_key], outputs)
    return demo


def launch() -> None:
    settings = get_settings()
    Path(settings.workspace_root).mkdir(parents=True, exist_ok=True)
    build_app().queue(default_concurrency_limit=settings.max_concurrent_jobs).launch(
        allowed_paths=[str(settings.workspace_root), str(EXAMPLE_CSV.parent)],  # examples
        max_file_size=f"{settings.max_input_mb}mb",
    )
