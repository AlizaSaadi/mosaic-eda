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
from mosaic.ingest.models import count_label
from mosaic.llm.quota import QuotaTracker
from mosaic.llm.routing import build_tracker
from mosaic.reporting.share import sharing_configured
from mosaic.workspace import create_workspace, sweep_stale

EXAMPLE_CSV = PROJECT_ROOT / "examples" / "datasets" / "messy_sales.csv"
EXAMPLE_IMAGES = PROJECT_ROOT / "examples" / "datasets" / "shapes_dataset.zip"
EXAMPLE_AUDIO = PROJECT_ROOT / "examples" / "datasets" / "speech_commands.zip"
EXAMPLE_TEXT = PROJECT_ROOT / "examples" / "datasets" / "support_tickets.zip"
EXAMPLE_VIDEO = PROJECT_ROOT / "examples" / "datasets" / "pattern_clips.zip"
EXAMPLE_MIXED = PROJECT_ROOT / "examples" / "datasets" / "shapes_survey.zip"
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
        f"**Data quality{' (average across types)' if state.group else ''}:** "
        f"{state.quality_raw}/100 before cleaning, {state.quality_clean}/100 after · "
        f"**{state.unit.capitalize()}:** {state.rows_before:,} → {state.rows_after:,}",
    ]
    if state.target:
        lines.append(f"· **Target:** `{state.target}`")
    lines += ["", "### Cross-type findings" if state.group else "### Findings"]
    for f in state.findings:
        lines.append(
            f"- **[{f['severity'].upper()}] {f['title']}.** {f['statement']}"
            + (f" *{f['recommendation']}*" if f.get("recommendation") else "")
            + f" `{', '.join(f['evidence'])}`"
        )
    for p in state.parts:
        lines += ["", f"### {p['modality'].capitalize()} ({p['files']} files)"]
        if p["status"] != "done":
            lines.append(f"Not analyzed: {p['error']}")
            continue
        lines.append(
            f"Quality {p['quality_before']} → {p['quality_after']} · {p['unit']} "
            f"{p['items_before']} → {p['items_after']}"
        )
        lines += [f"- **{f['title']}.** {f['statement']}" for f in p["finding_list"]]
    if n.get("next_steps"):
        lines += ["", "### Next steps", *[f"1. {s}" for s in n["next_steps"]]]
    if state.share_urls:
        lines += ["", "### Share links (public)"]
        lines += [f"- [{name}]({url})" for name, url in state.share_urls.items()]
    if state.notes:
        lines += ["", "### Notes", *[f"- {x}" for x in state.notes]]
    return "\n".join(lines)


def choice_markdown(counts: dict[str, int]) -> str:
    found = ", ".join(f"**{count_label(m, n)}**" for m, n in counts.items())
    return (
        f"### This dataset has several data types: {found}\n\n"
        "**Analyze each type, then link them** runs a separate crew for every type (two at a "
        "time) and then looks for links between them, such as table rows that point to "
        "missing files. It uses about twice the model calls. **Only the main type** analyzes "
        "the most common type and skips the rest."
    )


def run_analysis(
    upload, url: str, goal: str, user_key: str, mixed: str = "ask", share: bool = False
):
    settings = get_settings()
    source = upload if upload else (url or "").strip()
    empty_plots = [gr.update(value=None, visible=False)] * MAX_PLOTS
    hidden_gallery = gr.update(value=None, visible=False)
    hide_choice = gr.update(visible=False)

    def frame(feed, counters, results, files, plots, gallery, choice=hide_choice, text=""):
        return (
            feed,
            counters,
            results,
            files,
            *plots,
            gallery,
            runs_left_text(settings),
            choice,
            text,
        )

    if not source:
        yield frame([], "Upload a file or paste a link first.", gr.update(), None,
                    empty_plots, hidden_gallery)  # fmt: skip
        return

    sweep_stale(settings.workspace_root, settings.job_ttl_minutes)
    user_key = (user_key or "").strip()
    tracker = build_tracker(settings) if user_key else shared_tracker(settings)
    api_key = user_key or (
        settings.gemini_api_key.get_secret_value().strip() if settings.has_gemini_key else ""
    )
    if not api_key:
        yield frame([], "No Gemini key is configured. Add your own key under Settings.",
                    gr.update(), None, empty_plots, hidden_gallery)  # fmt: skip
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
    inputs = {
        "source": str(source),
        "goal": goal or "",
        "mixed_choice": mixed or "ask",
        "share": bool(share),
    }
    thread = threading.Thread(target=lambda: flow.kickoff(inputs=inputs), daemon=True)
    thread.start()

    feed: list[dict] = []
    index = 0
    while thread.is_alive():
        events, index = runtime.reporter.since(index)
        feed.extend(to_message(e) for e in events)
        yield frame(feed, counters_text(runtime.reporter, started), gr.update(), None,
                    empty_plots, hidden_gallery)  # fmt: skip
        time.sleep(POLL_SECONDS)
    thread.join()
    events, index = runtime.reporter.since(index)
    feed.extend(to_message(e) for e in events)
    ACTIVE.reporter = None

    state = flow.state
    if state.status == "needs_choice":  # the Flow paused: ask, then run again with the choice
        yield frame(feed, counters_text(runtime.reporter, started), gr.update(visible=False),
                    None, empty_plots, hidden_gallery, gr.update(visible=True),
                    choice_markdown(state.mixed_counts))  # fmt: skip
        return
    if "trace" not in state.outputs:  # failed runs still get their trace
        state.outputs["trace"] = str(runtime.reporter.write_trace(runtime.ws.out / "trace.json"))
    files = [
        state.outputs[k]
        for k in ("report", "pdf", "cleaned", "manifest", "pipeline", "trace")
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
    yield frame(feed, counters_text(runtime.reporter, started),
                gr.update(value=results_markdown(state), visible=True), files or None, plots,
                gallery)  # fmt: skip


EXAMPLES = [
    ("Try the messy sales CSV example", EXAMPLE_CSV, "Predict which customers churned"),
    ("Try the messy text dataset example", EXAMPLE_TEXT, "Train a support ticket classifier"),
    ("Try the messy image dataset example", EXAMPLE_IMAGES, "Train an image classifier"),
    ("Try the messy audio dataset example", EXAMPLE_AUDIO, "Train a keyword-spotting model"),
    ("Try the messy video dataset example", EXAMPLE_VIDEO, "Train a video classifier"),
    ("Try the mixed images + table example", EXAMPLE_MIXED, "Train an image classifier"),
]


def build_app() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="MOSAIC EDA") as demo:
        gr.Markdown(
            "# MOSAIC EDA\n"
            "**Multimodal Orchestrated System for Analysis, Inspection & Cleaning.** "
            "Drop in a messy table, text, logs, or a zip of documents, images, audio, or video. "
            "A crew of AI agents plans the cleaning, the code checks every plan and every "
            "number, and you get a report, the cleaned data, and a pipeline script you can "
            "rerun.\n\n"
            "It analyzes tables (CSV, Excel, JSON Lines), text (documents, transcripts, and "
            "logs), and image, audio, and video datasets (a zip with one folder per class). A "
            "zip with several types can be analyzed type by type and then linked. "
            "*Uses the Gemini free tier: don't upload sensitive data.*"
        )
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Upload"):
                    upload = gr.File(
                        label="A table, text, log, image, audio, video, or a zip of them",
                        file_types=[
                            ".csv", ".tsv", ".txt", ".log", ".xlsx", ".xls", ".jsonl", ".zip",
                            ".jpg", ".jpeg", ".png", ".webp", ".wav", ".mp3", ".m4a", ".flac",
                            ".ogg", ".mp4", ".mov", ".webm", ".mkv",
                        ],
                        type="filepath",
                    )  # fmt: skip
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
                share = gr.Checkbox(
                    label="Save a public copy of the report and get a share link",
                    info="Only the HTML and PDF reports are saved (quotes are short and "
                    "masked); your data isn't.",
                    value=False,
                    visible=sharing_configured(settings),
                )
                run_btn = gr.Button("Analyze", variant="primary")
                example_btns = [(gr.Button(text), path, g) for text, path, g in EXAMPLES]
                runs_left = gr.Markdown(runs_left_text(settings))
            with gr.Column(scale=3):
                feed = gr.Chatbot(label="Agent room", height=460)
                counters = gr.Markdown("")
                with gr.Group(visible=False) as choice_panel:
                    choice_text = gr.Markdown("")
                    mixed = gr.Radio(
                        choices=[
                            ("Analyze each type, then link them", "group"),
                            ("Only the main type", "dominant"),
                        ],
                        value="group",
                        label="How should MOSAIC analyze it?",
                    )
                    continue_btn = gr.Button("Continue", variant="primary")
        results = gr.Markdown(visible=False)
        with gr.Row():
            plots = [gr.Plot(visible=False) for _ in range(MAX_PLOTS)]
        gallery = gr.Gallery(
            label="Contact sheets, keyframes, or spectrograms",
            visible=False,
            columns=2,
            height="auto",
        )
        downloads = gr.File(
            label="Downloads: report (HTML and PDF), cleaned data, pipeline script, trace",
            file_count="multiple",
        )
        gr.Markdown(
            f"<small>Version {__version__} · Built with CrewAI and Gradio · "
            "Illustrations planned from Highlights (CC0)</small>"
        )

        outputs = [
            feed, counters, results, downloads, *plots, gallery, runs_left, choice_panel,
            choice_text,
        ]  # fmt: skip
        ask = gr.State("ask")
        run_btn.click(run_analysis, [upload, url, goal, user_key, ask, share], outputs)
        continue_btn.click(run_analysis, [upload, url, goal, user_key, mixed, share], outputs)
        for button, path, example_goal in example_btns:
            button.click(lambda p=path, g=example_goal: (str(p), g), None, [upload, goal]).then(
                run_analysis, [upload, url, goal, user_key, ask, share], outputs
            )
    return demo


def launch() -> None:
    settings = get_settings()
    Path(settings.workspace_root).mkdir(parents=True, exist_ok=True)
    build_app().queue(default_concurrency_limit=settings.max_concurrent_jobs).launch(
        allowed_paths=[str(settings.workspace_root), str(EXAMPLE_CSV.parent)],  # examples
        max_file_size=f"{settings.max_input_mb}mb",
    )
