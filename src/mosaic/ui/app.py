"""Gradio UI: four screens (welcome, upload, the office while agents work, results) and a
review box at the end."""

from __future__ import annotations

import html
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
from mosaic.reporting.share import fetch_shared_report, report_frame, sharing_configured
from mosaic.ui import replay
from mosaic.ui.art import drink, team_lineup, upload_icon
from mosaic.ui.office import office_state, office_svg
from mosaic.ui.palette import COLOR_BLIND_SAFE, HARVEST
from mosaic.ui.reviews import review_record, save_review
from mosaic.ui.theme import css_paths, harvest_theme, page_head
from mosaic.workspace import create_workspace, sweep_stale

EXAMPLE_CSV = PROJECT_ROOT / "examples" / "datasets" / "messy_sales.csv"
EXAMPLE_IMAGES = PROJECT_ROOT / "examples" / "datasets" / "shapes_dataset.zip"
EXAMPLE_AUDIO = PROJECT_ROOT / "examples" / "datasets" / "speech_commands.zip"
EXAMPLE_TEXT = PROJECT_ROOT / "examples" / "datasets" / "support_tickets.zip"
EXAMPLE_VIDEO = PROJECT_ROOT / "examples" / "datasets" / "pattern_clips.zip"
EXAMPLE_MIXED = PROJECT_ROOT / "examples" / "datasets" / "shapes_survey.zip"
MAX_PLOTS = 4
POLL_SECONDS = 0.6
SCREENS = ("welcome", "upload", "office", "results")

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


def tiles_html(state) -> str:
    if state.status != "done":
        return ""
    tiles = [
        ("Data quality before", f"{state.quality_raw}/100"),
        ("After cleaning", f"{state.quality_clean}/100"),
        (state.unit.capitalize(), f"{state.rows_before:,} → {state.rows_after:,}"),
        ("Findings", str(len(state.findings))),
    ]
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{html.escape(v)}</div></div>'
        for k, v in tiles
    ) + "</div>"  # fmt: skip


_CB = dict(zip(HARVEST["chart"], COLOR_BLIND_SAFE["chart"], strict=False))


def recolor(fig: dict) -> dict:
    """Swap Harvest chart colors for the color-blind-safe ones."""
    text = json.dumps(fig)
    for a, b in _CB.items():
        text = text.replace(a, b).replace(a.lower(), b)
    return json.loads(text)


def show(screen: str) -> list:
    return [gr.update(visible=s == screen) for s in SCREENS]


def run_analysis(
    upload,
    url: str,
    goal: str,
    user_key: str,
    mixed: str = "ask",
    share: bool = False,
    drink_kind: str = "tea",
    color_blind: bool = False,
):
    settings = get_settings()
    source = upload if upload else (url or "").strip()
    empty_plots = [gr.update(value=None, visible=False)] * MAX_PLOTS
    hidden_gallery = gr.update(value=None, visible=False)

    def frame(
        screen,
        *,
        feed=None,
        counters="",
        results=None,
        tiles="",
        files=None,
        plots=None,
        gallery=None,
        choice=False,
        choice_text="",
        office=None,
    ):
        return (
            feed if feed is not None else [],
            counters,
            results if results is not None else gr.update(),
            tiles,
            files,
            *(plots or empty_plots),
            gallery or hidden_gallery,
            runs_left_text(settings),
            gr.update(visible=choice),
            choice_text,
            *show(screen),
            office if office is not None else gr.update(),
        )  # fmt: skip

    if not source:
        gr.Warning("Upload a file or paste a link first.")
        yield frame("upload")
        return

    sweep_stale(settings.workspace_root, settings.job_ttl_minutes)
    user_key = (user_key or "").strip()
    tracker = build_tracker(settings) if user_key else shared_tracker(settings)
    api_key = user_key or (
        settings.gemini_api_key.get_secret_value().strip() if settings.has_gemini_key else ""
    )
    if not api_key:
        gr.Warning("No Gemini key is configured. Add your own key under Settings.")
        yield frame("upload")
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
    job = runtime.ws.job_id

    feed: list[dict] = []
    index = 0
    while thread.is_alive():
        events, index = runtime.reporter.since(index)
        feed.extend(to_message(e) for e in events)
        yield frame(
            "office",
            feed=feed,
            counters=counters_text(runtime.reporter, started),
            office=office_state(job, runtime.reporter.events(), drink_kind),
        )
        time.sleep(POLL_SECONDS)
    thread.join()
    events, index = runtime.reporter.since(index)
    feed.extend(to_message(e) for e in events)
    ACTIVE.reporter = None
    office = office_state(job, runtime.reporter.events(), drink_kind)

    state = flow.state
    if state.status == "needs_choice":  # the Flow paused: ask, then run again with the choice
        yield frame("upload", feed=feed, choice=True,
                    choice_text=choice_markdown(state.mixed_counts), office=office)  # fmt: skip
        return
    # let the office finish its last scene (Quill's "Report ready!") before moving on
    yield frame("office", feed=feed, counters=counters_text(runtime.reporter, started),
                office=office)  # fmt: skip
    time.sleep(4.0 if state.status == "done" else 3.5)
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
        figure = runtime.store.get(chart_id).data["figure"]
        fig = pio.from_json(json.dumps(recolor(figure) if color_blind else figure))
        plots.append(gr.update(value=fig, visible=True))
    pictures = adapter.gallery() if adapter and state.status == "done" else []
    gallery = gr.update(value=pictures or None, visible=bool(pictures))
    plots += [gr.update(value=None, visible=False)] * (MAX_PLOTS - len(plots))
    yield frame(
        "results",
        feed=feed,
        counters=counters_text(runtime.reporter, started),
        results=gr.update(value=results_markdown(state), visible=True),
        tiles=tiles_html(state),
        files=files or None,
        plots=plots,
        gallery=gallery,
        office=office,
    )


def replay_analysis(name: str, drink_kind: str = "tea", color_blind: bool = False):
    """Play a recorded run in the office, then show its saved results. No model calls."""
    settings = get_settings()
    data = replay.load(name)
    events: list[RunEvent] = data["events"]
    times = replay.schedule(events)
    job = f"replay-{name}-{time.time():.0f}"
    empty_plots = [gr.update(value=None, visible=False)] * MAX_PLOTS

    def frame(screen, shown, *, results=None, tiles="", files=None, plots=None):
        return (
            [to_message(e) for e in shown],
            f"Replay of a recorded run · {len(shown)} of {len(events)} steps",
            results if results is not None else gr.update(),
            tiles,
            files,
            *(plots or empty_plots),
            gr.update(value=None, visible=False),
            runs_left_text(settings),
            gr.update(visible=False),
            "",
            *show(screen),
            office_state(job, shown, drink_kind),
        )

    start = time.time()
    shown: list[RunEvent] = []
    for event, at in zip(events, times, strict=True):
        delay = at - (time.time() - start)
        if delay > 0:
            time.sleep(delay)
        shown.append(event)
        yield frame("office", shown)
    time.sleep(5.0)  # the office's last scene
    plots = []
    for figure in data["charts"][:MAX_PLOTS]:
        fig = pio.from_json(json.dumps(recolor(figure) if color_blind else figure))
        plots.append(gr.update(value=fig, visible=True))
    plots += [gr.update(value=None, visible=False)] * (MAX_PLOTS - len(plots))
    note = "\n\n*This was a replay of a recorded run: no data was uploaded or analyzed.*"
    yield frame(
        "results",
        shown,
        results=gr.update(value=data["results_md"] + note, visible=True),
        tiles=data["tiles"],
        files=data["files"] or None,
        plots=plots,
    )


EXAMPLES = [
    ("Messy sales table", EXAMPLE_CSV, "Predict which customers churned"),
    ("Support tickets (text)", EXAMPLE_TEXT, "Train a support ticket classifier"),
    ("Shapes (images)", EXAMPLE_IMAGES, "Train an image classifier"),
    ("Spoken commands (audio)", EXAMPLE_AUDIO, "Train a keyword-spotting model"),
    ("Pattern clips (video)", EXAMPLE_VIDEO, "Train a video classifier"),
    ("Images + a table (mixed)", EXAMPLE_MIXED, "Train an image classifier"),
]

WELCOME = (
    '<div class="welcome"><div class="eyebrow">Welcome</div><h1>Mosaic EDA</h1>'
    '<div class="by">A project by Aliza Saadi: a crew of AI agents that cleans, explores, and '
    "fact-checks any dataset.</div>"
    '<div class="meet">Meet the Mosaic team</div>'
    f"{team_lineup()}"
    '<div class="ask">When your report is ready, I\'d love a quick review or a suggestion. '
    "It helps me make Mosaic better.</div></div>"
)
STEPS = (
    '<div class="steps">'
    '<div class="step"><div class="k">Step 1</div><div class="t">Drop in anything</div>'
    '<div class="d">A table, text, logs, images, audio, video, or a zip of them.</div></div>'
    '<div class="step"><div class="k">Step 2</div><div class="t">The team gets to work</div>'
    '<div class="d">Agents plan, clean, and analyze. Code checks every number, and Rex checks '
    "the reasoning.</div></div>"
    '<div class="step"><div class="k">Step 3</div><div class="t">Take it with you</div>'
    '<div class="d">A report (HTML and PDF), the cleaned data, and a script that repeats the '
    "cleaning.</div></div></div>"
)


def drink_html(kind: str) -> str:
    if kind not in ("tea", "coffee"):
        return '<div class="drinks"></div>'
    return (
        f'<div class="drinks drink-choice"><svg viewBox="-34 -34 68 60">{drink(kind)}</svg></div>'
    )


def show_shared_report(request: gr.Request):
    """Links like /?report=<job id> open a shared report inside the app (and skip the intro)."""
    job_id = (request.query_params.get("report") or "").strip() if request else ""
    if not job_id:
        return gr.update(visible=False), gr.update(visible=True)
    try:
        report = fetch_shared_report(get_settings(), job_id)
    except Exception:
        return (
            gr.update(
                value="<p><b>That shared report couldn't be found.</b> The link may be "
                "mistyped, or the report was removed.</p>",
                visible=True,
            ),
            gr.update(visible=False),
        )
    return gr.update(value=report_frame(report), visible=True), gr.update(visible=False)


def send_review(stars, comment: str, name: str, results_md: str):
    if not stars:
        gr.Warning("Pick a star rating first.")
        return gr.update(), gr.update()
    context = {"finished": "The run stopped" not in (results_md or "")}
    try:
        save_review(get_settings(), review_record(int(stars), comment, name, context))
    except Exception:
        gr.Warning("Your review couldn't be saved just now. Please try again in a moment.")
        return gr.update(), gr.update()
    thanks = (
        "### Thank you!\n\nYour review went straight to Aliza. It really helps. "
        "Hope you enjoyed your visit to the Mosaic office."
    )
    return gr.update(value=thanks, visible=True), gr.update(visible=False)


def build_app() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="Mosaic EDA") as demo:
        shared_view = gr.HTML(visible=False)

        with gr.Column(visible=True, elem_classes="m-screen") as welcome:
            gr.HTML(WELCOME)
            with gr.Row():
                gr.HTML("")
                start_btn = gr.Button("Start", variant="primary", size="lg", scale=0, min_width=200)
                gr.HTML("")

        with gr.Column(visible=False, elem_classes="m-screen") as upload_screen:
            gr.HTML(STEPS)
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    upload_art = gr.HTML(upload_icon())
                    with gr.Tab("Upload"):
                        upload = gr.File(
                            label="A table, text, log, image, audio, video, or a zip of them",
                            file_types=[
                                ".csv", ".tsv", ".txt", ".log", ".xlsx", ".xls", ".jsonl",
                                ".zip", ".jpg", ".jpeg", ".png", ".webp", ".wav", ".mp3",
                                ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".webm", ".mkv",
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
                    share = gr.Checkbox(
                        label="Save a public copy of the report and get a share link",
                        info="Only the HTML and PDF reports are saved (quotes are short and "
                        "masked); your data isn't.",
                        value=False,
                        visible=sharing_configured(settings),
                    )
                    with gr.Accordion("Settings", open=False):
                        user_key = gr.Textbox(
                            label="Your own Gemini API key (optional)",
                            type="password",
                            info="Used for this run only, never stored.",
                        )
                        color_blind = gr.Checkbox(label="Color-blind-friendly colors", value=False)
                with gr.Column(scale=2):
                    drink_kind = gr.Radio(
                        choices=[("Tea", "tea"), ("Coffee", "coffee"), ("No thanks", "none")],
                        value="tea",
                        label="Tea or coffee while the team works?",
                    )
                    drink_art = gr.HTML(drink_html("tea"))
                    run_btn = gr.Button("Analyze", variant="primary", size="lg")
                    gr.Markdown("Or try an example:")
                    example_btns = [
                        (gr.Button(text, size="sm"), path, g) for text, path, g in EXAMPLES
                    ]
                    replays = replay.available()
                    replay_btns = []
                    if replays:
                        gr.Markdown("Or watch a recorded run (no quota needed):")
                        replay_btns = [
                            (gr.Button(f"Replay: {label}", size="sm"), name)
                            for name, label in replays
                        ]
                    runs_left = gr.Markdown(runs_left_text(settings))
            with gr.Group(visible=False) as choice_panel:
                choice_text = gr.Markdown("")
                mixed = gr.Radio(
                    choices=[
                        ("Analyze each type, then link them", "group"),
                        ("Only the main type", "dominant"),
                    ],
                    value="group",
                    label="How should Mosaic analyze it?",
                )
                continue_btn = gr.Button("Continue", variant="primary")

        with gr.Column(visible=False, elem_classes="m-screen") as office_screen:
            gr.HTML(office_svg())
            counters = gr.Markdown("")
            with gr.Accordion("What are the agents doing?", open=False):
                feed = gr.Chatbot(label="Live log", height=380)
            office_box = gr.Textbox(elem_id="office-state", elem_classes="m-hidden", label="")

        with gr.Column(visible=False, elem_classes="m-screen") as results_screen:
            tiles = gr.HTML("")
            results = gr.Markdown(visible=False)
            plots = []
            for _ in range(MAX_PLOTS // 2):  # two charts per row
                with gr.Row():
                    plots += [gr.Plot(visible=False), gr.Plot(visible=False)]
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
            again_btn = gr.Button("Analyze another dataset")
            with gr.Group(elem_classes="review-box"):
                gr.Markdown(
                    "### How was your visit?\n\nA quick rating or a suggestion goes straight "
                    "to Aliza, privately."
                )
                with gr.Column() as review_form:
                    stars = gr.Radio(
                        choices=[("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5)],
                        label="Your rating",
                        elem_classes="stars",
                    )
                    comment = gr.Textbox(
                        label="Review or suggestion (optional)", lines=3, max_length=2000
                    )
                    name = gr.Textbox(label="Your name (optional)", max_length=80)
                    send_btn = gr.Button("Send to Aliza", variant="primary")
                review_done = gr.Markdown(visible=False)

        gr.Markdown(
            f"<small>Version {__version__} · Built with CrewAI and Gradio · A project by "
            "Aliza Saadi</small>"
        )

        screens = [welcome, upload_screen, office_screen, results_screen]
        outputs = [
            feed, counters, results, tiles, downloads, *plots, gallery, runs_left, choice_panel,
            choice_text, *screens, office_box,
        ]  # fmt: skip
        ask = gr.State("ask")
        common = [upload, url, goal, user_key]
        extra = [share, drink_kind, color_blind]

        start_btn.click(lambda: show("upload"), None, screens)
        again_btn.click(lambda: show("upload"), None, screens)
        upload.change(lambda f: upload_icon(done=bool(f)), upload, upload_art)
        drink_kind.change(drink_html, drink_kind, drink_art)
        color_blind.change(
            None, color_blind, None,
            js="(on) => { document.body.classList.toggle('cb-safe', on); return []; }",
        )  # fmt: skip
        run_btn.click(run_analysis, [*common, ask, *extra], outputs)
        continue_btn.click(run_analysis, [*common, mixed, *extra], outputs)
        for button, path, example_goal in example_btns:
            button.click(lambda p=path, g=example_goal: (str(p), g), None, [upload, goal]).then(
                run_analysis, [*common, ask, *extra], outputs
            )
        for button, name_ in replay_btns:
            button.click(
                lambda d, c, n=name_: (yield from replay_analysis(n, d, c)),
                [drink_kind, color_blind],
                outputs,
            )
        send_btn.click(send_review, [stars, comment, name, results], [review_done, review_form])
        demo.load(show_shared_report, None, [shared_view, welcome])
    return demo


def launch() -> None:
    settings = get_settings()
    Path(settings.workspace_root).mkdir(parents=True, exist_ok=True)
    build_app().queue(default_concurrency_limit=settings.max_concurrent_jobs).launch(
        allowed_paths=[str(settings.workspace_root), str(EXAMPLE_CSV.parent)],  # examples
        max_file_size=f"{settings.max_input_mb}mb",
        theme=harvest_theme(),
        css_paths=css_paths(),
        head=page_head(),
    )
