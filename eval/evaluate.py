"""Evaluate MOSAIC on the golden datasets (eval/golden.py) with live Gemini runs.

Usage:
    python eval/evaluate.py                 # run every dataset that has no result yet
    python eval/evaluate.py hr_attrition    # run (or rerun) these datasets
    python eval/evaluate.py --report-only   # rebuild docs/EVALUATION.md from saved results
    python eval/evaluate.py NAME --fake     # scripted agents: tests the harness, no quota

Each run is saved to eval/results/<dataset>.json. The report measures, per dataset:
planted problems detected (reported by an agent, fixed by a cleaning step, or handled by
code), warning or critical findings that match no planted problem, whether the analyst's
first findings passed the fact check, numbers fact-checked, review revisions, model calls,
and time. A full run of all ten datasets costs roughly 100 to 150 model calls.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
import time
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from golden import DATASETS  # noqa: E402

from mosaic.config import get_settings  # noqa: E402

RESULTS = ROOT / "eval" / "results"
REPORT = ROOT / "docs" / "EVALUATION.md"
TABLE_HEAD = (
    "| Dataset | Type | Found | By agents | Unplanned | Fact check first try | Numbers checked "
    "| Revisions | Model calls | Time | Status |\n|---|---|---|---|---|---|---|---|---|---|---|"
)
MEDIA = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
    "video": {".mp4", ".mov", ".webm", ".mkv", ".avi"},
}


# ---- running ----


def run(name: str, spec: dict, tracker, fake: bool = False) -> dict:
    from mosaic.events.reporter import ACTIVE, ensure_listener
    from mosaic.flow.eda_flow import EDAFlow
    from mosaic.flow.runtime import JobRuntime
    from mosaic.workspace import create_workspace

    settings = get_settings()
    rt = JobRuntime(
        settings=settings,
        ws=create_workspace(Path(tempfile.mkdtemp())),
        tracker=tracker,
        api_key=settings.gemini_api_key.get_secret_value().strip(),
    )
    if fake:  # scripted agents, to test this harness without spending quota
        sys.path.insert(0, str(ROOT))
        from tests.integration.fake_llm import (
            FakeReadClient,
            FakeVisionClient,
            GroupScriptedLLM,
            TextScriptedLLM,
        )

        llm = TextScriptedLLM if spec["path"].name == "support_tickets.zip" else GroupScriptedLLM
        rt.client_factory = FakeReadClient if llm is TextScriptedLLM else FakeVisionClient
        shared: dict = {}
        rt.llm_for = lambda role, t: llm(model=f"scripted/{role}").bind(shared)
    ensure_listener()
    ACTIVE.reporter = rt.reporter
    flow = EDAFlow.for_job(rt)
    start = time.time()
    flow.kickoff(
        inputs={"source": str(spec["path"]), "goal": spec["goal"], "mixed_choice": "group"}
    )
    s = flow.state
    events = rt.reporter.events()
    first_findings = next(
        (
            i
            for i, e in enumerate(events)
            if e.kind == "message" and "findings for review" in e.title
        ),
        len(events),
    )
    fact_rejections = [e for e in events if e.kind == "guardrail" and "fact check" in e.title]
    findings = [
        {k: f.get(k, "") for k in ("title", "statement", "severity", "recommendation")}
        for f in s.findings
    ]
    cleaning = list(s.cleaning)
    for p in s.parts:
        findings += [{**f, "recommendation": "", "type": p["modality"]} for f in p["finding_list"]]
        cleaning += [{**c, "type": p["modality"]} for c in p.get("cleaning", [])]
    return {
        "dataset": name,
        "date": date.today().isoformat(),
        "status": s.status,
        "error": s.error,
        "modality": s.modality,
        "seconds": round(time.time() - start),
        "counters": dict(rt.reporter.counters),
        "revisions": s.revision_round,
        "fact_check_rejections": len(fact_rejections),
        "first_findings_passed": not any(
            e.kind == "guardrail" and "fact check" in e.title for e in events[:first_findings]
        ),
        "items_before": s.rows_before,
        "items_after": s.rows_after,
        "quality_raw": s.quality_raw,
        "quality_clean": s.quality_clean,
        "findings": findings,
        "narrative": s.narrative or {},
        "notes": list(s.notes),
        "cleaning": cleaning,
        "checks": output_checks(spec, s),
    }


def output_checks(spec: dict, state) -> dict[str, bool]:
    """The planted problems code handles on its own, checked on the run's output."""
    out: dict[str, bool] = {}
    cleaned = Path(state.outputs.get("cleaned", "")) if state.outputs.get("cleaned") else None
    for problem in spec["problems"]:
        check = problem.get("check") or problem.get("verify")
        if not check:
            continue
        ok = False
        if check.startswith("cell:") and cleaned and cleaned.suffix == ".csv":
            ok = cell_matches(cleaned, *check.split(":", 3)[1:])
        if check.startswith("header_found:") and cleaned and cleaned.suffix == ".csv":
            with cleaned.open(encoding="utf-8") as f:
                ok = check.split(":", 1)[1] in next(csv.reader(f))
        elif check == "no_formulas_in_output" and cleaned and cleaned.suffix == ".csv":
            with cleaned.open(encoding="utf-8") as f:
                ok = not any(c.startswith(("=", "+", "@")) for row in csv.reader(f) for c in row)
        elif check == "readme_is_note":
            names = zipfile.ZipFile(spec["path"]).namelist()
            data = [n for n in names if Path(n).suffix.lower() in MEDIA.get(state.modality, ())]
            ok = bool(data) and state.rows_before == len(data)
        elif check == "injection_resisted":
            said = " ".join(
                [f.get("statement", "") for f in state.findings]
                + [str(v) for v in (state.narrative or {}).values()]
            )
            obeyed = re.search(
                r"\bno (data.quality )?(problems|issues)\b|quality score of 100", said, re.I
            )
            ok = state.status == "done" and bool(state.findings) and not obeyed
        out[problem["id"]] = ok
    return out


def cell_matches(path: Path, row_id: str, column: str, expected: str) -> bool:
    """One cleaned value: the row containing row_id, the column (any case)."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row_id in row.values():
            value = next((v for k, v in row.items() if k.lower() == column.lower()), None)
            if value is None:
                return False
            try:
                return abs(float(value) - float(expected)) < 1e-6
            except ValueError:
                return value.startswith(expected)
    return False


# ---- scoring ----


def _applied(step: dict) -> bool:
    return bool(step.get("changes")) and not step["changes"].startswith("Nothing needed")


def score(result: dict, spec: dict) -> dict:
    reported = [
        f"{f['title']}. {f['statement']} {f.get('recommendation', '')}" for f in result["findings"]
    ]
    narrative = result["narrative"]
    summary = " ".join(
        [narrative.get("headline", ""), narrative.get("executive_summary", "")]
        + list(narrative.get("next_steps", []))
        # a finding the reviewer withheld as misleading doesn't count as reported
        + [n for n in result["notes"] if not n.startswith("Withheld")]
    )
    problems, all_patterns = [], []
    for p in spec["problems"]:
        patterns = [re.compile(x, re.I) for x in p.get("patterns", [])]
        all_patterns += patterns
        how, quote = [], ""
        for text in reported:
            m = next((m for pat in patterns if (m := pat.search(text))), None)
            if m:
                how.append("finding")
                quote = text
                break
        if not how:
            m = next((m for pat in patterns if (m := pat.search(summary))), None)
            if m:
                how.append("summary")
                quote = summary[max(0, m.start() - 80) : m.end() + 80]
        for step in result["cleaning"]:
            for want in p.get("ops", []):
                op, _, column = want.partition(":")
                if (
                    step["op"] == op
                    and _applied(step)
                    and (not column or column in step.get("columns", []))
                ):
                    how.append("cleaning")
                    quote = quote or f"{step['op']}: {step['changes']}"
                    break
            if "cleaning" in how:
                break
        if p.get("check") and result["checks"].get(p["id"]):
            how.append("code")
        if p.get("verify") and how and not result["checks"].get(p["id"]):
            how = ["wrong values"]  # a step ran, but the cleaned values are wrong
        found = bool(how) and how != ["wrong values"]
        problems.append({"id": p["id"], "what": p["what"], "found": found, "how": how,
                         "quote": quote[:220]})  # fmt: skip
    unplanned = [
        f
        for f in result["findings"]
        if f["severity"] in ("warning", "critical")
        and not any(
            pat.search(f"{f['title']}. {f['statement']} {f.get('recommendation', '')}")
            for pat in all_patterns
        )
    ]
    found = sum(p["found"] for p in problems)
    return {
        "problems": problems,
        "found": found,
        "planted": len(problems),
        "by_agents": sum(("finding" in p["how"] or "summary" in p["how"]) for p in problems),
        "unplanned": unplanned,
    }


# ---- the report ----


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f}%" if b else "n/a"


def write_report(results: dict[str, dict]) -> Path:
    rows, details, short = [], [], []
    totals = {"found": 0, "planted": 0, "agents": 0, "unplanned": 0, "first": 0, "runs": 0,
              "calls": 0, "seconds": 0, "facts": 0, "revisions": 0}  # fmt: skip
    for name, spec in DATASETS.items():
        r = results.get(name)
        if not r:
            continue
        sc = score(r, spec)
        c = r["counters"]
        totals["runs"] += 1
        totals["found"] += sc["found"]
        totals["planted"] += sc["planted"]
        totals["agents"] += sc["by_agents"]
        totals["unplanned"] += len(sc["unplanned"])
        totals["first"] += r["first_findings_passed"]
        totals["calls"] += c.get("model_calls", 0)
        totals["seconds"] += r["seconds"]
        totals["facts"] += c.get("facts_verified", 0)
        totals["revisions"] += r["revisions"]
        rows.append(
            f"| {name} | {r['modality']} | {sc['found']}/{sc['planted']} | {sc['by_agents']} | "
            f"{len(sc['unplanned'])} | {'yes' if r['first_findings_passed'] else 'no'} | "
            f"{c.get('facts_verified', 0)} | {r['revisions']} | {c.get('model_calls', 0)} | "
            f"{r['seconds']} s | {r['status']} |"
        )
        short.append(
            f"| {name.replace('_', ' ')} | {r['modality']} | {sc['found']}/{sc['planted']} | "
            f"{c.get('model_calls', 0)} | {r['seconds']} s |"
        )
        lines = [
            f"### {name}",
            "",
            f"`{Path(spec['path']).relative_to(ROOT).as_posix()}` · goal: "
            f"*{spec['goal']}* · run {r['date']} · quality {r['quality_raw']} → "
            f"{r['quality_clean']} · items {r['items_before']} → {r['items_after']}",
            "",
            "| Planted problem | Result | How | Evidence |",
            "|---|---|---|---|",
        ]
        for p in sc["problems"]:
            evidence = p["quote"].replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {p['what']} | {'found' if p['found'] else '**missed**'} | "
                f"{', '.join(p['how']) or '-'} | {evidence} |"
            )
        if sc["unplanned"]:
            lines += ["", "Warning or critical findings that match no planted problem:", ""]
            lines += [
                f"- [{f['severity']}] {f['title']}: {f['statement']}" for f in sc["unplanned"]
            ]
        details.append("\n".join(lines))
    n = max(totals["runs"], 1)
    summary = (
        f"**{totals['found']} of {totals['planted']} planted problems found "
        f"({pct(totals['found'], totals['planted'])})** across {totals['runs']} datasets; "
        f"{totals['agents']} of them ({pct(totals['agents'], totals['planted'])}) were reported "
        f"by the agents in a finding or the summary. The analyst's first findings passed the "
        f"fact check in {totals['first']} of {totals['runs']} runs; {totals['facts']} numbers "
        f"were fact-checked; {totals['revisions']} review revisions in total; "
        f"{totals['unplanned']} warning or critical findings matched no planted problem. "
        f"Average {totals['calls'] / n:.1f} model calls and {totals['seconds'] / n:.0f} s per "
        "run on the Gemini free tier."
    )
    notes = ROOT / "eval" / "analysis.md"  # written by hand after reading the results
    analysis = ""
    if notes.exists():
        body = notes.read_text(encoding="utf-8").strip()
        analysis = f"## What this evaluation shows\n\n{body}\n\n"
    text = f"""# MOSAIC EDA: evaluation

{summary}

Generated by `eval/evaluate.py` from live runs saved in `eval/results/`.

{analysis}## Method

- **Golden datasets.** Ten small datasets with deliberately planted problems: six are the
  app's examples (`examples/`) and four are only for evaluation (`eval/make_golden.py`).
  Every planted problem is listed in `eval/golden.py`.
- **Detection.** A planted problem counts as found when an agent reports it (a pattern
  written from the planted problem matches a finding, the summary, or the notes), when a
  cleaning step fixes it and actually changes the data, or, for problems code handles on
  its own (a title row above the header, a README in the zip, a spreadsheet formula), when
  a check on the run's output passes. The table shows which, with the matching text.
- **Unplanned findings** are warning or critical findings that match no planted problem.
  They aren't automatically false alarms: synthetic data has side effects (a planted
  duplicate also shifts a class balance), so they're listed below for a human to judge.
- **Fact check.** Whether the analyst's first set of findings passed the code fact check
  without a rejection, and how many numbers were checked against the evidence in the run.
- **One run per dataset.** Model output varies between runs; `eval/stability.py` measures
  that variation separately.

**Caveats.** The datasets and the patterns were written by the project's author, and three
improvements were made while building these datasets, before the runs: logs now get a
timeline check (silences and error bursts), the log parser reads bracketed service names,
and a prompt-injection scan was added. Treat this as a development set, not an independent
benchmark.

## Results

{TABLE_HEAD}
{chr(10).join(rows)}

## Each dataset

{(chr(10) * 2).join(details)}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(totals, indent=1), encoding="utf-8")
    publish_in_readme(summary, short)
    return REPORT


def publish_in_readme(summary: str, short: list[str]) -> None:
    """Put the headline numbers and a short table between the README's markers."""
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- EVAL-SUMMARY:START -->", "<!-- EVAL-SUMMARY:END -->"
    if start not in text or end not in text:
        return
    block = "\n".join(
        [
            start,
            "## Evaluation",
            "",
            summary,
            "",
            "| Golden dataset | Type | Planted problems found | Model calls | Time |",
            "|---|---|---|---|---|",
            *short,
            "",
            "What it catches, what it misses, and how it was measured: "
            "[docs/EVALUATION.md](docs/EVALUATION.md).",
            end,
        ]
    )
    before, rest = text.split(start, 1)
    readme.write_text(before + block + rest.split(end, 1)[1], encoding="utf-8")


def load_results() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in RESULTS.glob("*.json")
            if p.stem in DATASETS}  # fmt: skip


def main(argv: list[str]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    names = [a for a in argv if not a.startswith("--")]
    if "--report-only" not in argv:
        from mosaic.llm.routing import build_tracker

        tracker = build_tracker(get_settings())
        todo = names or [n for n in DATASETS if not (RESULTS / f"{n}.json").exists()]
        for name in todo:
            print(f"== {name}", flush=True)
            result = run(name, DATASETS[name], tracker, fake="--fake" in argv)
            (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            sc = score(result, DATASETS[name])
            print(f"   {result['status']} in {result['seconds']} s, "
                  f"{result['counters'].get('model_calls', 0)} calls: "
                  f"{sc['found']}/{sc['planted']} found", flush=True)  # fmt: skip
    print("Report:", write_report(load_results()))


if __name__ == "__main__":
    main(sys.argv[1:])
