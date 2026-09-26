"""The PDF report, built directly with fpdf2 (pure Python, no system libraries).

It has the same content as the HTML report: summary, quality tiles, findings with their
evidence, per-type sections in group mode, the self-check timeline, the cleaning steps,
the charts (redrawn with matplotlib), key tables, and the notes. Fonts come from
matplotlib's bundled DejaVu family, so any language in the data prints correctly.
"""

from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

from mosaic.reporting.static_charts import render_png
from mosaic.ui.palette import HARVEST

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
SEVERITY = {"critical": "critical", "warning": "warning", "info": "info"}
MAX_TIMELINE = 14
MAX_TABLE_ROWS = 25


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _clip(text: Any, n: int = 400) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


class ReportPDF(FPDF):
    def __init__(self, source_name: str) -> None:
        super().__init__(format="A4")
        self.source_name = source_name
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(auto=True, margin=16)
        for style, name in (
            ("", "DejaVuSans.ttf"),
            ("B", "DejaVuSans-Bold.ttf"),
            ("I", "DejaVuSans-Oblique.ttf"),
            ("BI", "DejaVuSans-BoldOblique.ttf"),
        ):
            self.add_font("DejaVu", style, str(FONT_DIR / name))
        self.add_font("Mono", "", str(FONT_DIR / "DejaVuSansMono.ttf"))
        self.set_title(f"MOSAIC EDA report: {source_name}")
        self.set_creator("MOSAIC EDA")

    # ---- page furniture ----

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*_rgb(HARVEST["muted"]))
        self.cell(0, 6, f"MOSAIC EDA · {_clip(self.source_name, 70)} · page {self.page_no()}",
                  align="C")  # fmt: skip

    # ---- building blocks ----

    def color(self, key: str) -> None:
        self.set_text_color(*_rgb(HARVEST[key]))

    def heading(self, text: str) -> None:
        if self.get_y() > self.h - 50:
            self.add_page()
        self.ln(4)
        self.set_font("DejaVu", "B", 14)
        self.color("text")
        self.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*_rgb(HARVEST["accent"]))
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.l_margin + 24, self.get_y())
        self.ln(3)

    def para(self, text: str, size: float = 10, style: str = "", color: str = "text") -> None:
        self.set_font("DejaVu", style, size)
        self.color(color)
        self.multi_cell(0, size * 0.55, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def small(self, text: str) -> None:
        self.para(text, 8.5, color="muted")

    def mono(self, text: str) -> None:
        self.set_font("Mono", "", 8)
        self.color("muted")
        self.multi_cell(0, 4.2, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def chip(self, severity: str) -> None:
        key = SEVERITY.get(severity, "info")
        self.set_font("DejaVu", "B", 7.5)
        self.set_fill_color(*_rgb(HARVEST[key]))
        self.set_text_color(255, 255, 255)
        width = self.get_string_width(severity.upper()) + 5
        self.cell(width, 5, severity.upper(), fill=True, align="C")
        self.cell(2, 5, "")

    def tiles(self, items: list[tuple[str, str]]) -> None:
        width = (self.w - self.l_margin - self.r_margin - 3 * (len(items) - 1)) / len(items)
        y = self.get_y()
        for i, (label, value) in enumerate(items):
            x = self.l_margin + i * (width + 3)
            self.set_fill_color(*_rgb(HARVEST["surface"]))
            self.set_draw_color(*_rgb(HARVEST["border"]))
            self.rect(x, y, width, 18, style="DF")
            self.set_xy(x + 3, y + 2)
            self.set_font("DejaVu", "", 7.5)
            self.color("muted")
            self.cell(width - 6, 4, label)
            self.set_xy(x + 3, y + 8)
            self.set_font("DejaVu", "B", 13)
            self.color("text")
            self.cell(width - 6, 7, value)
        self.set_xy(self.l_margin, y + 22)

    def table(self, header: list[str], rows: list[list[Any]], widths: tuple[float, ...]) -> None:
        self.set_font("DejaVu", "", 8)
        self.color("text")
        self.set_draw_color(*_rgb(HARVEST["border"]))
        self.set_fill_color(255, 255, 255)  # chips leave their color behind
        with super().table(
            cell_fill_mode="NONE",
            col_widths=widths,
            line_height=4.5,
            headings_style=FontFace(
                emphasis="BOLD", color=_rgb(HARVEST["text"]), fill_color=_rgb(HARVEST["bg"])
            ),
            first_row_as_headings=True,
            borders_layout="HORIZONTAL_LINES",
        ) as t:
            head = t.row()
            for h in header:
                head.cell(h)
            for r in rows[:MAX_TABLE_ROWS]:
                row = t.row()
                for value in r:
                    row.cell(_clip(value, 160))
        if len(rows) > MAX_TABLE_ROWS:
            self.small(f"... and {len(rows) - MAX_TABLE_ROWS} more (see the HTML report).")
        self.ln(2)

    def image_file(self, path: Path, height: float = 70) -> None:
        if self.get_y() + height > self.h - 18:
            self.add_page()
        self.image(str(path), x=self.l_margin, w=self.w - self.l_margin - self.r_margin)
        self.ln(3)


def _finding(pdf: ReportPDF, f: dict[str, Any]) -> None:
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()
    pdf.chip(f.get("severity", "info"))
    pdf.set_font("DejaVu", "B", 10.5)
    pdf.color("text")
    pdf.multi_cell(0, 5, f["title"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.para(f["statement"], 9.5)
    if f.get("recommendation"):
        pdf.para(f"Recommendation: {f['recommendation']}", 9, "I", "muted")
    if f.get("evidence"):
        pdf.mono("Evidence: " + ", ".join(f["evidence"]))
    pdf.ln(2.5)


def _extras(pdf: ReportPDF, extras: dict[str, Any], unit: str, work: Path) -> None:
    balance = extras.get("balance")
    if balance:
        pdf.heading("Label balance" if extras.get("text") else "Class balance")
        rows = [[k, v["count"], v["share"]] for k, v in balance["classes"].items()]
        pdf.table(["Class", unit.capitalize(), "Share %"], rows, (60, 30, 30))
        pdf.small(f"Largest class / smallest class: {balance['imbalance_ratio']}")
    text = extras.get("text") or {}
    labels = text.get("labels") or {}
    if labels.get("mismatches"):
        pdf.heading("Documents that read like another label (checked by code)")
        rows = [[m["document"], m["label"], m["closest"], m["text"]] for m in labels["mismatches"]]
        pdf.table(["Document", "Label", "Reads like", "Text (masked)"], rows, (45, 22, 22, 89))
    topics = (text.get("topics") or {}).get("topics") or {}
    if topics:
        pdf.heading("Topics")
        rows = [[i, ", ".join(t["words"]), t["share"]] for i, t in enumerate(topics.values(), 1)]
        pdf.table(["Topic", "Strongest words", "Share %"], rows, (18, 130, 30))
    transcripts = extras.get("transcripts") or {}
    if transcripts.get("mismatches"):
        pdf.heading("Clips whose transcript doesn't match the folder label")
        rows = [[m["path"], m["label"], m["heard"]] for m in transcripts["mismatches"]]
        pdf.table(["Clip", "Label", "Heard"], rows, (70, 30, 78))
    timeline = extras.get("timeline") or {}
    if timeline.get("videos"):
        pdf.heading("Timeline: scenes and speech")
        rows = [
            [path if s["scene"] == 1 else "", s["scene"], f"{s['start']}-{s['end']}", s["said"]]
            for path, scenes in list(timeline["videos"].items())[:5]
            for s in scenes
        ]
        pdf.table(["Video", "Scene", "Seconds", "Said"], rows, (58, 16, 26, 78))
    pictures = [(s["class"], s["b64"]) for s in extras.get("sheets") or []]
    pictures += [(s["class"], s["b64"]) for s in extras.get("spectrograms") or []]
    if pictures:
        pdf.heading("Contact sheets, keyframes, or spectrograms")
        for i, (caption, b64) in enumerate(pictures[:6]):
            path = work / f"picture_{i}.png"
            path.write_bytes(base64.b64decode(b64))
            pdf.small(caption)
            pdf.image_file(path, height=60)


def _group(pdf: ReportPDF, parts: list[dict[str, Any]], links: dict[str, Any] | None) -> None:
    pdf.heading("Each data type")
    pdf.small(
        "Each type was profiled, cleaned, analyzed, and reviewed by its own crew. Their full "
        "reports are in cleaned_by_type.zip."
    )
    for p in parts:
        pdf.set_font("DejaVu", "B", 11)
        pdf.color("text")
        line = f"{p['modality'].capitalize()} · {p['files']} file(s)"
        if p["status"] == "done":
            line += (
                f" · quality {p['quality_before']} → {p['quality_after']} · "
                f"{p['unit']} {p['items_before']} → {p['items_after']}"
            )
        pdf.multi_cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if p["status"] != "done":
            pdf.para(f"Not analyzed: {p['error']}", 9, color="muted")
            continue
        for f in p["finding_list"]:
            pdf.para(f"• {f['title']}: {f['statement']}", 9)
        pdf.ln(2)
    if links:
        pdf.heading("Table-to-file links")
        pdf.para(
            f"{links['table']}, column '{links['file_column']}': {links['rows_matched']} of "
            f"{links['rows']} rows match a file; {links['rows_missing_file']} point to missing "
            f"files; {links['files_unreferenced']} of {links['files']} files have no row"
            + (
                f"; column '{links['label_column']}' disagrees with the folder for "
                f"{links['label_disagreements']} rows."
                if links["label_column"]
                else "."
            ),
            9,
        )
        rows = [["Row names a missing file", v] for v in links["missing_examples"]]
        rows += [["File with no row", v] for v in links["unreferenced_examples"]]
        rows += [
            [f"Label disagrees (row {d['row']})",
             f"{d['file']}: table '{d['table_label']}', folder '{d['folder']}'"]
            for d in links["disagreement_examples"]
        ]  # fmt: skip
        if rows:
            pdf.table(["Problem", "Example"], rows, (52, 126))


def render_pdf(path: Path, **ctx: Any) -> Path:
    """Build report.pdf from the same context as the HTML report."""
    pdf = ReportPDF(ctx["source_name"])
    pdf.add_page()
    narrative = ctx.get("narrative") or {}
    extras = ctx.get("extras") or {}
    unit = ctx.get("unit", "rows")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        pdf.set_font("DejaVu", "", 8)
        pdf.color("accent")
        pdf.cell(0, 5, f"MOSAIC EDA REPORT · {time.strftime('%d %B %Y')}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # fmt: skip
        pdf.para(narrative.get("headline") or ctx["source_name"], 18, "B")
        if narrative.get("executive_summary"):
            pdf.ln(1)
            pdf.para(narrative["executive_summary"], 10, color="muted")
        pdf.ln(3)
        rows_before, rows_after = ctx.get("rows_before", 0), ctx.get("rows_after", 0)
        pdf.tiles(
            [
                ("Data quality before", f"{ctx.get('quality_raw')}/100"),
                ("After cleaning", f"{ctx.get('quality_clean')}/100"),
                (unit.capitalize(), f"{rows_before:,} → {rows_after:,}"),
                ("Numbers fact-checked", str(ctx.get("facts_verified", 0))),
            ]
        )
        triage = ctx.get("triage")
        if triage:
            target = f" Target column: {ctx['target']}." if ctx.get("target") else ""
            pdf.para(f"What this dataset is: {triage['dataset_description']}{target}", 9.5)

        pdf.heading("Cross-type findings" if extras.get("group") else "Findings")
        for f in ctx.get("findings") or []:
            _finding(pdf, f)
        if extras.get("group"):
            _group(pdf, extras["group"], extras.get("links"))
        if narrative.get("next_steps"):
            pdf.heading("Next steps")
            for i, step in enumerate(narrative["next_steps"], 1):
                pdf.para(f"{i}. {step}", 9.5)

        pdf.heading("How the agents checked themselves")
        counters = ctx.get("counters") or {}
        pdf.small(
            f"Model calls {counters.get('model_calls', 0)} · guardrail catches "
            f"{counters.get('guardrail_catches', 0)} · self-corrections "
            f"{counters.get('self_corrections', 0)} · review rounds "
            f"{counters.get('review_rounds', 0)} · model switches {counters.get('fallbacks', 0)}"
        )
        checks = ctx.get("checks") or []
        if not checks:
            pdf.para("No corrections were needed: every plan and finding passed first time.", 9)
        for c in checks[:MAX_TIMELINE]:
            pdf.para(f"+{c['t']}s  {c['title']}", 9, "B")
            if c.get("detail"):
                pdf.mono(_clip(c["detail"], 300))
        if len(checks) > MAX_TIMELINE:
            pdf.small(f"... and {len(checks) - MAX_TIMELINE} more in the HTML report.")

        steps = ctx.get("steps") or []
        if steps or ctx.get("plan_summary"):
            pdf.heading("Cleaning applied")
            pdf.para(ctx.get("plan_summary", ""), 9.5)
        if steps:
            rows = [
                [s.index, s.op, s.risk, ", ".join(s.columns) or "all",
                 f"{s.rows_before:,} → {s.rows_after:,}" if s.rows_after != s.rows_before
                 else f"{s.rows_before:,}", s.rationale]
                for s in steps
            ]  # fmt: skip
            pdf.table(["#", "Operation", "Risk", "Columns", unit.capitalize(), "Why"], rows,
                      (8, 46, 18, 22, 26, 58))  # fmt: skip
            pdf.small("The same steps are in cleaning_pipeline.py, which reruns on the full data.")

        charts = ctx.get("charts") or []
        drawn = []
        for i, fig in enumerate(charts):
            png = work / f"chart_{i}.png"
            if render_png(fig, png):
                drawn.append(png)
        if drawn:
            pdf.heading("Charts")
            for png in drawn:
                pdf.image_file(png)

        _extras(pdf, extras, unit, work)

        notes = ctx.get("notes") or []
        if notes:
            pdf.heading("Notes")
            for n in notes:
                pdf.para(f"• {n}", 9)
        pdf.ln(4)
        models = ", ".join(ctx.get("models") or []) or "n/a"
        pdf.small(
            "Every number in the findings was checked against the evidence artifacts listed. "
            f"Models used: {models}."
        )
        pdf.output(str(path))
    return path
