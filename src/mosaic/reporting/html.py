"""Render the HTML report from the Flow's results."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mosaic.ui.palette import HARVEST

_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_report(path: Path, **context: Any) -> Path:
    charts = context.pop("charts", [])
    html = _ENV.get_template("report.html.j2").render(
        p=HARVEST,
        generated=time.strftime("%d %B %Y"),
        charts=charts,
        # </script> inside JSON would end the script tag early
        charts_json=json.dumps(charts).replace("</", "<\\/"),
        **context,
    )
    path.write_text(html, encoding="utf-8")
    return path
