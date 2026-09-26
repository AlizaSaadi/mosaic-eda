"""Static PNG versions of the report's charts, for the PDF.

The HTML report draws Plotly figures in the browser. A PDF can't run JavaScript, so the
same figure JSON is redrawn here with matplotlib. MOSAIC's charts are all bar charts
(counts, histograms, rankings, grouped before/after bars), which is what this supports;
anything else is skipped rather than drawn wrong.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on servers
import matplotlib.pyplot as plt

from mosaic.ui.palette import HARVEST

MAX_LABELS = 15


def _title(fig: dict[str, Any]) -> str:
    title = fig.get("layout", {}).get("title", "")
    return title.get("text", "") if isinstance(title, dict) else str(title or "")


def _values(trace: dict[str, Any], axis: str) -> list:
    values = trace.get(axis, [])
    if isinstance(values, dict):  # Plotly stores numeric arrays as base64 typed arrays
        if "bdata" not in values:
            return []
        data = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values.get("dtype", "f8"))
        return data.tolist()
    return list(values or [])


def render_png(fig: dict[str, Any], path: Path, width: float = 7.0, height: float = 3.2) -> bool:
    """Draw a Plotly bar figure as a PNG. Returns False if the figure isn't bars."""
    traces = [t for t in fig.get("data", []) if t.get("type", "bar") == "bar"]
    if not traces or len(traces) != len(fig.get("data", [])):
        return False
    colors = HARVEST["chart"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, ax = plt.subplots(figsize=(width, height), dpi=150)
    grouped = len(traces) > 1
    for i, trace in enumerate(traces):
        x, y = _values(trace, "x"), _values(trace, "y")
        if not x or not y:
            continue
        color = (trace.get("marker") or {}).get("color") or colors[i % len(colors)]
        if isinstance(color, list):
            color = colors[i % len(colors)]
        if trace.get("orientation") == "h":
            ax.barh([str(v) for v in y], x, color=color, label=trace.get("name"))
            continue
        numeric = all(isinstance(v, int | float) for v in x)
        if numeric and not grouped:  # a histogram: bars at bin centers with their widths
            widths = _values(trace, "width") or None
            ax.bar(x, y, width=widths or 0.8, color=color, edgecolor="white", linewidth=0.5)
        else:
            positions = list(range(len(x)))
            offset = (i - (len(traces) - 1) / 2) * (0.8 / len(traces)) if grouped else 0
            ax.bar(
                [p + offset for p in positions],
                y,
                width=0.8 / len(traces) if grouped else 0.8,
                color=color,
                label=trace.get("name"),
            )
            ax.set_xticks(positions)
            if len(x) > MAX_LABELS:  # too many names to read: leave the bars unlabeled
                ax.set_xticklabels([])
                ax.set_xlabel(f"{len(x)} items (names in the HTML report)")
                continue
            ax.set_xticklabels([str(v) for v in x], rotation=30 if len(x) > 6 else 0, ha="right"
                               if len(x) > 6 else "center")  # fmt: skip
    ax.set_title(_title(fig), loc="left", fontsize=11, color=HARVEST["text"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=HARVEST["text"])
    ax.grid(axis="y" if traces[0].get("orientation") != "h" else "x", alpha=0.25)
    if grouped:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=len(traces))  # fmt: skip
    figure.tight_layout()
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    return True
