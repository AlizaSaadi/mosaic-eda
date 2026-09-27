"""Static PNG versions of the report's charts, for the PDF.

The HTML report draws Plotly figures in the browser. A PDF can't run JavaScript, so the
same figure JSON is redrawn here with matplotlib: bars (vertical, horizontal, grouped, and
histograms), scatter plots, and heatmaps, with their titles, readings, axis titles, and
reference lines. Any other kind of trace is skipped rather than drawn wrong.
"""

from __future__ import annotations

import base64
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on servers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mosaic.ui.palette import HARVEST

MAX_LABELS = 15
SUPPORTED = {"bar", "scatter", "heatmap"}


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or "")
    return str(value or "")


def _title(fig: dict[str, Any]) -> str:
    return _text(fig.get("layout", {}).get("title", ""))


def _subtitle(fig: dict[str, Any]) -> str:
    title = fig.get("layout", {}).get("title")
    return _text(title.get("subtitle")) if isinstance(title, dict) else ""


def _axis_title(fig: dict[str, Any], axis: str) -> str:
    return _text((fig.get("layout", {}).get(axis) or {}).get("title"))


def _values(trace: dict[str, Any], axis: str) -> list:
    values = trace.get(axis, [])
    if isinstance(values, dict):  # Plotly stores numeric arrays as base64 typed arrays
        if "bdata" not in values:
            return []
        data = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values.get("dtype", "f8"))
        if "shape" in values:
            shape = [int(n) for n in str(values["shape"]).split(",")]
            return data.reshape(shape).tolist()
        return data.tolist()
    return list(values or [])


def _bars(ax, traces: list[dict]) -> None:
    colors = HARVEST["chart"]
    grouped = len(traces) > 1
    for i, trace in enumerate(traces):
        x, y = _values(trace, "x"), _values(trace, "y")
        if not x or not y:
            continue
        color = (trace.get("marker") or {}).get("color") or colors[i % len(colors)]
        if isinstance(color, list):
            color = colors[i % len(colors)]
        labels = trace.get("text") if isinstance(trace.get("text"), list) else None
        if trace.get("orientation") == "h":
            bars = ax.barh([str(v) for v in y], x, color=color, label=trace.get("name"))
            if labels:
                ax.bar_label(bars, labels=labels, padding=3, fontsize=7.5, color=HARVEST["text"])
            continue
        numeric = all(isinstance(v, int | float) for v in x)
        if numeric and not grouped:  # a histogram: bars at bin centers with their widths
            widths = _values(trace, "width") or None
            ax.bar(x, y, width=widths or 0.8, color=color, edgecolor="white", linewidth=0.5)
            continue
        positions = list(range(len(x)))
        offset = (i - (len(traces) - 1) / 2) * (0.8 / len(traces)) if grouped else 0
        bars = ax.bar(
            [p + offset for p in positions],
            y,
            width=0.8 / len(traces) if grouped else 0.8,
            color=color,
            label=trace.get("name"),
        )
        if labels:
            ax.bar_label(bars, labels=labels, padding=2, fontsize=7.5, color=HARVEST["text"])
        ax.set_xticks(positions)
        if len(x) > MAX_LABELS:  # too many names to read: leave the bars unlabeled
            ax.set_xticklabels([])
            continue
        tilt = len(x) > 6
        ax.set_xticklabels([str(v) for v in x], rotation=30 if tilt else 0,
                           ha="right" if tilt else "center")  # fmt: skip
    if grouped:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                  ncol=len(traces))  # fmt: skip


def _heatmap(figure, ax, trace: dict) -> None:
    z = np.asarray(_values(trace, "z"), dtype=float)
    names = [str(v) for v in _values(trace, "x")]
    scale = [c for _, c in trace.get("colorscale") or []] or HARVEST["diverging"]
    cmap = LinearSegmentedColormap.from_list("mosaic", scale)
    image = ax.imshow(z, cmap=cmap, vmin=trace.get("zmin", -1), vmax=trace.get("zmax", 1))
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    if len(names) <= 10:
        for (r, c), v in np.ndenumerate(z):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else HARVEST["text"])  # fmt: skip
    figure.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    ax.grid(False)


def _reference_lines(ax, fig: dict[str, Any]) -> None:
    """Vertical lines (median, mean, cutoffs) and their labels."""
    layout = fig.get("layout", {})
    notes = [a for a in layout.get("annotations", []) if a.get("xref") == "x"]
    for shape in layout.get("shapes", []):
        if shape.get("type") == "rect" and shape.get("xref") == "x":
            ax.axvspan(
                shape["x0"], shape["x1"], color=HARVEST["highlight"], alpha=0.18, lw=0, zorder=0
            )
        if shape.get("type") != "line" or shape.get("x0") != shape.get("x1"):
            continue
        line = shape.get("line") or {}
        x = shape["x0"]
        style = {"dash": "--", "dot": ":"}.get(line.get("dash"), "-")
        ax.axvline(x, color=line.get("color", HARVEST["text"]), ls=style, lw=1.1)
        label = next((a.get("text") for a in notes if a.get("x") == x), "").replace("<br>", " ")
        if label:
            ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(3, -10),
                        textcoords="offset points", fontsize=7.5,
                        color=line.get("color", HARVEST["text"]))  # fmt: skip


def render_png(fig: dict[str, Any], path: Path, width: float = 7.0, height: float = 3.4) -> bool:
    """Draw a Plotly figure as a PNG. Returns False if it has traces this can't draw."""
    traces = fig.get("data", [])
    kinds = {t.get("type", "bar") for t in traces}
    if not traces or not kinds <= SUPPORTED or len(kinds) > 1:
        return False
    kind = kinds.pop()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    if kind == "heatmap":
        height = max(height, 4.2)
    figure, ax = plt.subplots(figsize=(width, height), dpi=150)
    if kind == "bar":
        _bars(ax, traces)
    elif kind == "scatter":
        for i, trace in enumerate(traces):
            color = (trace.get("marker") or {}).get("color") or HARVEST["chart"][i]
            ax.scatter(_values(trace, "x"), _values(trace, "y"), s=14, alpha=0.7, color=color)
    else:
        _heatmap(figure, ax, traces[0])
    if kind != "heatmap":
        _reference_lines(ax, fig)
        ax.grid(axis="y" if traces[0].get("orientation") != "h" else "x", alpha=0.25)
    subtitle = "\n".join(textwrap.fill(p, 110) for p in _subtitle(fig).split("<br>") if p)
    figure.suptitle(_title(fig), x=0.012, ha="left", fontsize=10.5, color=HARVEST["text"])
    if subtitle:
        ax.set_title(subtitle, loc="left", fontsize=8, color=HARVEST["muted"], x=0)
    xaxis = fig.get("layout", {}).get("xaxis") or {}
    if xaxis.get("tickvals") and xaxis.get("ticktext"):  # e.g. a log-scale histogram
        ax.set_xticks(_values(xaxis, "tickvals"), xaxis["ticktext"])
    ax.set_xlabel(_axis_title(fig, "xaxis"), color=HARVEST["muted"])
    ax.set_ylabel(_axis_title(fig, "yaxis"), color=HARVEST["muted"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=HARVEST["text"])
    figure.tight_layout()
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    return True
