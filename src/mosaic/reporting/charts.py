"""Chart builders shared by every data type.

Each chart has a title, a one-line reading under it (the numbers someone would otherwise
have to work out from the bars), labeled axes, and hover text. Histograms mark the median,
the mean, and the middle half of the data; rankings are sorted with values on the bars;
class balances show where an even split would be. Figures are returned as Plotly JSON, the
form the evidence store, the app, the HTML report, and the PDF renderer all use.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from mosaic.ui.palette import HARVEST

HEIGHT = 380
GRID = "rgba(107,86,70,0.16)"
FONT = "Inter, system-ui, sans-serif"


def fmt(v: float) -> str:
    """A short, readable number: 1,234 or 12.3 or 0.042."""
    v = float(v)
    if not np.isfinite(v):
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10 or v == int(v):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.3g}"


def finish(
    fig: go.Figure,
    title: str,
    *,
    x: str,
    y: str,
    subtitle: str = "",
    height: int = HEIGHT,
    legend: bool = False,
) -> dict:
    """Apply the house style and return the figure as JSON. Both axis titles are required."""
    fig.update_layout(
        title={
            "text": title,
            "subtitle": {"text": subtitle, "font": {"size": 12.5, "color": HARVEST["muted"]}},
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 16, "family": "Fraunces, Georgia, serif", "color": HARVEST["text"]},
        },
        template="plotly_white",
        colorway=HARVEST["chart"],
        height=height,
        font={"family": FONT, "color": HARVEST["text"], "size": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 64, "r": 24, "t": 84 if subtitle else 60, "b": 64},
        showlegend=legend,
        legend={"orientation": "h", "y": -0.22, "x": 0},
        hoverlabel={"bgcolor": HARVEST["surface"], "bordercolor": HARVEST["border"],
                    "font": {"family": FONT, "color": HARVEST["text"]}},
        bargap=0.18,
    )  # fmt: skip
    axis = {
        "gridcolor": GRID,
        "zeroline": False,
        "linecolor": HARVEST["border"],
        "ticks": "outside",
        "tickcolor": HARVEST["border"],
        "title": {"font": {"size": 12.5, "color": HARVEST["muted"]}, "standoff": 10},
        "automargin": True,
    }
    fig.update_xaxes(**axis, title_text=x)
    fig.update_yaxes(**axis, title_text=y)
    return json.loads(fig.to_json())


def _bins(values: np.ndarray) -> np.ndarray:
    """Freedman-Diaconis bin edges, kept between 6 and 40 bins."""
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.array([lo - 0.5, hi + 0.5])
    q1, q3 = np.percentile(values, [25, 75])
    width = 2 * (q3 - q1) / max(len(values), 1) ** (1 / 3)
    n = int(np.clip((hi - lo) / width, 6, 40)) if width > 0 else 12
    return np.linspace(lo, hi, n + 1)


def histogram(
    values: Sequence[float],
    *,
    title: str,
    x: str,
    noun: str,
    color: int = 1,
    markers: Sequence[tuple[float, str]] = (),
    log_note: str = "",
) -> dict:
    """Distribution with the middle half shaded and the median and mean marked.

    Long-tailed positive values (the 99th percentile over 20x the median, like revenue) are
    binned on a log scale so the bulk isn't squashed into one bar; other extreme values
    outside the 1st to 99th percentile are left out of the drawing and counted in the
    reading. `markers` adds labeled cutoffs, such as the blur threshold."""
    v = np.asarray([float(a) for a in values if a is not None], dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return finish(go.Figure(), title, x=x, y=f"Number of {noun}", subtitle="No values")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    mean = float(v.mean())
    log = bool((v > 0).all() and med > 0 and np.percentile(v, 99) > 20 * med)
    to_axis = np.log10 if log else (lambda a: a)
    data = to_axis(v)
    hidden_note = ""
    if not log:  # a few extreme values would squash everything else into one bar
        lo_cut = np.percentile(v, 1, method="higher")  # real values, not in-between ones
        hi_cut = np.percentile(v, 99, method="lower")
        spread = max(q3 - q1, 1e-9)
        if v.max() > q3 + 6 * spread or v.min() < q1 - 6 * spread:
            data = v[(v >= lo_cut) & (v <= hi_cut)]
            extra = len(v) - len(data)
            hidden_note = f"{extra} extreme {noun} outside {fmt(lo_cut)} to {fmt(hi_cut)} not drawn"
    counts, edges = np.histogram(data, bins=_bins(data))
    lo, hi = edges[:-1], edges[1:]
    real = (lambda a: 10**a) if log else (lambda a: a)
    fig = go.Figure(
        go.Bar(
            x=(lo + hi) / 2,
            y=counts,
            width=np.diff(edges) * 0.96,
            marker={"color": HARVEST["chart"][color], "line": {"width": 0}},
            customdata=np.stack([real(lo), real(hi)], axis=1),
            hovertemplate=f"%{{customdata[0]:,.3g}} to %{{customdata[1]:,.3g}}<br>%{{y}} {noun}"
            "<extra></extra>",
        )
    )
    fig.add_vrect(x0=to_axis(q1), x1=to_axis(q3), fillcolor=HARVEST["highlight"], opacity=0.14,
                  line_width=0, layer="below")  # fmt: skip
    fig.add_vline(x=to_axis(med), line={"color": HARVEST["text"], "width": 1.6, "dash": "dash"},
                  annotation_text=f"median {fmt(med)}", annotation_position="top right",
                  annotation_font={"size": 11, "color": HARVEST["text"]})  # fmt: skip
    m = float(to_axis(mean))
    if abs(m - to_axis(med)) > 0.03 * (edges[-1] - edges[0] or 1) and edges[0] <= m <= edges[-1]:
        fig.add_vline(x=m, line={"color": HARVEST["muted"], "width": 1.2, "dash": "dot"},
                      annotation_text=f"mean {fmt(mean)}", annotation_position="bottom right",
                      annotation_font={"size": 11, "color": HARVEST["muted"]})  # fmt: skip
    for at, label in markers:
        fig.add_vline(x=at, line={"color": HARVEST["critical"], "width": 1.6},
                      annotation_text=label, annotation_position="bottom left",
                      annotation_font={"size": 11, "color": HARVEST["critical"]})  # fmt: skip
    if log:
        powers = np.arange(np.floor(edges[0]), np.ceil(edges[-1]) + 1)
        fig.update_xaxes(tickvals=powers.tolist(), ticktext=[fmt(10**k) for k in powers])
    skew = "right-skewed" if mean > med * 1.15 and med > 0 else ""
    subtitle = (
        f"{len(v):,} {noun} · median {fmt(med)} · middle half {fmt(q1)} to {fmt(q3)} (shaded)"
        + (f" · {skew}" if skew else "")
        + (" · log scale" if log else "")
        + (f" · {log_note}" if log_note else "")
        + (f"<br>{hidden_note}" if hidden_note else "")
    )
    return finish(fig, title, x=x + (" (log scale)" if log else ""), y=f"Number of {noun}",
                  subtitle=subtitle)  # fmt: skip


def ranked_bars(
    labels: Sequence[Any],
    values: Sequence[float],
    *,
    title: str,
    x: str,
    y: str,
    color: int = 0,
    top: int = 15,
    total: float | None = None,
    percent: bool = False,
    reference: tuple[float, str] | None = None,
    subtitle: str = "",
) -> dict:
    """Horizontal bars, largest on top, with the value (and share of `total`) on each bar."""
    pairs = sorted(zip([str(a) for a in labels], values, strict=False), key=lambda p: -p[1])
    hidden = len(pairs) - top
    pairs = pairs[:top][::-1]
    names = [n if len(n) <= 28 else n[:26] + "…" for n, _ in pairs]
    vals = [float(v) for _, v in pairs]
    if percent:
        text = [f"{fmt(v)}%" for v in vals]
    elif total:
        text = [f"{fmt(v)} ({100 * v / total:.0f}%)" for v in vals]
    else:
        text = [fmt(v) for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            text=text,
            textposition="outside",
            cliponaxis=False,
            marker={"color": HARVEST["chart"][color]},
            customdata=[n for n, _ in pairs],
            hovertemplate="%{customdata}<br>%{text}<extra></extra>",
        )
    )
    if reference:
        fig.add_vline(x=reference[0], line={"color": HARVEST["text"], "width": 1.4, "dash": "dash"},
                      annotation_text=reference[1], annotation_position="top",
                      annotation_font={"size": 11, "color": HARVEST["text"]})  # fmt: skip
    if hidden > 0:
        subtitle = (subtitle + " · " if subtitle else "") + f"top {top} shown, {hidden} more"
    fig.update_xaxes(range=[0, max([*vals, reference[0] if reference else 0]) * 1.18 or 1])
    height = int(np.clip(130 + 30 * len(pairs), 300, 560))
    return finish(fig, title, x=x, y=y, subtitle=subtitle, height=height)


def class_balance(counts: dict[str, int], *, noun: str, what: str = "class") -> dict:
    """Items per class, with the even-split line and the largest-to-smallest ratio."""
    total = sum(counts.values())
    k = max(len(counts), 1)
    ratio = max(counts.values()) / max(min(counts.values()), 1) if counts else 1
    verdict = "balanced" if ratio < 1.5 else "imbalanced" if ratio < 3 else "heavily imbalanced"
    return ranked_bars(
        list(counts),
        list(counts.values()),
        title=f"{noun.capitalize()} per {what}",
        x=f"Number of {noun}",
        y=what.capitalize(),
        total=total,
        reference=(total / k, "even split"),
        subtitle=f"{k} {what}{'es' if what.endswith('s') else 's'} · {total:,} {noun} · "
        f"largest is {ratio:.1f}x the smallest ({verdict})",
    )


def grouped_bars(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    title: str,
    x: str,
    y: str,
    subtitle: str = "",
    y_range: tuple[float, float] | None = None,
) -> dict:
    fig = go.Figure(
        [
            go.Bar(
                name=name,
                x=list(categories),
                y=list(vals),
                text=[fmt(v) for v in vals],
                textposition="outside",
                cliponaxis=False,
                marker={"color": HARVEST["chart"][i % len(HARVEST["chart"])]},
                hovertemplate=f"{name}<br>%{{x}}: %{{y}}<extra></extra>",
            )
            for i, (name, vals) in enumerate(series.items())
        ]
    )
    fig.update_layout(barmode="group")
    if y_range:
        fig.update_yaxes(range=list(y_range))
    return finish(fig, title, x=x, y=y, subtitle=subtitle, legend=True)


def heatmap(
    z: np.ndarray,
    labels: Sequence[str],
    *,
    title: str,
    subtitle: str = "",
    x: str = "",
    y: str = "",
) -> dict:
    """A square matrix such as correlations (-1 to 1), with the value in each cell."""
    names = [n if len(n) <= 18 else n[:16] + "…" for n in labels]
    z = np.round(np.asarray(z, dtype=float), 2)
    z[np.triu_indices_from(z, k=1)] = np.nan  # each pair once
    scale = HARVEST["diverging"]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale=[[0, scale[0]], [0.5, scale[1]], [1, scale[2]]],
            text=[["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}" if len(names) <= 10 else "",
            hovertemplate="%{y} and %{x}: %{z}<extra></extra>",
            colorbar={"title": {"text": "r", "side": "top"}, "thickness": 12, "len": 0.8},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    size = int(np.clip(180 + 42 * len(names), 360, 620))
    return finish(fig, title, x=x or "Column", y=y or "Column", subtitle=subtitle, height=size)


def scatter(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    title: str,
    x: str,
    y: str,
    names: Sequence[str] = (),
    subtitle: str = "",
    color: int = 3,
) -> dict:
    fig = go.Figure(
        go.Scatter(
            x=list(xs),
            y=list(ys),
            mode="markers",
            text=list(names) or None,
            marker={"color": HARVEST["chart"][color], "size": 8, "opacity": 0.7,
                    "line": {"width": 1, "color": HARVEST["surface"]}},
            hovertemplate=("%{text}<br>" if names else "") + f"{x}: %{{x}}<br>{y}: %{{y}}"
            "<extra></extra>",
        )
    )  # fmt: skip
    return finish(fig, title, x=x, y=y, subtitle=subtitle)
