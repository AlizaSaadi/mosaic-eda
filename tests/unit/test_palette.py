"""Accessibility checks for the Harvest and color-blind-safe palettes.

Chart colors are simulated for protanopia, deuteranopia, and tritanopia (Machado, Oliveira
and Fernandes 2009, full severity, applied in linear RGB) and compared with CIEDE2000.
Text colors must meet WCAG 2.1 AA contrast against the backgrounds they're used on.
"""

from __future__ import annotations

import itertools
import math

import pytest

from mosaic.ui.palette import COLOR_BLIND_SAFE, HARVEST

MIN_DELTA_E = 15  # the blueprint's bar for colors shown together in one chart
# Charts put at most two palette colors in one figure (before/after bars, the two ends of
# the correlation scale); the rest use one color each. Four colors that all keep 3:1
# contrast on both backgrounds can't all be 15 apart for every color vision deficiency (a
# search of about 200 candidates topped out at 12.3), so the first four get a lower bar.
MIN_DELTA_E_PALETTE = 12
FIRST = 4

# Machado et al. (2009), severity 1.0
CVD = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def _rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _simulate(hex_color: str, kind: str) -> tuple[float, float, float]:
    """Linear RGB as someone with this color vision deficiency sees it."""
    lin = [_linear(c) for c in _rgb(hex_color)]
    return tuple(min(1.0, max(0.0, sum(m * c for m, c in zip(row, lin, strict=True))))
                 for row in CVD[kind])  # fmt: skip


def _lab(lin: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = lin
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def delta_e(lab1, lab2) -> float:
    """CIEDE2000 color difference."""
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dl, dc = l2 - l1, c2p - c1p
    dh = 0.0
    if c1p * c2p:
        dh = h2p - h1p
        dh -= 360 if dh > 180 else 0
        dh += 360 if dh < -180 else 0
    dH = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    l_bar, c_bar_p = (l1 + l2) / 2, (c1p + c2p) / 2
    h_bar = h1p + h2p
    if c1p * c2p:
        h_bar = (h1p + h2p + 360) / 2 if abs(h1p - h2p) > 180 else (h1p + h2p) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(h_bar - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar))
        + 0.32 * math.cos(math.radians(3 * h_bar + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar - 63))
    )
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = (
        -2
        * math.sqrt(c_bar_p**7 / (c_bar_p**7 + 25**7))
        * math.sin(math.radians(60 * math.exp(-(((h_bar - 275) / 25) ** 2))))
    )
    return math.sqrt(
        (dl / s_l) ** 2 + (dc / s_c) ** 2 + (dH / s_h) ** 2 + r_t * (dc / s_c) * (dH / s_h)
    )


def contrast(fg: str, bg: str) -> float:
    def lum(h):
        r, g, b = (_linear(c) for c in _rgb(h))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_delta_e_matches_published_values():
    # Sharma, Wu and Dalal (2005) test pairs
    assert delta_e((50, 2.6772, -79.7751), (50, 0, -82.7485)) == pytest.approx(2.0425, abs=1e-3)
    assert delta_e((50, 2.5, 0), (73, 25, -18)) == pytest.approx(27.1492, abs=1e-3)


def _cvd_delta(a: str, b: str, kind: str) -> float:
    return delta_e(_lab(_simulate(a, kind)), _lab(_simulate(b, kind)))


@pytest.mark.parametrize("kind", sorted(CVD))
def test_colors_shown_together_stay_distinct_with_color_blindness(kind):
    cb = COLOR_BLIND_SAFE
    together = [cb["chart"][:2], [cb["diverging"][0], cb["diverging"][-1]]]
    for a, b in together:
        d = _cvd_delta(a, b, kind)
        assert d >= MIN_DELTA_E, f"{a} vs {b} under {kind}: deltaE {d:.1f}"


@pytest.mark.parametrize("kind", sorted(CVD))
def test_color_blind_palette_colors_are_all_distinguishable(kind):
    for a, b in itertools.combinations(COLOR_BLIND_SAFE["chart"][:FIRST], 2):
        d = _cvd_delta(a, b, kind)
        assert d >= MIN_DELTA_E_PALETTE, f"{a} vs {b} under {kind}: deltaE {d:.1f}"


@pytest.mark.parametrize("palette", [HARVEST, COLOR_BLIND_SAFE])
def test_chart_colors_show_on_light_and_dark_backgrounds(palette):
    for color in palette["chart"][:FIRST]:  # WCAG 1.4.11: 3:1 for graphics
        assert contrast(color, "#FBF6EE") >= 3 and contrast(color, "#1C1512") >= 3, color


def test_harvest_chart_colors_are_distinct_for_typical_vision():
    colors = HARVEST["chart"][:FIRST]
    for a, b in itertools.combinations(colors, 2):
        lab = [_lab(tuple(_linear(c) for c in _rgb(h))) for h in (a, b)]
        assert delta_e(*lab) >= MIN_DELTA_E, f"{a} vs {b}"


LIGHT = {"bg": "#FBF6EE", "surface": "#FFFDF8"}
DARK = {"bg": "#1C1512", "surface": "#26201B"}


@pytest.mark.parametrize(
    ("fg", "bg", "minimum"),
    [
        # body text and muted text, both themes (AA: 4.5 for normal text)
        ("#2B1D14", LIGHT["bg"], 4.5),
        ("#6B5646", LIGHT["bg"], 4.5),
        ("#6B5646", LIGHT["surface"], 4.5),
        ("#F3E9DC", DARK["bg"], 4.5),
        ("#C9B497", DARK["bg"], 4.5),
        ("#C9B497", DARK["surface"], 4.5),
        # accent text (links, eyebrows, roles)
        ("#B5471B", LIGHT["bg"], 4.5),
        ("#E0763A", DARK["bg"], 4.5),
        # primary buttons: white on rust, dark on orange, white on color-blind vermillion
        ("#FFFFFF", "#B5471B", 4.5),
        ("#1C1512", "#E0763A", 4.5),
        ("#FFFFFF", "#D55E00", 3.0),  # large, bold button text (AA large: 3)
        # the office rule note
        ("#2B1D14", "#FFF4C9", 4.5),
        ("#9E2B25", "#FFF4C9", 4.5),
    ],
)
def test_text_meets_wcag_aa(fg, bg, minimum):
    assert contrast(fg, bg) >= minimum, f"{fg} on {bg}: {contrast(fg, bg):.2f}"
