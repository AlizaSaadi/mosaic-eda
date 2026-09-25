"""Harvest palette (default) and the color-blind-safe palette, shared by charts and reports."""

from __future__ import annotations

HARVEST = {
    "bg": "#FBF6EE",
    "surface": "#FFFDF8",
    "border": "#E4D5BF",
    "text": "#2B1D14",
    "muted": "#6B5646",
    "accent": "#B5471B",
    "highlight": "#E8B04A",
    "critical": "#9E2B25",
    "warning": "#8A5A12",
    "info": "#3F6475",
    "ok": "#4F6B2F",
    "chart": [
        "#B5471B",
        "#2F6F73",
        "#A57A00",
        "#7A3E65",
        "#5E7F33",
        "#4E6E9E",
        "#B8604F",
        "#8A6440",
    ],
    "sequential": ["#FBF6EE", "#E8B04A", "#B5471B", "#2B1D14"],
    "diverging": ["#2F6F73", "#FBF6EE", "#B5471B"],
}

# Okabe-Ito based, adjusted for contrast on parchment
COLOR_BLIND_SAFE = {
    **HARVEST,
    "accent": "#C0470A",
    "chart": [
        "#C0470A",
        "#0072B2",
        "#A57A00",
        "#CC79A7",
        "#009E73",
        "#56B4E9",
        "#8A6440",
        "#2B1D14",
    ],
    "sequential": "cividis",
    "diverging": ["#0072B2", "#F7F7F7", "#E69F00"],
}
