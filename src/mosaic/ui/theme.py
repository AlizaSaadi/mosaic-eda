"""The Harvest theme for Gradio, plus the page head (fonts and the office script)."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

STATIC = Path(__file__).parent / "static"

RUST = gr.themes.Color(
    c50="#FBF1EB", c100="#F6DDD0", c200="#EDBBA1", c300="#E09470", c400="#CC6A40",
    c500="#B5471B", c600="#9C3C15", c700="#7F3112", c800="#63260E", c900="#4A1C0A",
    c950="#2E1106", name="rust",
)  # fmt: skip
PARCHMENT = gr.themes.Color(
    c50="#FFFDF8", c100="#FBF6EE", c200="#F3E9DC", c300="#E4D5BF", c400="#C9B497",
    c500="#A38C6E", c600="#6B5646", c700="#4E3D30", c800="#3A2C22", c900="#2B1D14",
    c950="#1C1512", name="parchment",
)  # fmt: skip


def harvest_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=RUST,
        secondary_hue=RUST,
        neutral_hue=PARCHMENT,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )
    return theme.set(
        body_background_fill="#FBF6EE",
        body_background_fill_dark="#1C1512",
        body_text_color="#2B1D14",
        body_text_color_dark="#F3E9DC",
        body_text_color_subdued="#6B5646",
        body_text_color_subdued_dark="#C9B497",
        background_fill_primary="#FFFDF8",
        background_fill_primary_dark="#26201B",
        background_fill_secondary="#FBF6EE",
        background_fill_secondary_dark="#1C1512",
        block_background_fill="#FFFDF8",
        block_background_fill_dark="#26201B",
        block_border_color="#E4D5BF",
        block_border_color_dark="#3A2C22",
        block_label_text_color="#6B5646",
        block_title_text_color="#2B1D14",
        border_color_primary="#E4D5BF",
        border_color_primary_dark="#3A2C22",
        button_primary_background_fill="#B5471B",
        button_primary_background_fill_hover="#9C3C15",
        button_primary_background_fill_dark="#E0763A",
        button_primary_background_fill_hover_dark="#CC6A40",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#1C1512",
        button_secondary_background_fill="#F3E9DC",
        button_secondary_background_fill_hover="#E4D5BF",
        button_secondary_background_fill_dark="#3A2C22",
        button_secondary_text_color="#2B1D14",
        button_secondary_text_color_dark="#F3E9DC",
        color_accent="#B5471B",
        color_accent_soft="#F6DDD0",
        color_accent_soft_dark="#4A1C0A",
        input_background_fill="#FFFDF8",
        input_background_fill_dark="#1C1512",
        input_border_color="#E4D5BF",
        input_border_color_focus="#B5471B",
        link_text_color="#B5471B",
        link_text_color_dark="#E0763A",
    )


def page_head() -> str:
    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;'
        '9..144,600&family=Inter:wght@400;500&display=swap" rel="stylesheet">'
    )
    script = (STATIC / "office.js").read_text(encoding="utf-8")
    return f"{fonts}<script>{script}</script>"


def css_paths() -> list[Path]:
    return [STATIC / "mosaic.css"]
