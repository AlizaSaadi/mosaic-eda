"""MOSAIC EDA: Hugging Face Space entry point (Phase 0 placeholder)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from mosaic import __version__
from mosaic.config import get_settings
from mosaic.gpu import gpu_ready  # noqa: F401  (ZeroGPU needs a @spaces.GPU function at startup)
from mosaic.logging_setup import setup_logging


def build_app() -> gr.Blocks:
    settings = get_settings()
    key_status = "configured" if settings.has_gemini_key else "missing"
    with gr.Blocks(title="MOSAIC EDA") as demo:
        gr.Markdown(
            "# MOSAIC EDA\n"
            "**Multimodal Orchestrated System for Analysis, Inspection & Cleaning**\n\n"
            "A crew of AI agents that cleans, explores, and fact-checks any dataset: "
            "tables, text, images, audio, and video.\n\n"
            "This Space is under construction. The first working version will analyze CSV files."
        )
        gr.Markdown(f"Version {__version__} · Gemini key {key_status}")
    return demo


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    build_app().queue(default_concurrency_limit=get_settings().max_concurrent_jobs).launch()
